from __future__ import annotations

from app.config import settings
from app.observability.logger import get_logger
from app.retrieval.bm25 import BM25Index
from app.retrieval.lexical import get_lexical_corpus
from app.retrieval.reranker import rerank
from app.retrieval.vector_store import get_vector_store


_RRF_K = 60.0
_logger = get_logger("retrieval.hybrid")


def vector_only_search(query: str, top_k: int | None = None) -> list[dict]:
    store = get_vector_store()
    return store.search(query, top_k=top_k or settings.retrieval_top_k)


def _vector_candidates(query: str, top_k: int | None = None) -> list[dict]:
    return vector_only_search(query, top_k=top_k or settings.retrieval_hybrid_vector_k)


def _bm25_candidates(query: str, top_k: int | None = None) -> list[dict]:
    corpus = get_lexical_corpus()
    return corpus.search(query, top_k=top_k or settings.retrieval_hybrid_bm25_k)


def _rank_fuse(*ranked_lists: list[dict], top_k: int | None = None) -> list[dict]:
    fused: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            key = str(hit.get("id") or f"anon-{rank}")
            item = fused.get(key)
            if item is None:
                item = dict(hit)
                item["score"] = 0.0
                fused[key] = item
            item["score"] += 1.0 / (_RRF_K + rank)
            if not item.get("text") and hit.get("text"):
                item["text"] = hit["text"]
            if item.get("source") in (None, "", "unknown") and hit.get("source"):
                item["source"] = hit["source"]

    out = list(fused.values())
    out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return out[: top_k or len(out)]


def lightweight_hybrid_search(query: str) -> list[dict]:
    """Legacy lightweight hybrid: BM25 reranks vector hits only."""
    vector_hits = _vector_candidates(query, top_k=settings.retrieval_vector_top_k)
    if not vector_hits:
        return []

    bm25 = BM25Index(texts=[str(hit.get("text") or "") for hit in vector_hits])
    bm25_hits = bm25.search(query, top_k=min(settings.retrieval_bm25_top_k, len(vector_hits)))
    bm25_by_idx = {idx: float(score) for idx, score in bm25_hits}
    bm25_max = max(bm25_by_idx.values(), default=0.0)

    out: list[dict] = []
    for idx, hit in enumerate(vector_hits):
        item = dict(hit)
        bm25_score = bm25_by_idx.get(idx, 0.0)
        bm25_norm = (bm25_score / bm25_max) if bm25_max > 0 else 0.0
        item["score"] = 0.7 * float(hit.get("score", 0.0)) + 0.3 * bm25_norm
        out.append(item)

    out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return out[: settings.retrieval_top_k]


def true_hybrid_search(query: str, use_reranker: bool = True) -> list[dict]:
    if not query.strip():
        return []

    vector_hits = _vector_candidates(query)
    bm25_hits = _bm25_candidates(query)
    if not vector_hits and not bm25_hits:
        return []

    fused = _rank_fuse(
        vector_hits,
        bm25_hits,
        top_k=settings.retrieval_hybrid_fused_k,
    )
    fused_top_score = float(fused[0]["score"]) if fused else 0.0

    reranked = fused
    reranker_used = False
    if use_reranker:
        reranked, reranker_used = rerank(
            query,
            fused,
            top_k=settings.retrieval_reranker_top_k,
        )

    if not use_reranker:
        reranked = fused[: settings.retrieval_top_k]
    elif reranker_used and settings.retrieval_reranker_top_k != settings.retrieval_top_k:
        reranked = reranked[: settings.retrieval_top_k]
    elif not reranker_used:
        reranked = fused[: settings.retrieval_top_k]

    final_top_score = float(reranked[0]["score"]) if reranked else 0.0
    _logger.info(
        "hybrid.search query=%r vector_n=%d bm25_n=%d fused_n=%d reranker=%s fused_top=%.3f final_top=%.3f",
        query,
        len(vector_hits),
        len(bm25_hits),
        len(fused),
        "used" if reranker_used else ("disabled" if not use_reranker else "skipped"),
        fused_top_score,
        final_top_score,
    )
    return reranked[: settings.retrieval_top_k]


def hybrid_search(query: str) -> list[dict]:
    mode = (settings.retrieval_mode or "lightweight_hybrid").strip().lower()

    if mode in {"vector", "vector_only", "vector-only"}:
        return vector_only_search(query)

    if mode in {"lightweight_hybrid", "lightweight-hybrid", "light_hybrid"}:
        return lightweight_hybrid_search(query)

    if mode in {"true_hybrid", "true-hybrid", "hybrid"}:
        return true_hybrid_search(query, use_reranker=settings.retrieval_use_reranker)

    _logger.warning(
        "hybrid.unknown_mode mode=%r fallback=%r",
        settings.retrieval_mode,
        "lightweight_hybrid",
    )
    return lightweight_hybrid_search(query)
