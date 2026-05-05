from __future__ import annotations

from app.config import settings
from app.retrieval.bm25 import BM25Index
from app.retrieval.vector_store import get_vector_store


def hybrid_search(query: str) -> list[dict]:
    """Simple hybrid retrieval.

    - Vector search is always available (offline hash embeddings).
    - BM25 is applied over the *vector* candidate pool for simplicity.
    """

    store = get_vector_store()
    vector_hits = store.search(query, top_k=settings.retrieval_vector_top_k)
    if not vector_hits:
        return []

    # BM25 rerank over the candidate pool.
    candidates = [h.get("text") or "" for h in vector_hits]
    bm25 = BM25Index(texts=candidates)
    bm25_hits = bm25.search(query, top_k=min(settings.retrieval_bm25_top_k, len(candidates)))
    bm25_by_idx = {idx: score for idx, score in bm25_hits}
    bm25_max = max(bm25_by_idx.values(), default=0.0)

    out: list[dict] = []
    for idx, hit in enumerate(vector_hits):
        vector_score = float(hit.get("score", 0.0))
        bm25_score = float(bm25_by_idx.get(idx, 0.0))
        bm25_norm = (bm25_score / bm25_max) if bm25_max > 0 else 0.0

        item = dict(hit)
        item["score"] = 0.7 * vector_score + 0.3 * bm25_norm
        out.append(item)

    out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return out[: settings.retrieval_top_k]
