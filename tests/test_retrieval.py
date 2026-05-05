"""Tests for retrieval subsystem: VectorStore, BM25, hybrid search."""
from __future__ import annotations

import pytest

from app.retrieval.bm25 import BM25Index
from app.retrieval.chunking import chunk_text, normalize_text
from app.retrieval.hybrid import (
    _rank_fuse,
    hybrid_search,
    lightweight_hybrid_search,
    true_hybrid_search,
)
from app.retrieval.lexical import get_lexical_corpus
from app.retrieval.reranker import rerank
from app.retrieval.vector_store import get_vector_store
from tests.conftest import seed_store


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class TestVectorStore:
    def test_add_and_search_returns_hits(self):
        store = get_vector_store()
        store.add_texts(
            texts=["Retrieval augmented generation augments LLMs with external context."],
            metadatas=[{"source": "test"}],
            ids=["rag-doc"],
        )
        hits = store.search("retrieval augmented generation", top_k=1)
        assert len(hits) == 1
        assert hits[0]["text"] != ""

    def test_search_returns_score_between_0_and_1(self):
        store = get_vector_store()
        store.add_texts(["Some document about transformers."], metadatas=[{"src": "t"}], ids=["t0"])
        hits = store.search("transformer attention", top_k=1)
        if hits:
            assert 0.0 <= hits[0]["score"] <= 1.0

    def test_empty_query_returns_empty(self):
        store = get_vector_store()
        store.add_texts(["Some text."], metadatas=[{"src": "d"}], ids=["d0"])
        hits = store.search("", top_k=5)
        assert hits == []

    def test_top_k_respected(self):
        store = get_vector_store()
        texts = [f"Document number {i} about machine learning." for i in range(10)]
        metas = [{"src": f"doc{i}"} for i in range(10)]
        ids = [f"doc{i}" for i in range(10)]
        store.add_texts(texts=texts, metadatas=metas, ids=ids)
        hits = store.search("machine learning", top_k=3)
        assert len(hits) <= 3

    def test_duplicate_ids_dont_raise(self):
        store = get_vector_store()
        store.add_texts(["text one"], metadatas=[{"src": "x"}], ids=["dup"])
        # Adding same id again should not crash (Chroma upserts).
        store.add_texts(["text two"], metadatas=[{"src": "x"}], ids=["dup"])

    def test_metadata_preserved_in_source(self):
        store = get_vector_store()
        store.add_texts(
            texts=["About neural networks"],
            metadatas=[{"source": "arxiv/2401.00001"}],
            ids=["nn-doc"],
        )
        hits = store.search("neural networks", top_k=1)
        if hits:
            assert hits[0]["source"] == "arxiv/2401.00001"


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class TestBM25:
    def test_returns_correct_n_results(self):
        texts = ["alpha beta gamma", "delta epsilon zeta", "theta iota kappa"]
        idx = BM25Index(texts=texts)
        results = idx.search("alpha gamma", top_k=2)
        assert len(results) <= 2

    def test_relevant_doc_scores_higher(self):
        texts = [
            "attention mechanism in transformers",
            "banana smoothie recipe",
            "self-attention scaled dot product",
        ]
        idx = BM25Index(texts=texts)
        results = idx.search("attention transformer", top_k=3)
        indices = [r[0] for r in results]
        assert 0 in indices or 2 in indices  # attention docs ranked

    def test_empty_corpus(self):
        idx = BM25Index(texts=[])
        results = idx.search("anything", top_k=5)
        assert results == []

    def test_scores_are_non_negative(self):
        idx = BM25Index(texts=["word1 word2", "word3 word4"])
        for _, score in idx.search("word1", top_k=2):
            assert score >= 0.0


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

class TestChunking:
    def test_short_text_is_single_chunk(self):
        chunks = chunk_text("Short text.", chunk_size=900, overlap=150)
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_long_text_produces_multiple_chunks(self):
        text = "word " * 500  # 2500 chars
        chunks = chunk_text(text, chunk_size=900, overlap=150)
        assert len(chunks) > 1

    def test_overlap_means_content_shared(self):
        text = "A " * 600  # 1200 chars
        chunks = chunk_text(text, chunk_size=900, overlap=150)
        if len(chunks) >= 2:
            # Last 150 chars of chunk 0 appear at start of chunk 1
            tail = chunks[0][-150:]
            head = chunks[1][:150]
            assert tail.strip() == head.strip()

    def test_normalize_collapses_whitespace(self):
        text = "hello\xa0world   foo"
        assert normalize_text(text) == "hello world foo"

    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------

class TestHybridSearch:
    def test_returns_hits_when_corpus_populated(self):
        seed_store(
            texts=["RAG combines retrieval with generation.", "BM25 is keyword retrieval."],
            sources=["paper-a", "paper-b"],
        )
        hits = hybrid_search("retrieval generation")
        assert hits
        assert any("RAG" in h["text"] for h in hits)

    def test_empty_corpus_returns_empty(self):
        hits = hybrid_search("anything at all")
        assert hits == []

    def test_top_k_is_bounded(self):
        texts = [f"Document {i} on deep learning architectures." for i in range(20)]
        seed_store(texts=texts)
        hits = hybrid_search("deep learning")
        from app.config import settings
        assert len(hits) <= settings.retrieval_top_k

    def test_scores_are_positive(self):
        seed_store(texts=["Neural networks are universal approximators."])
        hits = hybrid_search("neural networks")
        for h in hits:
            assert h["score"] > 0

    def test_results_sorted_descending(self):
        seed_store(texts=[
            "Attention is all you need — transformers paper.",
            "Random text about cooking.",
            "Transformers use self-attention across tokens.",
        ])
        hits = hybrid_search("transformer attention")
        scores = [h["score"] for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_rank_fusion_dedupes_shared_ids(self):
        fused = _rank_fuse(
            [
                {"id": "shared", "text": "alpha", "source": "vec", "score": 0.8},
                {"id": "vec-only", "text": "beta", "source": "vec", "score": 0.7},
            ],
            [
                {"id": "shared", "text": "alpha", "source": "bm25", "score": 2.0},
                {"id": "bm25-only", "text": "gamma", "source": "bm25", "score": 1.8},
            ],
        )
        ids = [h["id"] for h in fused]
        assert ids.count("shared") == 1
        assert "vec-only" in ids
        assert "bm25-only" in ids

    def test_true_hybrid_can_return_bm25_only_candidate(self, monkeypatch):
        monkeypatch.setattr(
            "app.retrieval.hybrid._vector_candidates",
            lambda query, top_k=None: [{"id": "v1", "text": "vector hit", "source": "vec", "score": 0.9}],
        )
        monkeypatch.setattr(
            "app.retrieval.hybrid._bm25_candidates",
            lambda query, top_k=None: [{"id": "b1", "text": "keyword exact hit", "source": "bm25", "score": 4.0}],
        )
        monkeypatch.setattr(
            "app.retrieval.hybrid.rerank",
            lambda query, docs, top_k=None: (docs[: top_k or len(docs)], False),
        )
        hits = true_hybrid_search("keyword hit", use_reranker=False)
        ids = [h["id"] for h in hits]
        assert "b1" in ids
        assert "v1" in ids

    def test_true_hybrid_can_return_vector_only_candidate(self, monkeypatch):
        monkeypatch.setattr(
            "app.retrieval.hybrid._vector_candidates",
            lambda query, top_k=None: [{"id": "v1", "text": "semantic hit", "source": "vec", "score": 0.9}],
        )
        monkeypatch.setattr(
            "app.retrieval.hybrid._bm25_candidates",
            lambda query, top_k=None: [],
        )
        monkeypatch.setattr(
            "app.retrieval.hybrid.rerank",
            lambda query, docs, top_k=None: (docs[: top_k or len(docs)], False),
        )
        hits = true_hybrid_search("semantic query", use_reranker=False)
        assert [h["id"] for h in hits] == ["v1"]

    def test_reranker_rescores_fused_candidates(self, monkeypatch):
        monkeypatch.setattr(
            "app.retrieval.hybrid._vector_candidates",
            lambda query, top_k=None: [
                {"id": "a", "text": "less relevant", "source": "vec", "score": 0.9},
                {"id": "b", "text": "more relevant", "source": "vec", "score": 0.8},
            ],
        )
        monkeypatch.setattr("app.retrieval.hybrid._bm25_candidates", lambda query, top_k=None: [])
        monkeypatch.setattr(
            "app.retrieval.hybrid.rerank",
            lambda query, docs, top_k=None: (
                [
                    {**docs[1], "score": 2.0},
                    {**docs[0], "score": 1.0},
                ][: top_k or len(docs)],
                True,
            ),
        )
        hits = true_hybrid_search("rerank me", use_reranker=True)
        assert [h["id"] for h in hits[:2]] == ["b", "a"]

    def test_dispatch_defaults_to_lightweight_hybrid(self, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "retrieval_mode", "lightweight_hybrid")
        monkeypatch.setattr(
            "app.retrieval.hybrid.lightweight_hybrid_search",
            lambda query: [{"id": "lw", "text": "lightweight", "source": "seed", "score": 1.0}],
        )
        hits = hybrid_search("rag")
        assert hits[0]["id"] == "lw"

    def test_dispatch_can_use_vector_only(self, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "retrieval_mode", "vector_only")
        monkeypatch.setattr(
            "app.retrieval.hybrid.vector_only_search",
            lambda query: [{"id": "vec", "text": "vector", "source": "seed", "score": 1.0}],
        )
        hits = hybrid_search("rag")
        assert hits[0]["id"] == "vec"

    def test_dispatch_can_use_true_hybrid_without_reranker(self, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "retrieval_mode", "true_hybrid")
        monkeypatch.setattr(settings, "retrieval_use_reranker", False)

        def _fake_true_hybrid(query: str, use_reranker: bool = True) -> list[dict]:
            assert use_reranker is False
            return [{"id": "true", "text": "true hybrid", "source": "seed", "score": 1.0}]

        monkeypatch.setattr("app.retrieval.hybrid.true_hybrid_search", _fake_true_hybrid)
        hits = hybrid_search("rag")
        assert hits[0]["id"] == "true"


class TestLexicalCorpus:
    def test_uses_vector_store_fallback_when_chunks_absent(self):
        seed_store(texts=["Exact keyword match lives in vector store fallback."])
        hits = get_lexical_corpus().search("keyword match", top_k=1)
        assert hits
        assert "keyword" in hits[0]["text"].lower()


class TestReranker:
    def test_fallback_returns_original_docs(self, monkeypatch):
        docs = [{"id": "a", "text": "alpha", "source": "seed", "score": 0.5}]
        monkeypatch.setattr("app.retrieval.reranker._get_reranker", lambda: None)
        out, used = rerank("alpha", docs, top_k=1)
        assert used is False
        assert out == docs

    def test_model_scores_docs_when_available(self, monkeypatch):
        class FakeModel:
            def predict(self, pairs):
                assert len(pairs) == 2
                return [0.1, 0.9]

        docs = [
            {"id": "a", "text": "alpha", "source": "seed", "score": 0.5},
            {"id": "b", "text": "beta", "source": "seed", "score": 0.4},
        ]
        monkeypatch.setattr("app.retrieval.reranker._get_reranker", lambda: FakeModel())
        out, used = rerank("beta", docs, top_k=2)
        assert used is True
        assert [d["id"] for d in out] == ["b", "a"]
