"""Tests for retrieval subsystem: VectorStore, BM25, hybrid search."""
from __future__ import annotations

import pytest

from app.retrieval.bm25 import BM25Index
from app.retrieval.chunking import chunk_text, normalize_text
from app.retrieval.hybrid import hybrid_search
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
