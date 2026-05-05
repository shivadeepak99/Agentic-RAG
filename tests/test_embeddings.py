"""Tests for the embedding module (embeddings.py)."""
from __future__ import annotations

import math

from app.retrieval.embeddings import _hash_embed, embed_text, embed_texts, embedding_dim


class TestHashEmbed:
    def test_returns_correct_dim(self):
        vec = _hash_embed("hello world", dim=128)
        assert len(vec) == 128

    def test_is_normalized(self):
        vec = _hash_embed("normalize me")
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-5

    def test_deterministic(self):
        a = _hash_embed("same text")
        b = _hash_embed("same text")
        assert a == b

    def test_different_texts_differ(self):
        a = _hash_embed("text one")
        b = _hash_embed("text two entirely different")
        assert a != b

    def test_custom_dim(self):
        for dim in (32, 64, 256):
            vec = _hash_embed("dim test", dim=dim)
            assert len(vec) == dim


class TestEmbedText:
    def test_returns_list_of_floats(self):
        vec = embed_text("test sentence")
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)
        assert len(vec) > 0

    def test_consistent_dim_with_embedding_dim(self):
        vec = embed_text("dim check")
        assert len(vec) == embedding_dim()

    def test_is_normalized(self):
        vec = embed_text("normalization check")
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-4


class TestEmbedTexts:
    def test_returns_one_vec_per_text(self):
        texts = ["first", "second", "third"]
        vecs = embed_texts(texts)
        assert len(vecs) == len(texts)

    def test_empty_input(self):
        vecs = embed_texts([])
        assert vecs == []

    def test_single_text(self):
        vecs = embed_texts(["only one"])
        assert len(vecs) == 1
        assert len(vecs[0]) == embedding_dim()

    def test_all_normalized(self):
        texts = ["alpha", "beta", "gamma delta epsilon"]
        for vec in embed_texts(texts):
            norm = math.sqrt(sum(x * x for x in vec))
            assert abs(norm - 1.0) < 1e-4

    def test_batch_equals_single(self):
        texts = ["foo bar baz", "qux quux corge"]
        single = [embed_text(t) for t in texts]
        batch = embed_texts(texts)
        for s, b in zip(single, batch):
            # Expect identical values (same model call)
            assert s == b
