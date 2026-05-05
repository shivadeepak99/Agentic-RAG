"""Shared fixtures for the test suite."""
from __future__ import annotations

import pytest
from pathlib import Path

from app.config import settings
from app.retrieval.lexical import reset_lexical_corpus
from app.retrieval.reranker import reset_reranker
import app.retrieval.vector_store as vs
from app.memory.store import reset_session
from app.llm.client import reset_llm_client


@pytest.fixture(autouse=True)
def isolated_vector_store(tmp_path: Path):
    """Each test gets its own Chroma directory so stores don't bleed."""
    settings.chroma_persist_dir = tmp_path / "chroma"
    settings.chunks_dir = tmp_path / "chunks"
    vs._store = None
    reset_lexical_corpus()
    reset_reranker()
    yield
    vs._store = None
    reset_lexical_corpus()
    reset_reranker()


@pytest.fixture(autouse=True)
def clean_sessions():
    """Reset memory sessions between tests."""
    reset_session("test")
    reset_session("default")
    reset_session("cli")
    yield
    reset_session("test")
    reset_session("default")
    reset_session("cli")


@pytest.fixture(autouse=True)
def clean_llm_client():
    """Reset LLM client singleton so tests can patch api_key cleanly."""
    reset_llm_client()
    yield
    reset_llm_client()


def seed_store(texts: list[str], sources: list[str] | None = None) -> None:
    """Insert chunks into the in-test vector store."""
    store = vs.get_vector_store()
    metas = [{"id": f"t{i}", "source": sources[i] if sources else "seed"} for i in range(len(texts))]
    ids = [f"t{i}" for i in range(len(texts))]
    store.add_texts(texts=texts, metadatas=metas, ids=ids)
    reset_lexical_corpus()
