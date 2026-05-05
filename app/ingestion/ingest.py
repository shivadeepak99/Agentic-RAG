from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.retrieval.vector_store import get_vector_store


def ingest_chunks(chunk_files: list[Path]) -> int:
    store = get_vector_store()

    texts: list[str] = []
    metas: list[dict] = []
    ids: list[str] = []

    for p in chunk_files:
        records = json.loads(p.read_text(encoding="utf-8"))
        for r in records:
            ids.append(r["id"])
            texts.append(r["text"])
            metas.append({"id": r["id"], "source": r.get("source", "unknown")})

    store.add_texts(texts=texts, metadatas=metas, ids=ids)
    return len(texts)


def ingest_default_chunks_dir() -> int:
    chunk_files = sorted(settings.chunks_dir.glob("*.chunks.json"))
    return ingest_chunks(chunk_files)
