from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from app.config import settings
from app.observability.logger import get_logger
from app.retrieval.embeddings import embed_text, embed_texts, embedding_dim, is_semantic


_CHROMA_ADD_BATCH_SIZE = 1000
_logger = get_logger("retrieval.vector_store")


@dataclass
class VectorStore:
    persist_dir: Path
    collection_name: str

    def __post_init__(self) -> None:
        # Use a collection suffix tied to the embedding model so a model
        # change does not silently produce dimension-mismatch errors against
        # an existing Chroma collection.
        suffix = "semantic" if is_semantic() else "hash"
        self._effective_name = f"{self.collection_name}_{suffix}"

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self._effective_name,
                metadata={"embed_dim": embedding_dim()},
            )
        except Exception as exc:
            self._client = None
            self._collection = None
            self._docs: list[dict] = []
            _logger.warning(
                "vector_store.fallback_in_memory persist_dir=%s collection=%s error=%s",
                self.persist_dir,
                self._effective_name,
                exc,
            )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Iterable[dict] | None = None,
        ids: Iterable[str] | None = None,
    ) -> None:
        texts_list = list(texts)
        if not texts_list:
            return

        # Chroma requires non-empty metadata dicts; ensure at least one key.
        raw_metas = list(metadatas) if metadatas is not None else [{} for _ in texts_list]
        metas_list = [m if m else {"_src": "unknown"} for m in raw_metas]
        ids_list = list(ids) if ids is not None else [f"doc-{i}" for i in range(len(texts_list))]
        if self._collection is not None:
            # Chroma enforces a maximum add() batch size; keep inserts bounded.
            for start in range(0, len(texts_list), _CHROMA_ADD_BATCH_SIZE):
                end = min(len(texts_list), start + _CHROMA_ADD_BATCH_SIZE)
                batch_texts = texts_list[start:end]
                batch_metas = metas_list[start:end]
                batch_ids = ids_list[start:end]
                batch_embs = embed_texts(batch_texts)
                self._collection.add(
                    documents=batch_texts,
                    metadatas=batch_metas,
                    ids=batch_ids,
                    embeddings=batch_embs,
                )
            return

        embeddings = embed_texts(texts_list)

        for i, t in enumerate(texts_list):
            self._docs.append(
                {"id": ids_list[i], "text": t, "meta": metas_list[i], "emb": embeddings[i]}
            )

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        if not query.strip():
            return []

        qemb = embed_text(query)

        if self._collection is not None:
            res = self._collection.query(
                query_embeddings=[qemb],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            out: list[dict] = []
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            distances = res.get("distances", [[]])[0]
            for i, text in enumerate(docs):
                meta = metas[i] if i < len(metas) else {}
                dist = distances[i] if i < len(distances) else 0.0
                score = float(1.0 / (1.0 + dist))
                out.append(
                    {
                        "id": str(meta.get("id") or f"chroma-{i}"),
                        "text": text,
                        "source": str(meta.get("source") or "chroma"),
                        "score": score,
                    }
                )
            return out

        # Fallback cosine similarity search.
        out = []
        for d in getattr(self, "_docs", []):
            score = float(np.dot(np.array(qemb), np.array(d["emb"])))
            out.append(
                {
                    "id": d["id"],
                    "text": d["text"],
                    "source": str(d["meta"].get("source", "mem")),
                    "score": score,
                }
            )
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:top_k]


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        _store = VectorStore(settings.chroma_persist_dir, settings.chroma_collection)
    return _store


def reset_vector_store() -> None:
    """Clear the cached singleton (useful for tests / re-ingestion)."""
    global _store
    _store = None
