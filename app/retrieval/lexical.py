from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.config import settings
from app.observability.logger import get_logger
from app.retrieval.bm25 import BM25Index
from app.retrieval.vector_store import get_vector_store


_logger = get_logger("retrieval.lexical")


@dataclass
class LexicalCorpus:
    records: list[dict]

    def __post_init__(self) -> None:
        texts = [str(r.get("text") or "") for r in self.records]
        self._bm25 = BM25Index(texts=texts)

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        hits = self._bm25.search(query, top_k=top_k)
        out: list[dict] = []
        for idx, score in hits:
            if idx >= len(self.records):
                continue
            rec = self.records[idx]
            out.append(
                {
                    "id": str(rec.get("id") or f"lex-{idx}"),
                    "text": str(rec.get("text") or ""),
                    "source": str(rec.get("source") or "lexical"),
                    "score": float(score),
                }
            )
        return out


_corpus: LexicalCorpus | None = None
_corpus_key: tuple[str, str] | None = None


def _load_records_from_chunks(chunks_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(chunks_dir.glob("*.chunks.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            _logger.warning("lexical.skip_chunk_file path=%s error=%s", path, exc)
            continue
        for rec in data:
            text = str(rec.get("text") or "")
            if not text.strip():
                continue
            records.append(
                {
                    "id": str(rec.get("id") or f"{path.stem}:{len(records)}"),
                    "text": text,
                    "source": str(rec.get("source") or "chunk"),
                }
            )
    return records


def _load_records() -> list[dict]:
    records = _load_records_from_chunks(settings.chunks_dir)
    if records:
        return records

    fallback = get_vector_store().all_docs()
    if fallback:
        _logger.info("lexical.using_vector_store_fallback n_docs=%d", len(fallback))
    return fallback


def get_lexical_corpus() -> LexicalCorpus:
    global _corpus, _corpus_key
    key = (str(settings.chunks_dir.resolve()), str(settings.chroma_persist_dir.resolve()))
    if _corpus is None or _corpus_key != key:
        records = _load_records()
        _corpus = LexicalCorpus(records=records)
        _corpus_key = key
    return _corpus


def reset_lexical_corpus() -> None:
    global _corpus, _corpus_key
    _corpus = None
    _corpus_key = None
