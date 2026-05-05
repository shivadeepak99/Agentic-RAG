from __future__ import annotations

import re
from dataclasses import dataclass

from app.observability.logger import get_logger


_logger = get_logger("retrieval.bm25")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


@dataclass
class BM25Index:
    texts: list[str]

    def __post_init__(self) -> None:
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([_tokenize(t) for t in self.texts])
        except Exception as exc:
            self._bm25 = None
            _logger.warning("bm25.fallback_overlap error=%s", exc)

    def search(self, query: str, top_k: int = 8) -> list[tuple[int, float]]:
        if not query.strip() or not self.texts:
            return []

        q = _tokenize(query)
        if self._bm25 is None:
            # naive term-overlap scoring
            qset = set(q)
            scored = [(i, float(len(qset.intersection(set(_tokenize(t)))))) for i, t in enumerate(self.texts)]
        else:
            scores = self._bm25.get_scores(q)
            scored = [(i, float(scores[i])) for i in range(len(scores))]

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
