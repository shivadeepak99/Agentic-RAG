from __future__ import annotations

import math
import os
import threading

from app.config import settings
from app.observability.logger import get_logger


os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


_logger = get_logger("retrieval.reranker")
_reranker = None
_reranker_failed = False
_reranker_model_name = None
_reranker_lock = threading.Lock()


def _get_reranker():
    global _reranker, _reranker_failed, _reranker_model_name
    model_name = settings.retrieval_reranker_model
    if _reranker is not None and _reranker_model_name == model_name:
        return _reranker
    if _reranker_failed and _reranker_model_name == model_name:
        return None

    with _reranker_lock:
        if _reranker is not None and _reranker_model_name == model_name:
            return _reranker
        try:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(model_name)
            _reranker_model_name = model_name
            _reranker_failed = False
        except Exception as exc:
            _reranker = None
            _reranker_model_name = model_name
            _reranker_failed = True
            _logger.warning(
                "reranker.fallback_skip model=%s error=%s",
                model_name,
                exc,
            )
        return _reranker


def rerank(query: str, docs: list[dict], top_k: int | None = None) -> tuple[list[dict], bool]:
    if not query.strip() or not docs:
        return docs[: top_k or len(docs)], False

    model = _get_reranker()
    if model is None:
        return docs[: top_k or len(docs)], False

    pairs = [(query, str(d.get("text") or "")) for d in docs]
    scores = model.predict(pairs)
    rescored: list[dict] = []
    for doc, score in zip(docs, scores):
        item = dict(doc)
        raw = float(score)
        item["score"] = 1.0 / (1.0 + math.exp(-raw))
        rescored.append(item)

    rescored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    limit = top_k or len(rescored)
    return rescored[:limit], True


def reset_reranker() -> None:
    global _reranker, _reranker_failed, _reranker_model_name
    _reranker = None
    _reranker_failed = False
    _reranker_model_name = None
