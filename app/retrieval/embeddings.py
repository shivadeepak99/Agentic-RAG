from __future__ import annotations

import hashlib
import os
import threading

import numpy as np

from app.observability.logger import get_logger


# SentenceTransformers uses HuggingFace Transformers. If TensorFlow is installed in the
# environment, Transformers may import it during framework detection, which can produce
# noisy logs and (in some envs) protobuf compatibility errors. This project uses the
# PyTorch path; disable TF/Flax backends proactively.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


_HASH_DIM = 128
_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_model = None
_model_lock = threading.Lock()
_use_semantic = True
_logger = get_logger("retrieval.embeddings")


def _hash_embed(text: str, dim: int = _HASH_DIM) -> list[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    data = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
    vec = np.tile(data, int(np.ceil(dim / len(data))))[:dim]
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec.tolist()


def _get_model():
    global _model, _use_semantic
    if _model is not None or not _use_semantic:
        return _model

    with _model_lock:
        if _model is not None or not _use_semantic:
            return _model
        try:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(_MODEL_NAME)
        except Exception as exc:
            _use_semantic = False
            _model = None
            _logger.warning(
                "embeddings.hash_fallback model=%s error=%s",
                _MODEL_NAME,
                exc,
            )
        return _model


def embed_text(text: str) -> list[float]:
    model = _get_model()
    if model is None:
        return _hash_embed(text)
    vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    if model is None:
        return [_hash_embed(t) for t in texts]
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]


def embedding_dim() -> int:
    model = _get_model()
    if model is None:
        return _HASH_DIM
    try:
        return int(model.get_sentence_embedding_dimension())
    except Exception:
        return _HASH_DIM


def is_semantic() -> bool:
    _get_model()
    return _use_semantic and _model is not None
