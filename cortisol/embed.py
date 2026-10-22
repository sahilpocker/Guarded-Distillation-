"""One frozen sentence encoder, shared by the map space and the sbert student.

Every text is encoded once per process and cached, because students are refitted often and the
encoder never changes. Vectors are L2-normalised.
"""
from __future__ import annotations

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_encoders: dict[str, object] = {}
_cache: dict[tuple[str, str], np.ndarray] = {}


def encode(texts: list[str], model: str = MODEL) -> np.ndarray:
    todo = [t for t in dict.fromkeys(texts) if (model, t) not in _cache]
    if todo:
        if model not in _encoders:
            from sentence_transformers import SentenceTransformer
            _encoders[model] = SentenceTransformer(model)
        V = _encoders[model].encode(todo, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
        for t, v in zip(todo, V):
            _cache[(model, t)] = np.asarray(v, dtype=np.float32)
    if not len(texts):
        return np.zeros((0, 384), dtype=np.float32)
    return np.vstack([_cache[(model, t)] for t in texts])
