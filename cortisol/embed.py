"""One frozen sentence encoder, shared by the map space and the sbert student.

Every text is encoded once per process and cached, because students are refitted often and the
encoder never changes. Vectors are L2-normalised.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DISK = Path("data/embeddings.npz")     # optional on-disk copy of the cache; written by `save()`

_encoders: dict[str, object] = {}
_cache: dict[tuple[str, str], np.ndarray] = {}
_disk_loaded = False


def _load_disk() -> None:
    global _disk_loaded
    _disk_loaded = True
    if DISK.exists():
        z = np.load(DISK, allow_pickle=False)
        if str(z["model"]) == MODEL:
            for t, v in zip(z["texts"], z["vectors"]):
                _cache[(MODEL, str(t))] = v


def save() -> None:
    """Persist the cache so other processes and later runs skip the encoder."""
    items = [(t, v) for (m, t), v in _cache.items() if m == MODEL]
    DISK.parent.mkdir(parents=True, exist_ok=True)
    tmp = DISK.with_name(DISK.name + ".tmp.npz")
    np.savez(tmp, model=MODEL, texts=np.array([t for t, _ in items]), vectors=np.vstack([v for _, v in items]))
    tmp.replace(DISK)


def encode(texts: list[str], model: str = MODEL) -> np.ndarray:
    if not _disk_loaded:
        _load_disk()
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
