"""The map space: where local cortisol lives.

Queries are embedded, then grouped into cells (k-means). Each cell holds its own cortisol level,
and a kernel between cell centroids lets stress spread to neighbouring cells (diffusion).

The map is fitted once on the warmup traffic and then frozen, so cortisol in a cell means the
same region of query space for the whole replay. Students are refitted often; the map is not.

Backends:
  tfidf  character n-gram TF-IDF -> 128-d LSA. No downloads; works anywhere. Default.
  sbert  a sentence-transformers encoder (pip install sentence-transformers). Better geometry,
         especially for queries with unseen words; use it on the A5000 or the Mac.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from . import embed


class MapSpace:
    def __init__(self, n_cells: int = 40, backend: str = "tfidf", dim: int = 128,
                 sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 kernel_tau: float = 0.08, kernel_k: int = 5, seed: int = 0):
        self.n_cells, self.backend, self.dim = n_cells, backend, dim
        self.sbert_model, self.kernel_tau, self.kernel_k, self.seed = sbert_model, kernel_tau, kernel_k, seed

    # -- embedding
    def _fit_embed(self, texts: list[str]) -> None:
        if self.backend == "tfidf":
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
            X = self.vec.fit_transform(texts)
            self.svd = TruncatedSVD(n_components=min(self.dim, X.shape[1] - 1), random_state=self.seed).fit(X)
        elif self.backend == "sbert":
            pass   # the shared frozen encoder (guarded_distillation.embed) is loaded on first use
        else:
            raise ValueError(f"unknown map backend {self.backend}")

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.backend == "tfidf":
            Z = self.svd.transform(self.vec.transform(texts))
        else:
            Z = embed.encode(list(texts), self.sbert_model)
        return normalize(np.asarray(Z, dtype=np.float64))

    # -- cells and kernel
    def fit(self, texts: list[str]) -> "MapSpace":
        self._fit_embed(texts)
        Z = self.embed(texts)
        k = min(self.n_cells, len(texts) // 10)
        self.km = KMeans(n_clusters=k, n_init=4, random_state=self.seed).fit(Z)
        self.centroids = normalize(self.km.cluster_centers_)
        self.n_cells = k
        S = self.centroids @ self.centroids.T
        W = np.exp((S - 1.0) / self.kernel_tau)
        # keep each cell's k nearest neighbours (and itself); symmetrise
        keep = np.zeros_like(W, dtype=bool)
        nn = np.argsort(-S, axis=1)[:, : self.kernel_k + 1]
        np.put_along_axis(keep, nn, True, axis=1)
        keep |= keep.T
        self.W = np.where(keep, W, 0.0)
        np.fill_diagonal(self.W, 1.0)
        self.order = self._spectral_order()
        return self

    def cell(self, Z: np.ndarray) -> np.ndarray:
        return np.argmax(Z @ self.centroids.T, axis=1)

    def _spectral_order(self) -> np.ndarray:
        """1-D ordering of cells so that similar cells sit next to each other in the heatmap."""
        S = np.clip(self.centroids @ self.centroids.T, 0, None)
        L = np.diag(S.sum(1)) - S
        _, vecs = np.linalg.eigh(L)
        return np.argsort(vecs[:, 1])
