"""Semantic cortisol: stress per intent, spread over a graph of intents.

The k-means cells of the cortisol field know where a query is in embedding space, not which intent
it is about, and a freshly learned intent loses to its old neighbour inside the same cell. Here the
unit is the intent.

GRAPH. Nodes are intents. Edge weight = max(cosine similarity of the intents' MiniLM vectors,
normalised confusion counts from escalations). An intent's vector is its name plus the mean of its
training examples. The confusion part is rebuilt from the escalation log at every retrain.

STRESS. One number in [0, 1] per intent, injected by three sensors and spread to the top-3 graph
neighbours (scaled by edge weight and `spread`):
  (a) a new intent appearing: the teacher uses an intent the student does not know;
  (b) a fit-check failure: the query's distance to the nearest training example of its predicted
      intent exceeds the 95th percentile of the same distance on history (label-free);
  (c) per-intent residuals: wrong minus expected-wrong-at-that-doubt on every teacher-seen answer;
  (d) optional LLM sensor: a cluster of 20+ escalated queries outside every intent's 95th-percentile
      radius is shown to Claude once, which names the intents it could be confused with.
Stress decays every query. While an intent's stress is at or above `hot`, it is on probation: its
predictions need more doubt-room to be accepted (level += gain * stress) and accepted ones are
audited at `audit_p`. Probation ends when `release` consecutive teacher-seen answers for that
intent agree with the student.
"""
from __future__ import annotations

import numpy as np

from . import embed


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


class IntentGraph:
    def __init__(self, intents: list[str]):
        self.intents = list(intents)
        self.K = len(intents)
        self.name_vec = embed.encode([n.replace("_", " ") for n in intents])
        self.cos = np.zeros((self.K, self.K))
        self.conf = np.zeros((self.K, self.K))
        self.W = np.zeros((self.K, self.K))
        self.vec = self.name_vec.copy()

    def rebuild(self, examples: dict[int, np.ndarray], escalations: list[tuple[int, int]]) -> None:
        """examples: intent index -> (n, d) embeddings of its training examples;
        escalations: (student's intent, teacher's intent) of every escalated query so far."""
        V = self.name_vec.copy()
        for k, E in examples.items():
            if len(E):
                V[k] = self.name_vec[k] + E.mean(0)
        self.vec = _unit(V)
        self.cos = self.vec @ self.vec.T
        C = np.zeros((self.K, self.K))
        n = np.zeros(self.K)
        for pred, truth in escalations:
            n[truth] += 1
            if pred != truth:
                C[pred, truth] += 1
        S = C + C.T
        self.conf = S / np.maximum(n[:, None] + n[None, :], 1.0)
        W = np.maximum(self.cos, self.conf)
        np.fill_diagonal(W, 0.0)
        self.W = np.clip(W, 0.0, 1.0)

    def neighbours(self, k: int, top: int = 3, among: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        w = self.W[k].copy()
        w[k] = -1.0
        if among is not None:
            w[~among] = -1.0
        idx = np.argsort(-w)[:top]
        return idx, np.clip(w[idx], 0.0, 1.0)


class SemanticField:
    def __init__(self, graph: IntentGraph, cfg):
        self.g, self.cfg = graph, cfg
        self.stress = np.zeros(graph.K)
        self.agree = np.zeros(graph.K, dtype=int)
        self.on = np.zeros(graph.K, dtype=bool)

    def level(self, k: int) -> float:
        return self.cfg.sem_gain * float(self.stress[k])

    def probation(self, k: int) -> bool:
        return bool(self.on[k])

    def inject(self, k: int, amount: float, spread: bool = True, mode: str = "add") -> None:
        """mode "add" accumulates (clipped to [0, 1]); "max" lifts the stress to at least `amount`."""
        s = self.stress
        if mode == "max":
            s[k] = max(s[k], amount)
        else:
            s[k] = min(1.0, max(0.0, s[k] + amount))
        if spread and amount > 0:
            idx, w = self.g.neighbours(k, self.cfg.sem_top)
            for j, wj in zip(idx, w):
                a = amount * self.cfg.sem_spread * wj
                s[j] = max(s[j], a) if mode == "max" else min(1.0, s[j] + a)
        self._flags()

    def _flags(self) -> None:
        now = self.stress >= self.cfg.sem_hot
        new = now & ~self.on
        self.agree[new] = 0
        self.on = now

    def tick(self) -> None:
        self.stress *= 1.0 - self.cfg.sem_decay
        self._flags()

    def observe(self, k: int, agrees: bool) -> bool:
        """A teacher-seen answer for intent k. Returns True when this releases it from probation."""
        if not self.on[k]:
            return False
        self.agree[k] = self.agree[k] + 1 if agrees else 0
        if self.agree[k] >= self.cfg.sem_release:
            self.stress[k] = 0.0
            self.on[k] = False
            self.agree[k] = 0
            return True
        return False


class OutsideClusters:
    """Escalated queries outside every intent's 95th-percentile radius, grouped greedily.

    A cluster with `size` members fires once (the sensor is asked about it) and is then cleared."""

    def __init__(self, sim: float = 0.6, size: int = 20, keep: int = 40):
        self.sim, self.size, self.keep = sim, size, keep
        self.clusters: list[dict] = []

    def clear(self) -> None:
        self.clusters = []

    def add(self, i: int, e: np.ndarray) -> list[int] | None:
        for c in self.clusters:
            if float(c["centroid"] @ e) >= self.sim:
                c["members"].append(i)
                c["sum"] = c["sum"] + e
                c["centroid"] = _unit(c["sum"])
                if len(c["members"]) >= self.size and not c["fired"]:
                    c["fired"] = True
                    return list(c["members"])
                return None
        self.clusters.append({"centroid": e.copy(), "sum": e.copy(), "members": [i], "fired": False})
        self.clusters = self.clusters[-self.keep:]
        return None


def radii(examples: dict[int, np.ndarray], K: int, q: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-intent unit centroid, 95th-percentile radius (1 - cosine to the centroid) and a mask of
    intents with enough examples to have one. Intents with < 5 examples get the median radius."""
    cen = np.zeros((K, examples[next(iter(examples))].shape[1]))
    rad = np.full(K, np.nan)
    have = np.zeros(K, dtype=bool)
    for k, E in examples.items():
        if len(E):
            cen[k] = _unit(E.mean(0))
            have[k] = True
            if len(E) >= 5:
                rad[k] = np.quantile(1.0 - E @ cen[k], q)
    med = np.nanmedian(rad) if np.isfinite(rad).any() else 0.5
    rad = np.where(np.isnan(rad), med, rad)
    return cen, rad, have
