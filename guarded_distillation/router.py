"""Routing scores for the gate: how doubtful is the student about this query?

Static scores (computed from the student's probabilities alone):
  margin    runner-up probability (the original doubt)
  entropy   entropy of the probability vector
  maxprob   1 - highest probability
  disagree  margin, plus 1 when the previous student version predicts a different intent
A learned router predicts "the student disagrees with the teacher" from seven features and its
probability is the score. It is trained online on teacher-seen queries: escalations (weight 1) and
audits (weight 1/p, correcting for audits being a random sample of what was accepted).
Whatever the score, it is turned into a percentile of recent traffic and goes through the same
risk-controlled gate.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

FEATURES = ("margin", "entropy", "maxprob_doubt", "fit_distance", "recent_error_rate", "length", "novelty")


def features(P_row: np.ndarray, dist: float, err_rate: float, n_words: int, novelty: float) -> np.ndarray:
    p = np.sort(P_row)[::-1]
    margin = p[1] if len(p) > 1 else 0.0
    ent = float(-(P_row * np.log(P_row + 1e-12)).sum())
    return np.array([margin, ent, 1.0 - p[0], dist, err_rate, n_words / 10.0, novelty])


class LearnedRouter:
    def __init__(self, refit_every: int = 100, min_samples: int = 60):
        self.X: list[np.ndarray] = []
        self.y: list[float] = []
        self.w: list[float] = []
        self.refit_every, self.min_samples = refit_every, min_samples
        self.model = None
        self.mu = self.sd = None
        self._since = 0

    def add(self, x: np.ndarray, wrong: float, weight: float) -> None:
        self.X.append(x)
        self.y.append(float(wrong > 0))
        self.w.append(float(weight))
        self._since += 1
        if self._since >= self.refit_every:
            self.fit()

    def fit(self, X=None, y=None, w=None) -> None:
        X = np.asarray(self.X if X is None else X)
        y = np.asarray(self.y if y is None else y)
        w = np.asarray(self.w if w is None else w)
        self._since = 0
        if len(y) < self.min_samples or y.min() == y.max():
            return
        w = w / w.mean()
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.model = LogisticRegression(C=1.0, max_iter=300).fit((X - self.mu) / self.sd, y, sample_weight=w)

    def score(self, x: np.ndarray) -> float:
        if self.model is None:
            return float(x[0])                     # until trained, fall back to margin
        return float(self.model.predict_proba(((x - self.mu) / self.sd)[None])[0, 1])

    def score_many(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return X[:, 0]
        return self.model.predict_proba((X - self.mu) / self.sd)[:, 1]
