"""Students: the cheap compiled models that answer when the gate accepts.

A student is TF-IDF (word 1-2 grams + character 3-5 grams) and a linear SVM. Its scores become
probabilities through a softmax whose temperature is fitted on held-out teacher labels. The gate
does not need calibrated probabilities (conformal methods work with any score), but a fitted
temperature keeps the scale comparable from one retrain to the next.

Probabilities always cover the full label universe. A label the student has never been trained
on gets probability 0, which is what lets cortisol notice a new intent: its true label can only
enter the prediction set once cortisol has pushed the bar down to 0.
"""
from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_union
from sklearn.svm import LinearSVC

TEMPS = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20], dtype=float)


class Student:
    def __init__(self, universe: list[str], C: float = 0.5, seed: int = 0):
        self.universe = list(universe)
        self.index = {lab: i for i, lab in enumerate(self.universe)}
        self.C, self.seed = C, seed
        self.T = 5.0
        self.version = 0
        self.known: set[str] = set()
        self.holdout_acc = float("nan")

    def _features(self):
        return make_union(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_features=60000),
        )

    def fit(self, texts: list[str], labels: list[str], holdout: tuple[list[str], list[str]] | None = None) -> "Student":
        labels = [lab for lab in labels]
        self.known = set(labels)
        self.vec = self._features().fit(texts)
        X = self.vec.transform(texts)
        if len(self.known) == 1:
            self.const = next(iter(self.known))
            self.clf = None
        else:
            self.const = None
            self.clf = LinearSVC(C=self.C, random_state=self.seed).fit(X, labels)
        if holdout is not None and len(holdout[0]) >= 20:
            self._fit_temperature(*holdout)
            self.holdout_acc = self.accuracy(*holdout)
        return self

    def _scores(self, texts: list[str]) -> np.ndarray:
        X = self.vec.transform(texts)
        s = self.clf.decision_function(X)
        if s.ndim == 1:  # binary: decision_function gives the margin for classes_[1]
            s = np.column_stack([-s / 2, s / 2])
        return s

    def _fit_temperature(self, texts: list[str], labels: list[str]) -> None:
        if self.clf is None:
            return
        cls = list(self.clf.classes_)
        rows = [i for i, lab in enumerate(labels) if lab in self.known]
        if len(rows) < 20:
            return
        S = self._scores([texts[i] for i in rows])
        y = np.array([cls.index(labels[i]) for i in rows])
        nll = [-np.log(softmax(S * T, axis=1)[np.arange(len(y)), y] + 1e-12).mean() for T in TEMPS]
        self.T = float(TEMPS[int(np.argmin(nll))])

    def proba(self, texts: list[str]) -> np.ndarray:
        """(n, len(universe)) probabilities; 0 for labels the student has not been trained on."""
        P = np.zeros((len(texts), len(self.universe)))
        if self.clf is None:
            P[:, self.index[self.const]] = 1.0
            return P
        sub = softmax(self._scores(texts) * self.T, axis=1)
        cols = [self.index[c] for c in self.clf.classes_]
        P[:, cols] = sub
        return P

    def oov_fraction(self, texts: list[str]) -> np.ndarray:
        """Label-free novelty: share of each text's words the student never saw in training."""
        word_vec = self.vec.transformer_list[0][1]
        analyze, vocab = word_vec.build_analyzer(), word_vec.vocabulary_
        out = np.zeros(len(texts))
        for k, t in enumerate(texts):
            toks = [g for g in analyze(t) if " " not in g]
            if toks:
                out[k] = sum(g not in vocab for g in toks) / len(toks)
        return out

    def predict(self, texts: list[str]) -> list[str]:
        P = self.proba(texts)
        return [self.universe[i] for i in P.argmax(1)]

    def accuracy(self, texts: list[str], labels: list[str]) -> float:
        if not texts:
            return float("nan")
        return float(np.mean(np.array(self.predict(texts)) == np.array(labels)))
