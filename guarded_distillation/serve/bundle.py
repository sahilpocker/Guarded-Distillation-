"""A model bundle: everything the service needs to run the gate, as one small pickle.

Built the way the engine's go-live does it: a 3:1 train/calibration split of teacher-labelled
queries, scope and intent students, a doubt scale per skill, and a starting level per skill from
conformal risk control on the calibration split. The level then stays static (the recommended
setting is the static gate with retraining; retraining builds a new bundle).
"""
from __future__ import annotations

import json
import pickle
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..compat import load_pickle
from .. import embed
from ..gate import DoubtScale, choose_level, expected_wrong, runner_up
from ..router import LearnedRouter, features, init_router
from ..student import Student
from ..taxonomy import OOS

IN = "in_scope"
SCOPE_UNIVERSE = [IN, OOS]


class Calib:
    """Expected wrong rate as a function of doubt percentile (picklable twin of gate.expected_wrong)."""

    def __init__(self, u, wrong, w):
        f = expected_wrong(np.asarray(u), np.asarray(wrong), np.asarray(w))
        self.grid = np.linspace(0, 1, 101)
        self.vals = np.array([float(f(g)) for g in self.grid])

    def __call__(self, x):
        return np.interp(x, self.grid, self.vals)


@dataclass
class Bundle:
    version: str
    intents: list[str]
    scope: Student
    intent: Student
    scale_scope: DoubtScale
    scale_intent: DoubtScale
    q0_scope: float
    q0_intent: float
    calib_scope: Calib
    calib_intent: Calib
    alpha_scope: float
    alpha_intent: float
    meta: dict = field(default_factory=dict)
    router: LearnedRouter | None = None          # learned router for the intent gate (None in older bundles: margin gate)
    rwin: list = field(default_factory=list)     # recent router scores: the reference for the percentile
    ex_emb: dict = field(default_factory=dict)   # intent index -> training-example embeddings (the fit-distance feature)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "bundle.pkl", "wb") as f:
            pickle.dump(self, f)
        (path / "meta.json").write_text(json.dumps({"version": self.version, **self.meta}, indent=2))

    @staticmethod
    def load(path: str | Path) -> "Bundle":
        with open(Path(path) / "bundle.pkl", "rb") as f:
            return load_pickle(f)


def query_features(b: "Bundle", text: str, P_row: np.ndarray, e: np.ndarray, err_rate: np.ndarray):
    """Router features for one query: (feature vector, predicted intent index)."""
    k = int(P_row.argmax())
    ex = b.ex_emb.get(k)
    dist = 1.0 - float((e @ ex.T).max()) if ex is not None and len(ex) else 1.0
    nov = float(np.clip((b.intent.oov_fraction([text])[0] - 0.10) / 0.40, 0.0, 1.0))
    return features(P_row, dist, float(err_rate[k]), len(text.split()), nov), k


def build_bundle(texts: list[str], labels: list[str], intents: list[str], version: str, seed: int = 0,
                 alpha_scope: float = 0.01, alpha_intent: float = 0.03, balance: bool = True, learned: bool = True, meta: dict | None = None) -> Bundle:
    keep = [i for i, y in enumerate(labels) if y == OOS or y in intents]
    texts, labels = [texts[i] for i in keep], [labels[i] for i in keep]
    n = len(texts)
    perm = np.random.default_rng(seed).permutation(n)
    n_cal = n // 4
    cal, tr = sorted(perm[:n_cal].tolist()), sorted(perm[n_cal:].tolist())
    x_tr = [texts[i] for i in tr]
    y_tr = [labels[i] for i in tr]
    ys_tr = [OOS if y == OOS else IN for y in y_tr]
    x_cal = [texts[i] for i in cal]
    y_cal = [labels[i] for i in cal]
    ys_cal = [OOS if y == OOS else IN for y in y_cal]
    ik = [i for i, y in enumerate(y_tr) if y in intents]
    ick = [i for i, y in enumerate(y_cal) if y in intents]
    sc = Student(SCOPE_UNIVERSE, seed=seed, backend="sbert").fit(x_tr, ys_tr, holdout=(x_cal, ys_cal))
    sw = None
    if balance:       # an intent with few labels (a new one) counts as much as the median intent, up to 20x, until it has 50
        cnt = Counter(y_tr[i] for i in ik)
        med = float(np.median(list(cnt.values())))
        sw = np.array([min(20.0, max(1.0, med / cnt[y_tr[i]])) if cnt[y_tr[i]] < 50 else 1.0 for i in ik])
    it = Student(intents, seed=seed, backend="sbert").fit([x_tr[i] for i in ik], [y_tr[i] for i in ik],
                                                           holdout=([x_cal[i] for i in ick], [y_cal[i] for i in ick]), sample_weight=sw)
    out = {}
    for name, st, xs, ys, uni in (("scope", sc, x_cal, ys_cal, SCOPE_UNIVERSE),
                                  ("intent", it, [x_cal[i] for i in ick], [y_cal[i] for i in ick], intents)):
        P_all = st.proba(texts[-2000:])                     # doubt scale: recent traffic, no labels
        scale = DoubtScale(runner_up(P_all))
        P = st.proba(xs)
        u = scale(runner_up(P))
        wrong = (np.array(uni)[P.argmax(1)] != np.array(ys)).astype(float)
        w = np.ones(len(xs))
        out[name] = (scale, choose_level(u, wrong, w, alpha_scope if name == "scope" else alpha_intent), Calib(u, wrong, w))
    # learned router for the intent gate, trained on the calibration split (two-fold) as the engine does
    router = rwin = ex_emb = None
    if learned:
        iidx = {lab: i for i, lab in enumerate(intents)}
        ex_lists: dict[int, list[int]] = {}
        for j in ik:
            ex_lists.setdefault(iidx[y_tr[j]], []).append(j)
        E_tr = embed.encode([x_tr[j] for j in ik])
        pos = {j: n_ for n_, j in enumerate(ik)}
        ex_emb = {k: E_tr[[pos[j] for j in v]] for k, v in ex_lists.items()}
        proto = Bundle(version, list(intents), sc, it, out["scope"][0], out["intent"][0], out["scope"][1], out["intent"][1],
                       out["scope"][2], out["intent"][2], alpha_scope, alpha_intent, ex_emb=ex_emb)
        Xc = [x_cal[j] for j in ick]
        Pc, Ec = it.proba(Xc), embed.encode(Xc)
        prior = np.full(len(intents), 0.15)
        X = np.array([query_features(proto, Xc[n_], Pc[n_], Ec[n_], prior)[0] for n_ in range(len(Xc))])
        wr = np.array([float(intents[int(Pc[n_].argmax())] != y_cal[j]) for n_, j in enumerate(ick)])
        router, order, q0 = init_router(X, wr, alpha_intent, seed)
        out["intent"] = (out["intent"][0], q0, out["intent"][2])
        rwin = [float(v) for v in np.random.default_rng(seed).permutation(order)[-2000:]]
    return Bundle(version=version, intents=list(intents), scope=sc, intent=it, scale_scope=out["scope"][0], scale_intent=out["intent"][0],
                  q0_scope=out["scope"][1], q0_intent=out["intent"][1], calib_scope=out["scope"][2], calib_intent=out["intent"][2],
                  alpha_scope=alpha_scope, alpha_intent=alpha_intent, router=router, rwin=rwin or [], ex_emb=ex_emb or {},
                  meta={"built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_labels": n, "n_intents": len(intents),
                        "intent_holdout_acc": float(it.holdout_acc), "learned_router": bool(learned), "scope_holdout_acc": float(sc.holdout_acc),
                        "label_counts": dict(Counter(labels).most_common(5)), **(meta or {})})
