"""The replay engine: one pass over the stream for one method.

Per query:
  1. History (shadow mode before go-live, rows with phase "history"): everything goes to the
     teacher and its labels seed the pool. At go-live the map space is fitted, students v1 are
     trained, and each skill's starting cortisol comes from conformal risk control on held-out
     history labels.
  2. Scope gate (in scope vs out of scope). If it accepts "out of scope", done.
  3. Intent gate (77 intents). If it accepts, the student's intent is the answer.
  4. Any gate that does not accept escalates: one teacher call returns the label for both skills.
  5. Accepted answers are audited with probability p (uniform, or directed by cortisol).
  6. Cortisol updates (gate.py): guard every query, local on every teacher label, tonic every
     `refresh` queries from the recent window.
  7. Retraining is triggered by evidence (labels collected), not by time:
       - every `retrain_every` new teacher labels
       - once the teacher has used an intent the student does not know `novel_min` times
       - (full system) when a cell's cortisol is high and some fresh labels have arrived
     A retrained student is a shadow first; it is promoted only if it does at least as well as
     the current one on recent held-out labels (or if it knows intents the current one lacks).

Routing team and urgency are T0 skills (lookup from the intent) and need no gate.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import embed
from .gate import (CortisolField, DoubtScale, FieldConfig, accepts, choose_level, doubt, expected_wrong,
                   runner_up)
from .router import LearnedRouter, features
from .semantic import IntentGraph, OutsideClusters, SemanticField, radii
from .space import MapSpace
from .student import Student
from .taxonomy import OOS

IN = "in_scope"
SCOPE_UNIVERSE = [IN, OOS]


@dataclass(frozen=True)
class Method:
    name: str
    title: str
    llm_only: bool = False
    tonic: bool = True
    guard: bool = True
    local: bool = False
    novelty: bool = False
    directed_audits: bool = False
    retrain: bool = True
    hot_cell_retrain: bool = False
    semantic: bool = False        # per-intent stress on the intent graph (guarded_distillation/semantic.py)
    sensor: bool = False          # ... plus the LLM sensor on clusters outside every intent's radius


METHODS: dict[str, Method] = {m.name: m for m in [
    Method("llm_only", "LLM only", llm_only=True),
    Method("static", "Static gate", tonic=False, guard=False, retrain=False),
    Method("retrain", "Static gate + retraining", tonic=False, guard=False),
    Method("global", "Global cortisol"),
    Method("cortisol", "Cortisol field", local=True, novelty=True, directed_audits=True,
           hot_cell_retrain=True),
    Method("semantic", "Semantic cortisol", tonic=False, guard=False, semantic=True),
    Method("semantic_llm", "Semantic cortisol + LLM sensor", tonic=False, guard=False, semantic=True, sensor=True),
]}


# The recommended operating point, chosen on seeds 0-2 (fewest LLM calls with overall accuracy >= -0.3
# pts and launch-question accuracy >= -2 pts vs LLM only on every seed) and reported on seeds 3-5
# (docs/RESULTS.md): static gate + retraining, MiniLM student, alpha_intent 0.03, TF-IDF map. It needs
# the [sbert] extra, so EngineConfig's own defaults stay light (TF-IDF, 0.03) for tests and the free
# demo; `gdistill replay --preset recommended` applies the student.
# Class-balanced retraining was added afterwards (docs/RESULTS.md, semantic cortisol): same calls, launch Δ -3.4 -> -0.3 on seeds 3-5.
RECOMMENDED = {"method": "retrain", "student_backend": "sbert", "alpha_intent": 0.03, "map_backend": "tfidf", "balance": True,
               "router": "learned"}     # routing benchmark: the learned router beats margin at every call budget (docs/RESULTS.md)


@dataclass(frozen=True)
class EngineConfig:
    warmup: int = 800             # used only when the stream has no history rows
    alpha_scope: float = 0.01     # leak budget per query
    alpha_intent: float = 0.03    # leak budget per in-scope query
    gamma_guard: float = 0.002
    gamma_local: float = 0.03
    decay: float = 0.002
    novelty_gain: float = 0.5
    refresh: int = 25             # tonic re-estimated every this many queries
    window: int = 3000            # queries the tonic looks back over
    audit_base: float = 0.08
    audit_max: float = 0.5
    audit_gain: float = 0.5
    retrain_every: int = 400
    novel_min: int = 8
    hot_threshold: float = 0.25
    hot_min_labels: int = 60
    holdout_frac: float = 0.15
    pool_cap: int = 12000
    n_cells: int = 40
    map_backend: str = "tfidf"
    student_backend: str = "tfidf"   # "sbert": frozen MiniLM embeddings + logistic regression (E2)
    snapshot_every: int = 50
    seed: int = 0
    # class-balanced retraining (needs the [sbert] extra): a newly learned intent is weighted up to the
    # size of its nearest old neighbour until it has `balance_to` teacher labels
    balance: bool = False
    balance_to: int = 50
    # semantic cortisol (see guarded_distillation/semantic.py)
    sem_gain: float = 1.0         # level added to an intent's predictions per unit of stress
    sem_top: int = 3              # stress spreads to this many graph neighbours
    sem_spread: float = 0.5       # ... scaled by this and the edge weight
    sem_hot: float = 0.25         # probation starts at this stress
    sem_decay: float = 0.002
    sem_release: int = 30         # consecutive teacher-seen agreements that end probation
    sem_audit: float = 0.5        # audit probability for accepted answers of an intent on probation
    sem_new: float = 1.0          # (a) stress on a new intent the student does not know
    sem_fit: float = 0.05         # (b) stress per fit-check failure
    sem_resid: float = 0.15       # (c) gain on (wrong - expected wrong)
    sem_sensor: float = 0.6       # (d) stress on the intents the LLM sensor names
    seed_examples: dict | None = None   # {intent: [synthetic examples]}: declared at the first query of the launch phase (an empty
                                        # list declares the intent with probation only); None = no seeding
    router: str = "margin"        # intent-gate score: margin | entropy | maxprob | disagree | learned | oracle
    router_refit: int = 100
    sem_cluster: int = 20         # (d) queries in a cluster before the LLM sensor is asked
    sem_qfit: float = 0.0         # level added to a query that fails the fit check (label-free, immediate)
    sem_sim: float = 0.6          # (d) cosine similarity to join a cluster
    # forgetting (E3, docs/RESULTS.md): at retraining, drop older pool labels that the newest `forget_recent` teacher labels contradict
    forget: bool = False
    forget_recent: int = 600
    forget_sim: float = 0.85      # cosine similarity (MiniLM) for "the same kind of question"
    forget_min: int = 3           # at least this many recent look-alikes ...
    forget_frac: float = 0.6      # ... and at least this share of them labelled differently


@dataclass
class RunResult:
    method: Method
    calls: pd.DataFrame
    snapshots: list[dict]
    events: list[dict]
    cells: dict = field(default_factory=dict)


class SkillGate:
    """One gated skill: its student, doubt scale, cortisol field and evidence log."""

    def __init__(self, name: str, universe: list[str], alpha: float, n: int):
        self.name, self.universe, self.alpha, self.n = name, universe, alpha, n
        self.index = {lab: i for i, lab in enumerate(universe)}
        self.P = np.zeros((n, len(universe)))
        self.u = np.zeros(n)
        # evidence log for the tonic and the calibration curve (one entry per active query)
        self.log_t: list[int] = []
        self.log_u: list[float] = []
        self.log_wrong: list[float] = []
        self.log_w: list[float] = []
        self.expected = lambda x: np.full(np.shape(x), 0.1)
        self.kind = "margin"

    def refresh_predictions(self, texts: list[str], t0: int) -> None:
        """Probabilities and doubt percentiles from t0 on, under the current student.

        The doubt scale is fitted on the student's doubts over the 2,000 queries before t0
        (labels are not used)."""
        a = max(0, t0 - 2000)
        P = self.student.proba(texts[a:])
        self.P[a:] = P
        ref = doubt(P[: t0 - a], self.kind) if t0 - a >= 200 else doubt(P, self.kind)
        self.scale = DoubtScale(ref)
        self.u[t0:] = self.scale(doubt(self.P[t0:], self.kind))

    def top(self, i: int) -> str:
        return self.universe[int(np.argmax(self.P[i]))]

    def log(self, t: int, u: float, wrong: float, w: float) -> None:
        self.log_t.append(t)
        self.log_u.append(u)
        self.log_wrong.append(wrong)
        self.log_w.append(w)

    def window(self, t: int, size: int):
        k = np.searchsorted(np.asarray(self.log_t), t - size, side="left") if self.log_t else 0
        u = np.asarray(self.log_u[k:])
        wr = np.asarray(self.log_wrong[k:])
        w = np.asarray(self.log_w[k:])
        seen = w > 0
        return u[seen], np.nan_to_num(wr[seen]), w[seen], len(u)


class Replay:
    def __init__(self, stream: pd.DataFrame, teacher: pd.DataFrame, intents: tuple[str, ...],
                 method: Method, cfg: EngineConfig, call_cost_usd: np.ndarray, sensor=None):
        self.sensor = sensor
        self.s, self.m, self.cfg = stream.reset_index(drop=True), method, cfg
        self.texts = self.s.text.tolist()
        self.n = len(self.s)
        self.y_teacher = teacher.teacher.to_numpy()
        self.cost = np.asarray(call_cost_usd, dtype=float)
        self.intents = list(intents)
        self.rng = np.random.default_rng(cfg.seed + 7)
        n_hist = int((self.s.phase == "history").sum())
        self.warm = n_hist if n_hist else cfg.warmup
        self.t_of = self.s.t.to_numpy()
        self.events: list[dict] = []
        self.snapshots: list[dict] = []
        self.pool: list[int] = []
        self.labels_since = 0
        self.teacher_seen: Counter = Counter()
        self.hot_cells: set[int] = set()
        self.alarm = {"scope": False, "intent": False}
        self.S = SkillGate("scope", SCOPE_UNIVERSE, cfg.alpha_scope, self.n)
        self.I = SkillGate("intent", self.intents, cfg.alpha_intent, self.n)
        if cfg.router in ("entropy", "maxprob"):
            self.I.kind = cfg.router
        self.prev_I = None
        self.extra: list[tuple[str, str]] = []          # synthetic seed examples (text, intent) that join training
        self.extra_emb: dict[int, np.ndarray] = {}
        self.declared = False
        self.phase_arr = self.s.phase.to_numpy()
        self.router = LearnedRouter(cfg.router_refit) if cfg.router == "learned" else None
        self.rwin: list[float] = []
        self.err_rate = np.full(len(self.intents), 0.15)
        self.esc_log: list[tuple[int, int]] = []     # (student's intent, teacher's intent) of escalations
        self.sensor_calls = 0
        self.sensor_stop = False

    # ------------------------------------------------------------------ helpers
    def _event(self, i: int, kind: str, text: str, **kw) -> None:
        self.events.append({"t": int(self.t_of[i]), "kind": kind, "text": text, **kw})

    def _xy(self, idx: list[int]):
        x = [self.texts[i] for i in idx]
        y = [self.y_teacher[i] for i in idx]
        return x, y, [OOS if v == OOS else IN for v in y]

    def _fit_pair(self, idx: list[int], hold: list[int] | None, version: int):
        x, y, ys = self._xy(idx)
        if self.extra:                                  # synthetic seeds are training data only, never held-out data
            x = x + [t for t, _ in self.extra]
            y = y + [lab for _, lab in self.extra]
            ys = ys + [IN] * len(self.extra)
        scope_h = intent_h = None
        if hold is not None:
            hx, hy, hys = self._xy(hold)
            scope_h = (hx, hys)
            hk = [k for k, v in enumerate(hy) if v in self.intents]
            intent_h = ([hx[k] for k in hk], [hy[k] for k in hk])
        sc = Student(SCOPE_UNIVERSE, seed=self.cfg.seed, backend=self.cfg.student_backend).fit(x, ys, holdout=scope_h)
        keep = [k for k, v in enumerate(y) if v in self.intents]
        sw = self._weights([y[k] for k in keep]) if self.cfg.balance and hasattr(self, "graph") else None
        it = Student(self.intents, seed=self.cfg.seed, backend=self.cfg.student_backend).fit(
            [x[k] for k in keep], [y[k] for k in keep], holdout=intent_h, sample_weight=sw)
        sc.version = it.version = version
        return sc, it

    def _weights(self, labels: list[str]) -> np.ndarray:
        """Sample weights: an intent the go-live student did not know, with fewer than `balance_to`
        labels, counts as much as its nearest old neighbour (by intent-graph cosine)."""
        cnt = Counter(labels)
        sw = np.ones(len(labels))
        old = [j for j, lab in enumerate(self.intents) if lab in self.base_known and cnt.get(lab, 0) > 0]
        for lab, c in cnt.items():
            if lab in self.base_known or c >= self.cfg.balance_to or not old:
                continue
            k = self.I.index[lab]
            j = old[int(np.argmax(self.graph.cos[k, old]))]
            target = cnt[self.intents[j]]
            if target > c:
                sw[np.array(labels) == lab] = target / c
        return sw

    def _sem_refresh(self) -> None:
        """Rebuild the intent graph (cosine from names + training examples, confusion from the
        escalation log), the per-intent examples for the fit check, and the 95th-percentile radii."""
        ex: dict[int, list[int]] = {}
        for k in self.pool[-self.cfg.pool_cap:]:
            lab = self.y_teacher[k]
            if lab in self.I.index:
                ex.setdefault(self.I.index[lab], []).append(k)
        self.ex_emb = {k: self.E[v] for k, v in ex.items()}
        for k, e in self.extra_emb.items():
            self.ex_emb[k] = np.vstack([self.ex_emb[k], e]) if k in self.ex_emb else e
        self.graph.rebuild(self.ex_emb, self.esc_log)
        if self.m.semantic:
            self.cen, self.rad, self.have = radii(self.ex_emb, len(self.intents))

    def _refresh(self, i0: int) -> None:
        self.S.refresh_predictions(self.texts, i0)
        self.I.refresh_predictions(self.texts, i0)
        rest = self.texts[i0:]
        if not rest:
            return
        if self.cfg.router == "disagree" and self.prev_I is not None:
            same = self.I.P[i0:].argmax(1) == self.prev_I.proba(rest).argmax(1)
            self.I.u[i0:] = self.I.u[i0:] + (~same)        # a version disagreement outranks any margin
        # Label-free novelty in [0, 1], the larger of:
        #  (a) unseen words: share of the query's words the intent student never trained on
        #      (0 below 10% unseen, 1 at 50% or more)
        #  (b) distance in map space to the nearest recent training query, rescaled against the
        #      95th percentile of the same distance among training queries themselves
        oov = np.clip((self.I.student.oov_fraction(rest) - 0.10) / 0.40, 0.0, 1.0)
        tr = self.pool[-3000:]
        Zt = self.Z[tr]
        hold = tr[-max(50, len(tr) // 7):]
        d_hold = 1.0 - np.sort(self.Z[hold] @ Zt.T, axis=1)[:, -2]
        ref = float(np.quantile(d_hold, 0.95))
        for a in range(i0, self.n, 2000):
            b = min(a + 2000, self.n)
            d = 1.0 - (self.Z[a:b] @ Zt.T).max(1)
            knn = np.clip((d - ref) / max(1.0 - ref, 1e-6), 0.0, 1.0)
            self.nov[a:b] = np.maximum(knn, oov[a - i0:b - i0])

    def _update_tonic(self, i: int, force: bool = False) -> None:
        for G, F in ((self.S, self.Fs), (self.I, self.Fi)):
            u, wr, w, n_active = G.window(i, self.cfg.window)
            if len(u) >= 50:
                G.expected = expected_wrong(u, wr, w)
            if F.cfg.tonic and (force or n_active >= 200):
                F.tonic = choose_level(u, wr, w, G.alpha)

    # ------------------------------------------------------------------ lifecycle
    def _bootstrap(self, i: int) -> None:
        cfg, m = self.cfg, self.m
        warm = list(range(self.warm))
        self.map = MapSpace(n_cells=cfg.n_cells, backend=cfg.map_backend, seed=cfg.seed).fit(
            [self.texts[k] for k in warm])
        self.Z = self.map.embed(self.texts)
        self.cell = self.map.cell(self.Z)
        self.nov = np.zeros(self.n)

        perm = np.random.default_rng(cfg.seed).permutation(warm)
        n_cal = len(perm) // 4
        cal, tr = sorted(perm[:n_cal].tolist()), sorted(perm[n_cal:].tolist())
        self.S.student, self.I.student = self._fit_pair(tr, cal, version=1)
        self.base_known = set(self.I.student.known)
        if cfg.balance or m.semantic or cfg.router == "learned":
            self.E = embed.encode(self.texts)
            self.graph = IntentGraph(self.intents)
            self.clusters = OutsideClusters(sim=cfg.sem_sim, size=cfg.sem_cluster)
            self._sem_refresh()
            if m.semantic:
                self.sem = SemanticField(self.graph, cfg)
                # fit-check threshold: 95th percentile, over history, of the distance from a training
                # example to the nearest other example of its own intent
                d = []
                for E in self.ex_emb.values():
                    if len(E) >= 2:
                        S = E @ E.T
                        np.fill_diagonal(S, -np.inf)
                        d.append(1.0 - S.max(1))
                self.fit_thr = float(np.quantile(np.concatenate(d), 0.95))
        self._refresh(i + 1)

        # starting level: conformal risk control on the held-out history labels
        q0 = {}
        for G in (self.S, self.I):
            idx = [k for k in cal if (G is self.S) or self.y_teacher[k] in self.intents]
            P = G.student.proba([self.texts[k] for k in idx])
            u = G.scale(runner_up(P))
            truth = [(OOS if self.y_teacher[k] == OOS else IN) if G is self.S else self.y_teacher[k] for k in idx]
            wrong = (np.array(G.universe)[P.argmax(1)] != np.array(truth)).astype(float)
            q0[G.name] = choose_level(u, wrong, np.ones(len(idx)), G.alpha)
            for k, uu, ww in zip(idx, u, wrong):  # calibration labels seed the evidence window
                G.log(k, float(uu), float(ww), 1.0)
            G.expected = expected_wrong(u, wrong, np.ones(len(idx)))

        if cfg.router == "learned":
            q0["intent"] = self._init_router(cal, tr)

        def fc(alpha):
            return FieldConfig(alpha=alpha, tonic=m.tonic, guard=m.guard, local=m.local,
                               gamma_guard=cfg.gamma_guard, gamma_local=cfg.gamma_local, decay=cfg.decay,
                               novelty_gain=cfg.novelty_gain if m.novelty else 0.0)
        self.Fs = CortisolField(fc(cfg.alpha_scope), self.map.W, q0["scope"])
        self.Fi = CortisolField(fc(cfg.alpha_intent), self.map.W, q0["intent"])
        self._event(i, "golive",
                    f"Go-live: students v1 trained on {len(tr)} labels from shadow mode "
                    f"(intent accuracy {self.I.student.holdout_acc:.0%} on held-out); starting cortisol "
                    f"scope {q0['scope']:.2f}, intent {q0['intent']:.2f}",
                    q0_scope=q0["scope"], q0_intent=q0["intent"])

    def _feat(self, i: int, P_row: np.ndarray, ex: dict, k: int, nov: float | None = None) -> np.ndarray:
        E = ex.get(k)
        dist = 1.0 - float((self.E[i] @ E.T).max()) if E is not None and len(E) else 1.0
        return features(P_row, dist, float(self.err_rate[k]), len(self.texts[i].split()),
                        float(self.nov[i]) if nov is None else nov)

    def _init_router(self, cal: list[int], tr: list[int]) -> float:
        """Train the learned router on the calibration split (two-fold, so the scores that set the
        starting level are out of sample), and return the starting intent level."""
        cfg = self.cfg
        ex: dict[int, list[int]] = {}
        for k in tr:
            if self.y_teacher[k] in self.I.index:
                ex.setdefault(self.I.index[self.y_teacher[k]], []).append(k)
        ex = {k: self.E[v] for k, v in ex.items()}
        ci = [k for k in cal if self.y_teacher[k] in self.I.index]
        P = self.I.student.proba([self.texts[k] for k in ci])
        top = P.argmax(1)
        X = np.array([self._feat(k, P[j], ex, int(top[j]), nov=0.0) for j, k in enumerate(ci)])
        wrong = np.array([float(self.intents[top[j]] != self.y_teacher[k]) for j, k in enumerate(ci)])
        half = np.random.default_rng(cfg.seed).permutation(len(ci)) % 2
        s = np.zeros(len(ci))
        for f in (0, 1):
            r = LearnedRouter(min_samples=30)
            r.fit(X[half != f], wrong[half != f], np.ones((half != f).sum()))
            s[half == f] = r.score_many(X[half == f])
        self.router.fit(X, wrong, np.ones(len(ci)))
        for x, yy in zip(X, wrong):                              # the router keeps learning from these
            self.router.X.append(x)
            self.router.y.append(float(yy))
            self.router.w.append(1.0)
        self.rwin = list(s[-2000:])
        order = np.sort(s)
        u = np.searchsorted(order, s, side="right") / len(order)
        return choose_level(u, wrong, np.ones(len(ci)), cfg.alpha_intent)

    def _retrain(self, i: int, reason: str) -> None:
        cfg = self.cfg
        if hasattr(self, "graph"):
            self._sem_refresh()
        pool = self.pool[-cfg.pool_cap:]
        if cfg.forget:
            pool = self._forget(pool, i)
        recent = pool[-1500:]
        rng = np.random.default_rng(cfg.seed + len(pool))
        n_hold = min(max(30, int(len(recent) * cfg.holdout_frac)), len(recent) // 2)
        hold = sorted(rng.choice(recent, size=n_hold, replace=False).tolist())
        hs = set(hold)
        tr = [k for k in pool if k not in hs]
        version = self.I.student.version + 1
        new_s, new_i = self._fit_pair(tr, hold, version)
        hx, hy, hys = self._xy(hold)
        hk = [k for k, v in enumerate(hy) if v in self.intents]
        hix, hiy = [hx[k] for k in hk], [hy[k] for k in hk]
        cur_i, cur_s = self.I.student.accuracy(hix, hiy), self.S.student.accuracy(hx, hys)
        new_ia, new_sa = new_i.accuracy(hix, hiy), new_s.accuracy(hx, hys)
        learns = sorted(new_i.known - self.I.student.known)
        self.labels_since = 0
        if not ((new_ia >= cur_i - 0.01 and new_sa >= cur_s - 0.01) or learns):
            self._event(i, "shadow_rejected",
                        f"Shadow v{version} rejected ({reason}): intent {new_ia:.0%} vs current {cur_i:.0%}")
            return
        # promote; refit on the whole pool (keeps the newest labels) with the shadow's temperature
        fs, fi = self._fit_pair(pool, None, version)
        fs.T, fi.T = new_s.T, new_i.T
        fs.holdout_acc, fi.holdout_acc = new_sa, new_ia
        self.prev_I = self.I.student
        self.S.student, self.I.student = fs, fi
        self._refresh(i + 1)
        msg = f"Student v{version} promoted ({reason}); intent accuracy on recent labels {cur_i:.0%} -> {new_ia:.0%}"
        if learns:
            msg += f"; now knows {', '.join(learns)}"
        self._event(i, "promoted", msg, version=version, reason=reason, learned=learns,
                    intent_acc_before=cur_i, intent_acc_after=new_ia)

    def _forget(self, pool: list[int], i: int) -> list[int]:
        """Label-free forgetting: an older label is dropped when at least `forget_min` of the newest teacher labels are look-alike
        questions (cosine >= forget_sim) and `forget_frac` of those carry a different label. Uses no gold labels."""
        cfg = self.cfg
        if len(pool) <= cfg.forget_recent:
            return pool
        old, rec = pool[:-cfg.forget_recent], pool[-cfg.forget_recent:]
        e_old = embed.encode([self.texts[k] for k in old])
        e_rec = embed.encode([self.texts[k] for k in rec])
        y_old = np.array([self.y_teacher[k] for k in old])
        y_rec = np.array([self.y_teacher[k] for k in rec])
        near = (e_old @ e_rec.T) >= cfg.forget_sim
        n_nb = near.sum(1)
        n_dis = (near & (y_rec[None, :] != y_old[:, None])).sum(1)
        drop = (n_nb >= cfg.forget_min) & (n_dis >= cfg.forget_frac * n_nb)
        if drop.any():
            self._event(i, "forgot", f"Dropped {int(drop.sum())} older labels that recent teacher labels contradict")
        return [k for k, d in zip(old, drop) if not d] + rec

    def _maybe_retrain(self, i: int) -> None:
        cfg, m = self.cfg, self.m
        if not m.retrain or i + 1 >= self.n:
            return
        known = self.I.student.known
        novel = [lab for lab, c in self.teacher_seen.items()
                 if lab in self.intents and lab not in known and c >= cfg.novel_min]
        fresh = self.labels_since >= cfg.hot_min_labels
        if novel and fresh:
            self._retrain(i, f"new intent seen {cfg.novel_min}+ times: {', '.join(sorted(novel))}")
        elif self.labels_since >= cfg.retrain_every:
            self._retrain(i, f"{cfg.retrain_every} new labels")
        elif m.hot_cell_retrain and fresh:
            hot = np.flatnonzero(np.maximum(self.Fi.local, self.Fs.local) >= cfg.hot_threshold)
            if len(hot):
                self._retrain(i, f"cortisol high in {len(hot)} cell(s)")

    def _audit_p(self, c: int, nov: float) -> float:
        cfg = self.cfg
        if not self.m.directed_audits:
            return cfg.audit_base
        stress = max(0.0, self.Fi.local[c]) + max(0.0, self.Fs.local[c]) + nov
        return float(np.clip(cfg.audit_base + cfg.audit_gain * stress, cfg.audit_base, cfg.audit_max))

    def _watch(self, i: int) -> None:
        if self.m.local:
            d = np.maximum(self.Fi.local, self.Fs.local)
            hot = set(np.flatnonzero(d >= self.cfg.hot_threshold).tolist())
            new = hot - self.hot_cells
            if new:
                self._event(i, "spike", f"Cortisol spike in {len(new)} cell(s)", cells=sorted(new))
            if self.hot_cells and not hot:
                self._event(i, "calm", "All cells back below the alarm level")
            self.hot_cells = hot
        for name, F in (("scope", self.Fs), ("intent", self.Fi)):
            g = F.global_level
            if g >= 0.9 and not self.alarm[name]:
                self._event(i, "global_alarm", f"Global {name} cortisol at {g:.2f}: nearly everything escalates")
                self.alarm[name] = True
            elif g < 0.75 and self.alarm[name]:
                self._event(i, "global_calm", f"Global {name} cortisol back to {g:.2f}")
                self.alarm[name] = False

    def _observe(self, i: int) -> None:
        self.pool.append(i)
        self.labels_since += 1
        lab = self.y_teacher[i]
        st = getattr(self.I, "student", None)
        if i >= self.warm and lab in self.intents and lab not in self.teacher_seen \
                and st is not None and lab not in st.known:
            self._event(i, "novel_label", f"Teacher used an intent the student has never seen: {lab}", label=lab)
        if self.m.semantic and hasattr(self, "sem") and i >= self.warm and lab in self.intents \
                and st is not None and lab not in st.known:
            self.sem.inject(self.I.index[lab], self.cfg.sem_new, mode="max")          # (a) a new intent appears
        self.teacher_seen[lab] += 1

    def _declare(self, i: int) -> None:
        """A new product is declared at the start of the launch phase. Its synthetic examples (made from the intent's name, a
        one-line description and the fixed few-shot example only) join training at once, the student is retrained with the
        balancing weights, and the intents start on probation (stress at its maximum) until 30 teacher-seen answers agree."""
        self.declared = True
        for lab, exs in self.cfg.seed_examples.items():
            if lab not in self.I.index:
                continue
            for t in exs:
                self.extra.append((t, lab))
        if self.extra:
            E = embed.encode([t for t, _ in self.extra])
            by: dict[int, list[int]] = {}
            for j, (_, lab) in enumerate(self.extra):
                by.setdefault(self.I.index[lab], []).append(j)
            self.extra_emb = {k: E[v] for k, v in by.items()}
        sem = getattr(self, "sem", None)
        if sem is not None:
            for lab in self.cfg.seed_examples:
                if lab in self.I.index:
                    sem.inject(self.I.index[lab], self.cfg.sem_new, mode="max")
        self._event(i, "declared", f"{len(self.cfg.seed_examples)} new intents declared with {len(self.extra)} synthetic examples")
        if self.extra:
            self._retrain(i, "new intents declared with synthetic seeds")

    def _sense(self, i: int) -> None:
        """(d) An escalated query outside every intent's 95th-percentile radius joins a cluster; a
        cluster of `sem_cluster` is shown to the LLM once and the intents it names get stress."""
        e = self.E[i]
        d = 1.0 - self.cen @ e
        if np.any(d[self.have] <= self.rad[self.have]):
            return
        members = self.clusters.add(i, e)
        if members is None:
            return
        from .teacher import CapExceeded
        try:
            names = self.sensor.ask([self.texts[k] for k in members[: self.cfg.sem_cluster]], self.intents)
        except CapExceeded:
            self.sensor_stop = True
            return
        self.sensor_calls += 1
        for nm in names:
            self.sem.inject(self.I.index[nm], self.cfg.sem_sensor, spread=False, mode="max")
        self._event(i, "sensor", f"LLM sensor on {len(members)} unfamiliar queries: {', '.join(names) or 'no intents named'}",
                    intents=names)

    # ------------------------------------------------------------------ main loop
    def run(self) -> RunResult:
        cfg, m = self.cfg, self.m
        cols = ("decision", "audited", "llm_call", "final", "cost_usd", "level_scope", "level_intent",
                "glevel_scope", "glevel_intent", "u_scope", "u_intent", "cell", "version", "novelty",
                "audit_p", "leak_teacher", "probation", "extra_calls")
        rec = {k: [] for k in cols}
        for i in range(self.n):
            y = self.y_teacher[i]
            valid = y == OOS or y in self.intents
            if m.llm_only or i < self.warm:
                if valid:
                    self._observe(i)
                row = dict(decision="history" if i < self.warm else "llm", audited=False, llm_call=True,
                           final=y, cost_usd=self.cost[i], level_scope=np.nan, level_intent=np.nan,
                           glevel_scope=np.nan, glevel_intent=np.nan, u_scope=np.nan, u_intent=np.nan,
                           cell=-1, version=0, novelty=0.0, audit_p=0.0, leak_teacher=False,
                           probation=False, extra_calls=0.0)
                for k in cols:
                    rec[k].append(row[k])
                if not m.llm_only and i == self.warm - 1:
                    self._bootstrap(i)
                continue

            if self.cfg.seed_examples is not None and not self.declared and self.phase_arr[i] == "launch":
                self._declare(i)
            c = int(self.cell[i])
            nov = float(self.nov[i]) if m.novelty else 0.0
            ls, li = self.Fs.level(c, nov), self.Fi.level(c, nov)
            if cfg.router == "oracle":
                li = 0.0
            us, ui = float(self.S.u[i]), float(self.I.u[i])
            scope_top, intent_top = self.S.top(i), self.I.top(i)
            feat = None
            if cfg.router == "learned":
                feat = self._feat(i, self.I.P[i], self.ex_emb, self.I.index[intent_top])
                sc = self.router.score(feat)
                win = np.asarray(self.rwin[-2000:])
                ui = float((win < sc).mean()) if len(win) else 0.5
                self.rwin.append(sc)
            elif cfg.router == "oracle":
                ui = 0.0 if intent_top == y else 1.5       # escalate exactly the queries the student gets wrong
            scope_ok = bool(accepts(us, ls))
            intent_ran = scope_ok and scope_top == IN
            sem = self.sem if m.semantic else None
            ki = self.I.index[intent_top]
            li_base, on, fit_fail = li, False, False
            if sem is not None:
                on = intent_ran and sem.probation(ki)
                li = li + sem.level(ki)
                if intent_ran and ki in self.ex_emb:         # fit check: far from every training example of its predicted intent
                    fit_fail = 1.0 - float((self.E[i] @ self.ex_emb[ki].T).max()) > self.fit_thr
                    if fit_fail:
                        li += cfg.sem_qfit
            intent_ok = intent_ran and bool(accepts(ui, li))

            if scope_ok and scope_top == OOS:
                escalate, final = False, OOS
            elif intent_ok:
                escalate, final = False, intent_top
            else:
                escalate, final = True, y

            if escalate:
                observed, p_obs, audited, p_a = True, 1.0, False, 0.0
            else:
                p_a = self._audit_p(c, nov)
                if on:
                    p_a = max(p_a, cfg.sem_audit)
                audited = bool(self.rng.random() < p_a)
                observed, p_obs = audited, p_a
            w = (1.0 / p_obs) if (observed and valid) else 0.0
            extra = 0.0
            if on:      # LLM calls the probation caused: extra escalations, and audits above the base rate
                if escalate and bool(accepts(ui, li_base)):
                    extra = 1.0
                elif audited:
                    extra = 1.0 - self._audit_p(c, nov) / p_a

            # --- scope gate evidence (every query is in its population)
            truth_s = (OOS if y == OOS else IN) if (observed and valid) else None
            wrong_s = float(scope_top != truth_s) if truth_s is not None else np.nan
            scope_used = not escalate                       # the answer relied on the scope gate
            self.Fs.update_guard((wrong_s if scope_used and truth_s is not None else 0.0) * w)
            if truth_s is not None:
                self.Fs.update_local(c, wrong_s, float(self.S.expected(us)))
            self.S.log(i, us, wrong_s, w)

            # --- intent gate evidence (population: in-scope queries)
            truth_i = y if (observed and y in self.intents) else None
            wrong_i = float(intent_top != truth_i) if truth_i is not None else np.nan
            if intent_ran:
                self.Fi.update_guard((wrong_i if intent_ok and truth_i is not None else 0.0) * w)
            if truth_i is not None:
                self.Fi.update_local(c, wrong_i, float(self.I.expected(ui)))
            if intent_ran or truth_i is not None:
                self.I.log(i, ui, wrong_i, w if truth_i is not None else 0.0)

            if feat is not None and truth_i is not None:       # learned router: train on teacher-seen queries
                self.router.add(feat, wrong_i, w)
                self.err_rate[ki] = 0.9 * self.err_rate[ki] + 0.1 * wrong_i
            if sem is not None:
                if fit_fail:                                          # (b) fit check, label-free
                    sem.inject(ki, cfg.sem_fit)
                if truth_i is not None:
                    if escalate:
                        self.esc_log.append((ki, self.I.index[y]))
                    if sem.observe(ki, wrong_i == 0.0):
                        self._event(i, "released", f"{intent_top} released from probation")
                    sem.inject(ki, cfg.sem_resid * (wrong_i - float(self.I.expected(ui))))      # (c) residual
                if m.sensor and escalate and truth_i is not None and not self.sensor_stop and self.sensor is not None:
                    self._sense(i)
                sem.tick()
            self.Fs.tick()
            self.Fi.tick()
            if observed and valid:
                self._observe(i)
            leak_t = (not escalate) and observed and valid and final != y
            row = dict(decision="escalate" if escalate else "accept", audited=audited, llm_call=observed,
                       final=final, cost_usd=self.cost[i] if observed else 0.0, level_scope=ls, level_intent=li,
                       glevel_scope=self.Fs.global_level, glevel_intent=self.Fi.global_level, u_scope=us,
                       u_intent=ui, cell=c, version=self.I.student.version, novelty=float(self.nov[i]),
                       audit_p=p_a, leak_teacher=leak_t, probation=on, extra_calls=extra)
            for k in cols:
                rec[k].append(row[k])

            self._watch(i)
            if (i - self.warm) % cfg.refresh == 0:
                self._update_tonic(i)
            if (i - self.warm) % cfg.snapshot_every == 0:
                self.snapshots.append({"t": int(self.t_of[i]),
                                       "local_intent": self.Fi.local.round(4).tolist(),
                                       "local_scope": self.Fs.local.round(4).tolist(),
                                       "tonic_intent": self.Fi.tonic, "guard_intent": self.Fi.guard,
                                       "tonic_scope": self.Fs.tonic, "guard_scope": self.Fs.guard})
            self._maybe_retrain(i)

        calls = self.s[["t", "text", "gold", "phase", "is_launch"]].copy()
        calls["teacher"] = self.y_teacher
        for k in cols:
            calls[k] = rec[k]
        calls["correct"] = calls.final == calls.gold
        calls["leak"] = (calls.decision == "accept") & ~calls.correct
        cells = {}
        if hasattr(self, "map"):
            cells = {"order": self.map.order.tolist(), "n": int(self.map.n_cells), "cell": self.cell.tolist()}
        return RunResult(method=m, calls=calls, snapshots=self.snapshots, events=self.events, cells=cells)


def run_method(stream, teacher_df, intents, method: Method, cfg: EngineConfig, call_cost_usd, sensor=None) -> RunResult:
    return Replay(stream, teacher_df, intents, method, cfg, call_cost_usd, sensor).run()


def config_dict(cfg: EngineConfig) -> dict:
    return asdict(cfg)
