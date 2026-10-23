"""The cortisol gate.

DOUBT. For a query x the student gives probabilities over labels. Its doubt is the runner-up
probability p_2(x): how strongly the second-best label competes. Doubt is converted to a
percentile u(x) in [0, 1] against the doubts of recent traffic (label-free), so that levels
read the same way across retrains: a level of 0.4 means "escalate the 40% most doubtful".

GATE. Accept the student's answer when u(x) < 1 - level(x); escalate to the teacher otherwise.
In lesson-1 terms the prediction set holds the top label plus every label that clears the bar;
the gate accepts when the set is a single label. Raising the level only adds escalations.

LOSS. A mistake only costs something when the student's answer is used. The quantity each gate
controls is the LEAK rate: accepted and wrong (the teacher's label is the reference). Escalated
queries have loss 0 for sure. Accepted queries are checked only when audited (probability p),
so the signal is leak_hat = leak * audited / p, an unbiased estimate (see docs/THEORY.md).

CORTISOL. level(x) = tonic + guard + local[cell(x)] + novelty_gain * novelty(x)

  tonic    global, slow. Every `refresh` steps it is set to the lowest level whose leak rate over
           a recent window is at most alpha with 95% confidence (Wilson upper bound). It uses
           every teacher label in the window: escalations (weight 1) also say what would have
           happened at lower levels; audits (weight 1/p) cover the accepted region. Little audit
           evidence means a wide bound, so the tonic stays cautious until audits accumulate.
           In conformal-PID terms this is the "scorecaster".
  guard    global, fast, never below 0. One aci_update per step with leak_hat: the lesson-2
           rule. It adds caution when realised leaks run above alpha (whatever the tonic thinks),
           and it is what carries the long-run guarantee: average leak <= alpha + O(1/T),
           whatever the data does (docs/THEORY.md). Without the floor it would also spend the
           tonic's safety margin whenever audits happen to find few leaks.
  local    per cell of the map space. Moves on every teacher label in the cell by how much more
           often the student was wrong than its doubt predicted (wrong - expected(u)), spreads to
           neighbouring cells through the kernel W, and decays. Cells where the student is
           confidently wrong (a new product, a new kind of question) heat up on their own.
  novelty  label-free; only ever adds caution.

The guard's guarantee survives the other terms because they are bounded: once the guard is
large enough that the level is >= 1 everywhere, nothing is accepted, no leak can occur, and the
guard falls again.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEVEL_GRID = np.linspace(0.0, 1.0, 101)


def aci_update(q, err, alpha, gamma):
    """One step of online quantile tracking (adaptive conformal inference) for cortisol.

    q      current cortisol: a float, or a numpy array (one value per cell)
    err    the miss signal for this step: 1 = miss, 0 = hit, or an importance-weighted
           estimate err/p >= 0 when the label was only seen with probability p
    alpha  target miss rate (a float), or an array of per-cell targets
    gamma  step size: a float, or an array the same shape as q (the local field passes
           gamma_local * W[cell], so each cell moves by its kernel weight)

    Returns the new q. A miss (err=1) raises q by gamma*(1-alpha); a hit (err=0) lowers it by
    gamma*alpha. Works elementwise when q, alpha and gamma are numpy arrays; inputs are not modified.
    """
    return q + gamma * (err - alpha)


def runner_up(P: np.ndarray) -> np.ndarray:
    """Doubt: the second-highest probability in each row."""
    if P.shape[1] == 1:
        return np.zeros(len(P))
    return np.partition(P, -2, axis=1)[:, -2]


def doubt(P: np.ndarray, kind: str = "margin") -> np.ndarray:
    """Doubt score per row of P: margin (runner-up probability), entropy, or maxprob (1 - top probability)."""
    if kind == "entropy":
        return -(P * np.log(P + 1e-12)).sum(1)
    if kind == "maxprob":
        return 1.0 - P.max(1)
    return runner_up(P)


class DoubtScale:
    """Maps doubt to its percentile among a reference set of recent (unlabelled) doubts."""

    def __init__(self, ref: np.ndarray):
        self.ref = np.sort(np.asarray(ref, dtype=float))

    def __call__(self, d: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.ref, d, side="right") / max(len(self.ref), 1)


def accepts(u: float | np.ndarray, level: float | np.ndarray):
    return u < 1.0 - level


def wilson_upper(p: np.ndarray, n: np.ndarray, z: float) -> np.ndarray:
    """One-sided Wilson upper bound for a rate p seen on n (effective) samples; 1 where n = 0."""
    n = np.asarray(n, dtype=float)
    safe = np.maximum(n, 1e-9)
    centre = p + z * z / (2 * safe)
    spread = z * np.sqrt(np.clip(p * (1 - p), 0, None) / safe + z * z / (4 * safe * safe))
    ub = (centre + spread) / (1 + z * z / safe)
    return np.where(n > 0, np.clip(ub, 0.0, 1.0), 1.0)


def leak_curve(u: np.ndarray, wrong: np.ndarray, w: np.ndarray, grid: np.ndarray = LEVEL_GRID,
               z: float = 1.645) -> np.ndarray:
    """Upper confidence bound on the leak rate at every level on the grid.

    At level l the accepted region is u < 1 - l. leak(l) = share of traffic in the region
    x error rate inside it. Both are weighted estimates (w = 1/p: audited queries stand in for
    the unaudited ones). The error rate gets a Wilson upper bound on its effective sample size,
    so a region with little audit evidence is treated as risky rather than safe: picking the
    lowest level from a noisy point estimate would systematically under-escalate.
    """
    if len(u) == 0 or w.sum() <= 0:
        return np.ones_like(grid)
    acc = u[None, :] < (1.0 - grid[:, None])
    sw = (acc * w[None, :]).sum(1)
    sw2 = (acc * (w * w)[None, :]).sum(1)
    swr = (acc * (w * wrong)[None, :]).sum(1)
    frac = sw / w.sum()
    rate = np.where(sw > 0, swr / np.maximum(sw, 1e-12), 0.0)
    n_eff = np.where(sw2 > 0, sw * sw / np.maximum(sw2, 1e-12), 0.0)
    return frac * wilson_upper(rate, n_eff, z)


def choose_level(u: np.ndarray, wrong: np.ndarray, w: np.ndarray, alpha: float,
                 grid: np.ndarray = LEVEL_GRID, z: float = 1.645) -> float:
    """Lowest level whose leak upper bound is <= alpha (1.0 if none is)."""
    L = leak_curve(u, wrong, w, grid, z)
    ok = np.flatnonzero(L <= alpha)
    return float(grid[ok[0]]) if len(ok) else 1.0


def expected_wrong(u: np.ndarray, wrong: np.ndarray, w: np.ndarray, bins: int = 10):
    """Calibration curve: how often the student is wrong as a function of its doubt percentile.

    Returns a function u -> expected wrong rate (piecewise constant over quantile bins)."""
    if len(u) < 4 * bins:
        rate = float((wrong * w).sum() / max(w.sum(), 1e-9)) if len(u) else 0.1
        return lambda x: np.full(np.shape(x), rate)
    edges = np.unique(np.quantile(u, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, len(edges) - 2)
    num = np.bincount(idx, weights=wrong * w, minlength=len(edges) - 1)
    den = np.bincount(idx, weights=w, minlength=len(edges) - 1)
    overall = num.sum() / max(den.sum(), 1e-9)
    rates = np.where(den > 0, (num + overall) / (den + 1.0), overall)  # shrink sparse bins
    return lambda x: rates[np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(rates) - 1)]


@dataclass(frozen=True)
class FieldConfig:
    alpha: float
    tonic: bool = True           # windowed scorecaster; False = tonic frozen at its start value
    guard: bool = True           # ACI guard term
    local: bool = False          # per-cell term
    gamma_guard: float = 0.002
    gamma_local: float = 0.03
    decay: float = 0.002         # per step, pulls local terms back toward 0
    novelty_gain: float = 0.0
    guard_floor: float = 0.0     # the guard only ever adds caution; clipping from below keeps
                                 # the long-run guarantee (it can only push the level up)
    local_bounds: tuple[float, float] = (-0.1, 1.0)


class CortisolField:
    def __init__(self, cfg: FieldConfig, W: np.ndarray, tonic0: float):
        self.cfg = cfg
        self.W = W
        self.tonic = float(tonic0)
        self.guard = 0.0
        self.local = np.zeros(W.shape[0])

    def level(self, cell: int, novelty: float = 0.0) -> float:
        lv = self.tonic + self.guard + self.cfg.novelty_gain * novelty
        if self.cfg.local:
            lv += self.local[cell]
        return lv

    @property
    def global_level(self) -> float:
        return self.tonic + self.guard

    def update_guard(self, leak_hat: float) -> None:
        if self.cfg.guard:
            self.guard = max(self.cfg.guard_floor,
                             float(aci_update(self.guard, leak_hat, self.cfg.alpha, self.cfg.gamma_guard)))

    def update_local(self, cell: int, wrong: float, expected: float) -> None:
        if self.cfg.local:
            self.local = aci_update(self.local, wrong, expected, self.cfg.gamma_local * self.W[cell])
            np.clip(self.local, *self.cfg.local_bounds, out=self.local)

    def tick(self) -> None:
        """Once per query: local stress fades unless new evidence keeps it up."""
        if self.cfg.local:
            self.local *= 1.0 - self.cfg.decay
