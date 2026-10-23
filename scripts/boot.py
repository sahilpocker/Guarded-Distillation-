"""Paired bootstrap over per-query differences between two runs of the same stream (scripts/routing.py, scripts/seeding.py).

Per-query differences are pooled over seeds and resampled with replacement (2,000 draws, fixed seed); the interval is the 2.5th
to 97.5th percentile of the mean difference, in percentage points. Queries are treated as exchangeable, which ignores that
different seeds share most of the question pool, so the intervals are narrower than seed-to-seed spread; the number of seeds in
which A beat B is reported next to them.
"""
import numpy as np


def paired(a_list, b_list, key, mask_key=None, n_boot=2000, seed=0):
    """a_list, b_list: lists of np arrays (per seed). key: array name to difference (bool/float). mask_key: restrict to a boolean array
    (e.g. "first" & "is_launch"). Returns dict(mean, lo, hi, wins, n_seeds, n)."""
    diffs, wins = [], 0
    for A, B in zip(a_list, b_list):
        m = np.ones(len(A[key]), bool) if mask_key is None else np.logical_and.reduce([A[k] for k in mask_key])
        d = A[key][m].astype(float) - B[key][m].astype(float)
        diffs.append(d)
        wins += int(d.mean() > 0)
    d = np.concatenate(diffs)
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {"mean": float(d.mean() * 100), "lo": float(np.percentile(means, 2.5) * 100), "hi": float(np.percentile(means, 97.5) * 100),
            "wins": wins, "n_seeds": len(diffs), "n": int(len(d))}


def fmt(r, digits=2):
    return f"{r['mean']:+.{digits}f} [{r['lo']:+.{digits}f}, {r['hi']:+.{digits}f}] ({r['wins']}/{r['n_seeds']} seeds)"
