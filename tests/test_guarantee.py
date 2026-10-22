"""The guard's long-run guarantee, checked by simulation (docs/THEORY.md, claim G).

Whatever the leak process does, with the guard floored at 0 and leaks seen only through
audits (err_hat = leak * audited / p), the average TRUE leak rate ends up at most alpha plus a
term that shrinks with T. Here the data is nasty on purpose: the student's error rate jumps
between regimes and the tonic is badly wrong (always 0).
"""
import numpy as np

from cortisol.gate import CortisolField, FieldConfig
from conftest import needs_aci


@needs_aci
def test_average_leak_is_controlled_under_shifts_and_partial_feedback():
    rng = np.random.default_rng(0)
    alpha, p, T = 0.05, 0.2, 60_000
    F = CortisolField(FieldConfig(alpha=alpha, gamma_guard=0.01), np.eye(1), tonic0=0.0)
    leaks = 0
    for t in range(T):
        regime = (t // 5000) % 3                       # error rate of the student shifts every 5,000 steps
        err_rate = [0.02, 0.30, 0.10][regime]
        u = rng.random()                               # doubt percentile of this query
        wrong = rng.random() < err_rate * (0.2 + 1.6 * u)
        accepted = u < 1.0 - F.level(0)
        leak = accepted and wrong
        leaks += leak
        audited = accepted and rng.random() < p
        F.update_guard((leak and audited) / p)
    assert leaks / T <= alpha + 0.01
