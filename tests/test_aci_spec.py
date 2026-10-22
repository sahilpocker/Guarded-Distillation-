"""Spec for cortisol.gate.aci_update (lesson 2). These fail until you write it.

Run:  python -m pytest -q tests/test_aci_spec.py
"""
import numpy as np
import pytest

from cortisol.gate import aci_update


def test_miss_raises_by_gamma_times_one_minus_alpha():
    assert aci_update(0.5, 1.0, 0.1, 0.2) == pytest.approx(0.5 + 0.2 * 0.9)


def test_hit_lowers_by_gamma_times_alpha():
    assert aci_update(0.5, 0.0, 0.1, 0.2) == pytest.approx(0.5 - 0.2 * 0.1)


def test_balances_exactly_at_rate_alpha():
    q = 0.3
    for t in range(1000):                 # one miss in every ten steps, alpha = 0.1
        q = aci_update(q, 1.0 if t % 10 == 0 else 0.0, 0.1, 0.05)
    assert q == pytest.approx(0.3, abs=1e-9)


def test_importance_weighted_miss():
    # a miss seen through an audit with probability 0.25 counts as err = 1 / 0.25 = 4
    assert aci_update(0.0, 4.0, 0.05, 0.01) == pytest.approx(0.01 * (4.0 - 0.05))


def test_elementwise_on_arrays_with_per_cell_gamma():
    q = np.array([0.0, 0.5, 1.0])
    gamma = np.array([0.1, 0.0, 0.2])     # the local field passes gamma_local * W[cell]
    out = aci_update(q, 1.0, 0.2, gamma)
    np.testing.assert_allclose(out, [0.08, 0.5, 1.16])


def test_array_alpha():
    # the local field also passes a per-step expected error rate as alpha
    out = aci_update(np.zeros(2), 0.0, np.array([0.1, 0.3]), 1.0)
    np.testing.assert_allclose(out, [-0.1, -0.3])


def test_does_not_mutate_input():
    q = np.array([0.1, 0.2])
    aci_update(q, 1.0, 0.1, 0.1)
    np.testing.assert_allclose(q, [0.1, 0.2])
