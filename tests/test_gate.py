import numpy as np
import pytest

from cortisol.gate import (CortisolField, DoubtScale, FieldConfig, accepts, choose_level, expected_wrong,
                           leak_curve, runner_up, wilson_upper)
from conftest import needs_aci


def test_runner_up():
    P = np.array([[0.7, 0.2, 0.1], [0.4, 0.4, 0.2], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(runner_up(P), [0.2, 0.4, 0.0])


def test_doubt_scale_is_a_percentile():
    s = DoubtScale(np.arange(100) / 100)
    assert s(np.array([-1.0]))[0] == 0.0
    assert s(np.array([0.495]))[0] == pytest.approx(0.5)
    assert s(np.array([5.0]))[0] == 1.0


def test_raising_the_level_never_adds_acceptances():
    u = np.random.default_rng(0).random(500)
    acc = [accepts(u, lv).mean() for lv in np.linspace(0, 1, 11)]
    assert all(a >= b for a, b in zip(acc, acc[1:]))
    assert acc[0] == pytest.approx(1.0, abs=0.01) and acc[-1] == 0.0


def test_wilson_upper_basics():
    assert wilson_upper(np.array([0.0]), np.array([0.0]), 1.645)[0] == 1.0
    ub = wilson_upper(np.array([0.0, 0.0]), np.array([50.0, 5000.0]), 1.645)
    assert 0.03 < ub[0] < 0.08 and ub[1] < 0.001


def test_choose_level_is_cautious_with_little_evidence():
    rng = np.random.default_rng(1)
    u = rng.random(4000)
    wrong = (rng.random(4000) < 0.4 * u ** 3).astype(float)   # doubtful queries are wrong more often
    lvl_many = choose_level(u, wrong, np.ones(4000), alpha=0.03)
    lvl_few = choose_level(u[:200], wrong[:200], np.ones(200), alpha=0.03)
    assert 0 < lvl_many < 1
    assert lvl_few >= lvl_many           # fewer labels, wider bound, more escalation
    L = leak_curve(u, wrong, np.ones(4000))
    assert np.all(np.diff(L) <= 1e-12)   # the bound only falls as the level rises


def test_expected_wrong_tracks_the_calibration_curve():
    rng = np.random.default_rng(2)
    u = rng.random(5000)
    wrong = (rng.random(5000) < u).astype(float)
    f = expected_wrong(u, wrong, np.ones(5000))
    assert f(np.array(0.05)) < 0.2 and f(np.array(0.95)) > 0.8


@needs_aci
def test_guard_never_goes_below_zero_and_rises_on_leaks():
    F = CortisolField(FieldConfig(alpha=0.05, gamma_guard=0.1), np.eye(3), tonic0=0.2)
    for _ in range(50):
        F.update_guard(0.0)
    assert F.guard == 0.0
    F.update_guard(1.0 / 0.1)            # an audited leak at p = 0.1
    assert F.guard > 0.9
    assert F.level(0) == pytest.approx(F.tonic + F.guard)


@needs_aci
def test_local_stress_spreads_to_neighbours_and_decays():
    W = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])
    F = CortisolField(FieldConfig(alpha=0.05, local=True, gamma_local=0.1, decay=0.1), W, tonic0=0.0)
    F.update_local(0, wrong=1.0, expected=0.1)
    assert F.local[0] > F.local[1] > 0 and F.local[2] == 0
    before = F.local[0]
    F.tick()
    assert F.local[0] == pytest.approx(before * 0.9)
