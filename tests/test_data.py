import pandas as pd

from cortisol.data import Phase, StreamConfig, build_stream
from cortisol.taxonomy import LAUNCH_INTENTS, OOS, ROUTE_OF, route, urgency


def small_cfg(**kw):
    return StreamConfig(history=200, n=600, shots=1, seed=0, phases=(
        Phase("steady", 0.0, (0.0, 0.0), 0.02), Phase("launch", 0.4, (0.3, 0.1), 0.02),
        Phase("oos_surge", 0.7, (0.05, 0.05), 0.3), Phase("settle", 0.85, (0.05, 0.05), 0.03)), **kw)


def test_stream_shape_and_phases(tiny_pools):
    s, few = build_stream(tiny_pools, small_cfg())
    assert len(s) == 800 and (s.phase == "history").sum() == 200
    assert s.t.min() == -200 and s.t.max() == 599
    assert list(dict.fromkeys(s.phase)) == ["history", "steady", "launch", "oos_surge", "settle"]


def test_launch_intents_absent_before_launch(tiny_pools):
    s, _ = build_stream(tiny_pools, small_cfg())
    before = s[s.phase.isin(["history", "steady"])]
    assert not before.gold.isin(LAUNCH_INTENTS).any()
    assert s[s.phase == "launch"].is_launch.mean() > 0.1


def test_oos_surge(tiny_pools):
    s, _ = build_stream(tiny_pools, small_cfg())
    share = s.groupby("phase").gold.apply(lambda g: (g == OOS).mean())
    assert share["oos_surge"] > 3 * share["steady"]


def test_fewshot_examples_never_in_stream(tiny_pools):
    s, few = build_stream(tiny_pools, small_cfg())
    assert len(few) == len(tiny_pools.intents) + 1
    assert not set(few.text) & set(s.text)


def test_same_seed_same_stream(tiny_pools):
    a, _ = build_stream(tiny_pools, small_cfg())
    b, _ = build_stream(tiny_pools, small_cfg())
    pd.testing.assert_frame_equal(a, b)


def test_t0_tables():
    assert route("lost_or_stolen_card") == "fraud_security" and urgency("lost_or_stolen_card") == "high"
    assert route(OOS) == "general_desk"
    assert len(ROUTE_OF) == 78
