import numpy as np
import pandas as pd
import pytest

from cortisol.data import Phase, StreamConfig, build_stream
from cortisol.engine import METHODS, EngineConfig, run_method
from cortisol.report import summarise
from cortisol.teacher import SimulatedTeacher, build_system_prompt, precompute
from conftest import needs_aci


@pytest.fixture
def setup(tiny_pools):
    cfg = StreamConfig(history=500, n=900, shots=1, seed=0, phases=(
        Phase("steady", 0.0, (0.0, 0.0), 0.02), Phase("launch", 0.4, (0.3, 0.1), 0.02),
        Phase("oos_surge", 0.7, (0.05, 0.05), 0.3), Phase("settle", 0.85, (0.05, 0.05), 0.03)))
    stream, few = build_stream(tiny_pools, cfg)
    T = SimulatedTeacher(pd.concat([tiny_pools.banking, tiny_pools.oos]), acc=0.95)
    tdf = precompute(stream, T, build_system_prompt(tiny_pools.intents, few))
    ecfg = EngineConfig(n_cells=8, retrain_every=150, window=600, hot_min_labels=20, novel_min=5,
                        alpha_scope=0.05, alpha_intent=0.05)
    return stream, tdf, tiny_pools.intents, ecfg


def test_llm_only_calls_the_teacher_every_time(setup):
    stream, tdf, intents, ecfg = setup
    r = run_method(stream, tdf, intents, METHODS["llm_only"], ecfg, np.ones(len(stream)))
    live = r.calls[r.calls.phase != "history"]
    assert live.llm_call.all() and (live.final == live.teacher).all()


@needs_aci
@pytest.mark.parametrize("name", ["static", "retrain", "global", "cortisol"])
def test_gated_methods_run_and_save_llm_calls(setup, name):
    stream, tdf, intents, ecfg = setup
    r = run_method(stream, tdf, intents, METHODS[name], ecfg, np.ones(len(stream)))
    live = r.calls[r.calls.phase != "history"]
    assert len(live) == 900
    assert live.llm_call.mean() < 1.0
    assert live.correct.mean() > 0.6
    assert any(e["kind"] == "golive" for e in r.events)
    if METHODS[name].retrain:
        assert any(e["kind"] == "promoted" for e in r.events)   # the launch intents get learned


@needs_aci
def test_summary_has_every_method(setup):
    stream, tdf, intents, ecfg = setup
    res = {m: run_method(stream, tdf, intents, METHODS[m], ecfg, np.ones(len(stream))) for m in ("llm_only", "cortisol")}
    s = summarise(res)
    assert s["llm_only"]["accuracy_delta_pts"] == 0
    assert set(s["cortisol"]["by_phase"]) == {"steady", "launch", "oos_surge", "settle"}
