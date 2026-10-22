import json

import pandas as pd
import pytest

from cortisol.data import load_fewshot
from cortisol.teacher import CapExceeded, Ledger, LLMTeacher, PRICES


def test_ledger_cap_reserves_in_flight_cost(tmp_path):
    led = Ledger(tmp_path / "spend.json", cap_usd=1.0)
    assert led.reserve("m", 0.6)
    assert not led.reserve("m", 0.6)          # 0.6 in flight + 0.6 would pass the cap
    led.settle("m", 0.6, 0.5)
    assert led.spent("m") == pytest.approx(0.5)
    assert Ledger(tmp_path / "spend.json").spent("m") == pytest.approx(0.5)   # persisted


def test_fewshot_is_chosen_once_and_reused(tiny_pools, tmp_path):
    path = tmp_path / "fewshot.csv"
    a = load_fewshot(tiny_pools, 1, path)
    b = load_fewshot(tiny_pools, 1, path)
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError):
        load_fewshot(tiny_pools, 2, path)      # saved file does not match --shots


def test_readonly_teacher_never_calls(tmp_path):
    t = LLMTeacher("anthropic", "m", "sys", ["a"], tmp_path / "c.jsonl", readonly=True)
    t._client_get = lambda: pytest.fail("a readonly teacher must not build a client")
    with pytest.raises(RuntimeError, match="label-pool"):
        t.label_all(["hello"])


def test_cap_stops_calls_and_failures_are_not_cached(tmp_path):
    cache = tmp_path / "c.jsonl"
    led = Ledger(tmp_path / "spend.json", cap_usd=0.0)
    t = LLMTeacher("anthropic", "m", "sys", ["a"], cache, price=PRICES["haiku-5.5"], ledger=led, max_workers=1)
    t._call = lambda text: pytest.fail("the cap must stop the call before it is made")
    with pytest.raises(CapExceeded):
        t.label_all(["x", "y"], progress=False)
    assert not cache.exists() or cache.read_text() == ""
