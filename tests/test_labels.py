import json

import pandas as pd
import pytest

from guarded_distillation.data import load_fewshot
from guarded_distillation.teacher import CapExceeded, Ledger, LLMTeacher, PRICES


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


def test_sbert_student_runs_without_downloading_the_encoder(monkeypatch):
    """The sbert backend (frozen embeddings + logistic regression) fits, gives full-universe
    probabilities and a novelty signal; the encoder is replaced by a deterministic fake."""
    import hashlib

    import numpy as np

    from guarded_distillation import embed
    from guarded_distillation.student import Student

    def fake_encode(texts, model=embed.MODEL):
        out = []
        for t in texts:
            v = np.zeros(32)
            for w in t.split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 32] += 1.0
            out.append(v / (np.linalg.norm(v) or 1.0))
        return np.array(out)

    monkeypatch.setattr(embed, "encode", fake_encode)
    texts = ["card arrive late", "card not here", "exchange rate euro", "euro dollar rate"] * 10
    labels = ["card", "card", "fx", "fx"] * 10
    st = Student(["card", "fx", "never_seen"], backend="sbert").fit(texts, labels)
    P = st.proba(["my card arrive", "rate of euro"])
    assert P.shape == (2, 3) and P[:, 2].sum() == 0
    assert st.predict(["my card arrive", "rate of euro"]) == ["card", "fx"]
    assert st.oov_fraction(["zzz qqq"])[0] == 1.0
