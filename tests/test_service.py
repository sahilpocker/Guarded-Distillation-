import hashlib
import json

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from guarded_distillation import embed  # noqa: E402
from guarded_distillation.serve import monitor  # noqa: E402
from guarded_distillation.serve.app import make_app  # noqa: E402
from guarded_distillation.serve.bundle import build_bundle  # noqa: E402

INTENTS = ["card_arrival", "exchange_rate", "lost_or_stolen_card"]
TEXT = {"card_arrival": ["where is my card", "card has not arrived", "my new card is late", "card delivery status"],
        "exchange_rate": ["what is the exchange rate", "euro to dollar rate", "currency rate today", "rate for pounds"],
        "lost_or_stolen_card": ["i lost my card", "my card was stolen", "card missing help", "stolen card block it"]}


def fake_encode(texts, model=embed.MODEL):
    out = []
    for t in texts:
        v = np.zeros(48)
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 48] += 1.0
        out.append(v / (np.linalg.norm(v) or 1.0))
    return np.array(out)


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(embed, "encode", fake_encode)
    texts, labels = [], []
    for _ in range(40):
        for lab, ts in TEXT.items():
            for t in ts:
                texts.append(t)
                labels.append(lab)
    for t in ["what is the capital of peru", "tell me a joke", "who won the game"] * 30:
        texts.append(t)
        labels.append("out_of_scope")
    b = build_bundle(texts, labels, INTENTS, version="test")
    b.save(tmp_path / "bundle")
    cache = tmp_path / "labels.jsonl"
    cache.write_text("\n".join(json.dumps({"text": t, "label": l, "usd": 0.0}) for t, l in zip(texts, labels)))
    import guarded_distillation.serve.teacher_access as ta
    monkeypatch.setattr(ta.TeacherAccess, "__init__", _init_with(cache, tmp_path), raising=True)
    return TestClient(make_app(tmp_path / "bundle", db=str(tmp_path / "s.db"), live=False))


def _init_with(cache, tmp):
    from guarded_distillation.serve import teacher_access as ta
    orig = ta.TeacherAccess.__init__

    def init(self, intents, cache=cache, cap_usd=1.0, ledger=None, fewshot="x", live=False):
        orig(self, intents, cache=cache, cap_usd=cap_usd, ledger=tmp / "spend.json", fewshot=tmp / "none.csv", live=False)
    return init


def test_landing_page_and_alerts_route(client):
    h = client.get("/")
    assert h.status_code == 200 and "Route it" in h.text and "test" in h.text
    assert client.get("/alerts").json()["alerts"] == []


def test_health_triage_and_metrics(client):
    assert client.get("/healthz").json()["model_version"] == "test"
    r = client.post("/triage", json={"text": "where is my card"}).json()
    assert r["intent"] in INTENTS + ["out_of_scope"] and r["decision"] in ("accept", "escalate") and r["model_version"] == "test"
    for t in ["card has not arrived", "stolen card block it", "tell me a joke"] * 40:
        client.post("/triage", json={"text": t})
    m = client.get("/metrics").json()
    assert m["decisions_total"] == 121 and 0.0 <= m["escalation_rate"] <= 1.0
    assert m["leak_budget"] == pytest.approx(0.04) and "top" in m["stress"] and m["spend"]["cap_usd"] == 1.0


def test_unknown_text_without_live_api_degrades_instead_of_failing(client):
    r = client.post("/triage", json={"text": "zzzz qqqq unseen words"})
    assert r.status_code == 200
    j = r.json()
    if j["decision"] == "escalate":
        assert j.get("degraded")            # no cache hit and live calls are off: best-effort student answer, flagged


def test_alert_rules():
    base = {"stress": {"spike": False, "max": 0.1, "top": [], "spike_threshold": 0.6}, "leak_over_budget": False, "rollbacks": 0,
            "spend": {"cap_usd": 1.0, "today_usd": 0.1}, "estimated_leak_rate": 0.0, "leak_budget": 0.04, "window": 500, "model_version": "v1"}
    assert monitor.alerts(base) == []
    hot = {**base, "stress": {"spike": True, "max": 0.8, "top": [{"intent": "card_arrival", "stress": 0.8}], "spike_threshold": 0.6},
           "leak_over_budget": True, "estimated_leak_rate": 0.09, "spend": {"cap_usd": 1.0, "today_usd": 0.9}}
    assert {a["kind"] for a in monitor.alerts(hot)} == {"stress_spike", "leak_over_budget", "spend"}


def test_admin_routes_need_a_token(client, monkeypatch):
    assert client.get("/admin/export").status_code == 401
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    assert client.get("/admin/export", headers={"x-admin-token": "t"}).status_code == 200
