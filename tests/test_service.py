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


def test_key_with_trailing_whitespace_is_cleaned(monkeypatch, tmp_path, client):
    monkeypatch.setenv("ANTHROPIC_API_KEY", ' "sk-ant-x" \n')
    from guarded_distillation.serve.app import make_app as mk
    app = mk(tmp_path / "bundle", db=str(tmp_path / "k.db"), live=False)
    import os
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-x"
    assert TestClient(app).get("/healthz").json()["key_sanitized"] is True


def test_store_spend_state_and_db_ledger(tmp_path):
    from guarded_distillation.serve.store import Store
    from guarded_distillation.serve.teacher_access import DbLedger
    st = Store(tmp_path / "x.db")
    led = DbLedger(st, cap_usd=0.01)
    assert led.reserve("m", 0.004) and led.reserve("m", 0.004) and not led.reserve("m", 0.004)    # reserved amounts count against the cap
    led.settle("m", 0.004, 0.003)
    led.settle("m", 0.004, 0.003)
    assert led.spent("m") == pytest.approx(0.006) and led.reserve("m", 0.004) and not led.reserve("m", 0.001)
    st2 = Store(tmp_path / "x.db")                              # a "restart": same file, new process state
    assert DbLedger(st2, 0.01).spent("m") == pytest.approx(0.006)
    st.set_state("model_version", "v2")
    assert Store(tmp_path / "x.db").get_state("model_version") == "v2"


@pytest.mark.skipif(not __import__("os").environ.get("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to run against Postgres")
def test_postgres_store_matches_sqlite_behaviour():
    import os
    from guarded_distillation.serve.store import Store
    st = Store(url=os.environ["TEST_DATABASE_URL"])
    n0 = st.count()
    st.add_decision("pgtest", {"text": "hello", "teacher_label": "card_arrival", "source": "cache", "decision": "escalate"})
    st.add_event("promoted", {"to": "pgtest"})
    st.spend_add("2000-01-01", "m", 0.5)
    st.spend_add("2000-01-01", "m", 0.25)
    st.set_state("k", "a")
    st.set_state("k", "b")
    assert st.count() == n0 + 1 and st.last(1)[0]["text"] == "hello" and st.last(1, "pgtest")[0]["version"] == "pgtest"
    assert st.events("promoted", 1)[0]["to"] == "pgtest" and st.spend_today("2000-01-01") == pytest.approx(0.75)
    assert st.get_state("k") == "b" and {"text": "hello", "label": "card_arrival"} in st.labels()


def test_sliding_window_and_live_limits(monkeypatch):
    from guarded_distillation.serve.limits import Limits, SlidingWindow
    w = SlidingWindow(3, 10)
    assert [w.allow("a", now=t) for t in (0, 1, 2, 3)] == [True, True, True, False] and w.allow("b", now=3)
    assert w.allow("a", now=10.5) and w.retry_after("a", now=11) >= 1
    monkeypatch.setenv("LIVE_PER_IP_PER_HOUR", "2")
    monkeypatch.setenv("LIVE_GLOBAL_PER_HOUR", "3")
    lim = Limits()
    assert [lim.allow_live("x") for _ in range(3)] == [True, True, False]       # per visitor
    assert lim.allow_live("y") and not lim.allow_live("z")                      # global ceiling reached


def test_oversized_text_is_rejected(client):
    assert client.post("/triage", json={"text": "x" * 401}).status_code == 422


def test_webdav_replica_round_trip(tmp_path):
    import httpx
    from guarded_distillation.serve.replicate import WebDavReplica
    from guarded_distillation.serve.store import Store
    remote = {}

    def handler(req):
        if req.method == "PUT":
            if "/dir" not in remote:
                remote["/dir"] = True            # first PUT before the folder exists: 409, then MKCOL, then retry
                return httpx.Response(409)
            remote[req.url.path] = req.content
            return httpx.Response(201)
        if req.method == "MKCOL":
            return httpx.Response(201)
        if req.method == "GET":
            return httpx.Response(200, content=remote[req.url.path]) if req.url.path in remote else httpx.Response(404)
        return httpx.Response(405)

    rep = WebDavReplica("http://dav.test/dav", "u", "p", "d", interval=999)
    rep._client = lambda: httpx.Client(transport=httpx.MockTransport(handler), auth=rep.auth)
    st = Store(tmp_path / "a.db")
    rep.start(st)
    st.add_decision("v1", {"text": "hi", "decision": "accept"})
    st.spend_add("2026-10-23", "m", 0.5)             # critical: uploaded immediately
    assert rep.pushes >= 1 and rep.last_error is None
    out = tmp_path / "restored.db"
    assert WebDavReplica("http://dav.test/dav", "u", "p", "d").restore(out) is False      # a fresh instance has no transport: fails soft
    rep2 = WebDavReplica("http://dav.test/dav", "u", "p", "d")
    rep2._client = lambda: httpx.Client(transport=httpx.MockTransport(handler))
    assert rep2.restore(out) and Store(out).spend_today("2026-10-23") == pytest.approx(0.5) and Store(out).count() == 1
