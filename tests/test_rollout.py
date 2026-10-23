import hashlib
import json
import tarfile

import numpy as np
import pytest

from guarded_distillation import embed
from guarded_distillation.serve import rollout
from guarded_distillation.serve.bundle import build_bundle
from guarded_distillation.serve.evalbundle import gate
from guarded_distillation.serve.store import Store
from guarded_distillation.serve.teacher_access import TeacherAccess
from guarded_distillation.serve.triage import Triage

INTENTS = ["card_arrival", "exchange_rate"]


def fake_encode(texts, model=embed.MODEL):
    out = []
    for t in texts:
        v = np.zeros(32)
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 32] += 1.0
        out.append(v / (np.linalg.norm(v) or 1.0))
    return np.array(out)


def bundle(version):
    texts = ["where is my card", "card late", "exchange rate today", "rate for euros"] * 40 + ["tell me a joke"] * 30
    labels = ["card_arrival", "card_arrival", "exchange_rate", "exchange_rate"] * 40 + ["out_of_scope"] * 30
    return build_bundle(texts, labels, INTENTS, version=version)


@pytest.fixture()
def tri(monkeypatch, tmp_path):
    monkeypatch.setattr(embed, "encode", fake_encode)
    t = TeacherAccess(INTENTS, cache=tmp_path / "c.jsonl", ledger=tmp_path / "l.json", fewshot=tmp_path / "none.csv", live=False)
    return Triage(bundle("v1"), t, Store(tmp_path / "s.db"))


def fill(tri, version, audited, disagree, n=500):
    for i in range(n):
        rec = {"decision": "accept", "audited": i < audited, "teacher_label": "card_arrival" if i < audited else None,
               "disagree": i < disagree, "p_audit": 0.08, "text": f"q{i}"}
        tri.store.add_decision(version, rec)


def test_rollback_when_audit_disagreement_rises_by_half_or_more(tri):
    fill(tri, "v1", audited=40, disagree=2)                 # 5% before
    info = rollout.swap(tri, bundle("v2"))
    assert info["from"] == "v1" and tri.b.version == "v2" and info["previous_audit_rate"] == pytest.approx(0.05)
    fill(tri, "v2", audited=40, disagree=6)                 # 15%: three times the old rate
    tri.n_cur = 500
    out = rollout.maybe_rollback(tri)
    assert out and tri.b.version == "v1" and "v2" in tri.blocked
    kinds = [e["kind"] for e in tri.store.events(limit=10)]
    assert "rollback" in kinds and "audit_check" in kinds


def test_no_rollback_when_the_new_version_is_as_good(tri):
    fill(tri, "v1", audited=40, disagree=2)
    rollout.swap(tri, bundle("v2"))
    fill(tri, "v2", audited=40, disagree=2)                 # 5% again
    tri.n_cur = 500
    assert rollout.maybe_rollback(tri) is None and tri.b.version == "v2" and tri.checked
    assert rollout.maybe_rollback(tri) is None             # acts once


def test_no_decision_before_500_queries_or_with_too_few_audits(tri):
    fill(tri, "v1", audited=40, disagree=0)
    rollout.swap(tri, bundle("v2"))
    fill(tri, "v2", audited=10, disagree=10, n=500)
    tri.n_cur = 499
    assert rollout.maybe_rollback(tri) is None and not tri.checked
    tri.n_cur = 500
    assert rollout.maybe_rollback(tri) is None             # only 10 audited answers: not enough evidence
    assert tri.b.version == "v2"


def test_no_rollback_without_a_baseline_from_the_old_version(tri):
    rollout.swap(tri, bundle("v2"))                          # v1 served no audited traffic: nothing to compare with
    assert tri.prev["audited"] == 0
    fill(tri, "v2", audited=40, disagree=20)
    tri.n_cur = 500
    assert rollout.maybe_rollback(tri) is None and tri.b.version == "v2"
    ev = [e for e in tri.store.events("audit_check")]
    assert ev and ev[0]["baseline"] is False and ev[0]["worse"] is False


def test_blocked_versions_are_never_promoted_again(tri, monkeypatch, tmp_path):
    tri.blocked.add("v2")
    calls = []
    reg = rollout.Registry("o/r", tmp_path)
    monkeypatch.setattr(reg, "releases", lambda: [{"n": 2, "tag": "model-v2", "url": "u", "sha256": "0" * 64, "body": ""}])
    monkeypatch.setattr(reg, "fetch", lambda rel: calls.append(rel))
    assert rollout.poll_once(tri, reg)["updated"] is False and not calls


def test_registry_checks_the_hash_and_unpacks_a_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "encode", fake_encode)
    b = bundle("v2")
    b.save(tmp_path / "v2")
    tar = tmp_path / "bundle-v2.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        tf.add(tmp_path / "v2", arcname="v2")
    blob = tar.read_bytes()
    reg = rollout.Registry("o/r", tmp_path / "cache")
    monkeypatch.setattr(reg, "_get", lambda url: blob)
    rel = {"tag": "model-v2", "url": "u", "sha256": hashlib.sha256(blob).hexdigest()}
    assert reg.fetch(rel).version == "v2"
    with pytest.raises(ValueError):
        reg.fetch({**rel, "sha256": "0" * 64})


def test_promotion_gate_uses_the_ci_thresholds():
    cur = {"calls_pct": 50.0}
    assert all(r[3] for r in gate({"overall_delta": 0.1, "launch_delta": -1.0, "calls_pct": 51.9}, cur))
    assert [r[3] for r in gate({"overall_delta": -0.4, "launch_delta": -1.0, "calls_pct": 50}, cur)] == [False, True, True]
    assert [r[3] for r in gate({"overall_delta": 0.0, "launch_delta": -2.5, "calls_pct": 50}, cur)] == [True, False, True]
    assert [r[3] for r in gate({"overall_delta": 0.0, "launch_delta": 0.0, "calls_pct": 52.5}, cur)] == [True, True, False]
