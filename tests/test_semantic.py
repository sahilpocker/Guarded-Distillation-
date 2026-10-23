import hashlib
from types import SimpleNamespace

import numpy as np

from cortisol import embed
from cortisol.semantic import IntentGraph, OutsideClusters, SemanticField

CFG = SimpleNamespace(sem_gain=1.0, sem_top=3, sem_spread=0.5, sem_hot=0.25, sem_decay=0.002, sem_release=30)


def fake_encode(texts, model=embed.MODEL):
    out = []
    for t in texts:
        v = np.zeros(32)
        for w in t.split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 32] += 1.0
        out.append(v / (np.linalg.norm(v) or 1.0))
    return np.array(out)


def graph(monkeypatch, names=("card_arrival", "card_delivery", "exchange_rate", "top_up")):
    monkeypatch.setattr(embed, "encode", fake_encode)
    return IntentGraph(list(names))


def test_graph_edge_is_max_of_cosine_and_confusion(monkeypatch):
    g = graph(monkeypatch)
    g.rebuild({}, [(2, 3)] * 8 + [(3, 3)] * 2)          # exchange_rate escalations answered as top_up, but not conversely
    assert g.W[2, 3] == g.W[3, 2] >= g.cos[2, 3]
    assert g.conf[2, 3] > 0 and np.allclose(np.diag(g.W), 0)
    idx, w = g.neighbours(0, top=2)
    assert len(idx) == 2 and 0 not in idx and (w >= 0).all()


def test_stress_spreads_decays_and_releases(monkeypatch):
    g = graph(monkeypatch)
    g.rebuild({}, [])
    f = SemanticField(g, CFG)
    f.inject(0, 0.8)
    assert f.stress[0] == 0.8 and 0 < f.stress[1] < 0.8 and f.probation(0)      # card_delivery shares a word
    assert f.level(0) == 0.8
    f.tick()
    assert f.stress[0] < 0.8
    assert not any(f.observe(0, True) for _ in range(29))      # 29 agreements: still on probation
    assert f.observe(0, True) and f.stress[0] == 0 and not f.probation(0)
    f.inject(0, 0.8)
    f.observe(0, True)
    f.observe(0, False)                                          # a disagreement restarts the count
    assert f.agree[0] == 0 and f.probation(0)


def test_max_mode_lifts_without_adding(monkeypatch):
    g = graph(monkeypatch)
    g.rebuild({}, [])
    f = SemanticField(g, CFG)
    f.inject(1, 0.6, mode="max")
    f.inject(1, 0.4, mode="max")
    assert f.stress[1] == 0.6


def test_cluster_fires_once_at_size():
    c = OutsideClusters(sim=0.8, size=5)
    e = np.array([1.0, 0.0])
    fired = [c.add(i, e + 0.01 * i * np.array([0.0, 1.0])) for i in range(8)]
    assert [f is not None for f in fired] == [False] * 4 + [True] + [False] * 3
    assert c.add(99, np.array([0.0, 1.0])) is None and len(c.clusters) == 2
