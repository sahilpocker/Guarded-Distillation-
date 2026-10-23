import numpy as np

from guarded_distillation.gate import doubt
from guarded_distillation.router import LearnedRouter, features


def test_doubt_kinds_rank_a_confident_row_below_a_doubtful_one():
    P = np.array([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25]])
    for kind in ("margin", "entropy", "maxprob"):
        d = doubt(P, kind)
        assert d[0] < d[1], kind


def test_router_falls_back_to_margin_then_learns_and_weights_matter():
    rng = np.random.default_rng(0)
    r = LearnedRouter(refit_every=50, min_samples=40)
    x = features(np.array([0.5, 0.3, 0.2]), 0.4, 0.1, 8, 0.0)
    assert r.score(x) == x[2]                                    # untrained: max-probability doubt, not margin
    for _ in range(200):                                         # feature 3 (fit distance) predicts error
        dist = rng.random()
        v = features(np.array([0.5, 0.3, 0.2]), dist, 0.1, 8, 0.0)
        r.add(v, float(dist > 0.6), 1.0)
    assert r.model is not None
    near = r.score(features(np.array([0.5, 0.3, 0.2]), 0.1, 0.1, 8, 0.0))
    far = r.score(features(np.array([0.5, 0.3, 0.2]), 0.95, 0.1, 8, 0.0))
    assert far > near
