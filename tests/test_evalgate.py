from guarded_distillation.evalgate import check

BASE = {"calls_pct": 48.0}


def ok(rows):
    return [r[3] for r in rows]


def test_passes_inside_all_limits():
    assert all(ok(check({"overall_delta": 0.5, "launch_delta": -1.1, "calls_pct": 49.9}, BASE)))


def test_each_limit_fails_independently():
    assert ok(check({"overall_delta": -0.4, "launch_delta": 0.0, "calls_pct": 48.0}, BASE)) == [False, True, True]
    assert ok(check({"overall_delta": 0.0, "launch_delta": -2.1, "calls_pct": 48.0}, BASE)) == [True, False, True]
    assert ok(check({"overall_delta": 0.0, "launch_delta": 0.0, "calls_pct": 50.1}, BASE)) == [True, True, False]
