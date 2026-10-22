import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import operating_points as op  # noqa: E402


def row(method, calls, delta, lo=None):
    d = {"mean": delta, "min": lo if lo is not None else delta, "max": delta}
    return {"method": method, "student": "tfidf", "alpha": 0.05, "calls": {"mean": calls, "min": calls, "max": calls},
            "delta": d, "errors": {"mean": 20.0, "min": 20.0, "max": 20.0}, "passes": d["min"] >= op.BAR, "seeds": {}}


def test_pareto_and_selection():
    rows = [row("retrain", 50, 0.1), row("global", 60, 0.5), row("cortisol", 70, 0.4), row("global", 40, -0.5, lo=-0.6)]
    op.pareto(rows)
    assert [r["frontier"] for r in rows] == [True, True, False, True]    # dominated only by a cheaper, better point
    assert op.select(rows)["calls"]["mean"] == 50                        # the 40% setting fails the bar on a seed
    assert op.select(rows, {"global"})["calls"]["mean"] == 60
    assert op.select([row("global", 40, -0.5, lo=-0.6)]) is None


def test_svg_has_one_title_per_setting_and_the_default_ring():
    rows = [row("retrain", 50, 0.1), row("global", 60, 0.5)]
    op.pareto(rows)
    s = op.svg(rows, rows[1])
    assert s.count("<title>") == 2 and "DEFAULT" in s and s.startswith("<svg")
