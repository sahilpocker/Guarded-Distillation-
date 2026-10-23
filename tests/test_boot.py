import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import boot  # noqa: E402


def arrs(correct):
    c = np.array(correct, bool)
    return {"correct": c, "is_launch": np.ones(len(c), bool), "first": np.ones(len(c), bool)}


def test_paired_bootstrap_detects_a_real_difference_and_not_a_null_one():
    rng = np.random.default_rng(1)
    base = rng.random(600) < 0.8
    better = base | (rng.random(600) < 0.2)                       # A fixes some of B's mistakes, never breaks any
    r = boot.paired([arrs(better)] * 3, [arrs(base)] * 3, "correct")
    assert r["lo"] > 0 and r["wins"] == 3 and r["n"] == 1800
    same = boot.paired([arrs(base)] * 2, [arrs(base)] * 2, "correct")
    assert same["mean"] == 0 and same["lo"] == 0 == same["hi"]
    masked = boot.paired([arrs(better)], [arrs(base)], "correct", ["first", "is_launch"])
    assert masked["n"] == 600
