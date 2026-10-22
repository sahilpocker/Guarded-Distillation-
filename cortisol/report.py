"""Turn replay results into numbers people can check: REPORT.md, results.json and replay.html."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import RunResult
from .taxonomy import LAUNCH_INTENTS, OOS, route, urgency

TEMPLATE = Path(__file__).with_name("templates") / "replay.html"
BIN = 50  # queries per bin in the dashboard time series


def _live(calls: pd.DataFrame) -> pd.DataFrame:
    return calls[calls.phase != "history"]


def summarise(results: dict[str, RunResult]) -> dict[str, dict]:
    """Headline numbers per method, measured against the human (gold) labels."""
    base = _live(results["llm_only"].calls) if "llm_only" in results else None
    out = {}
    for name, r in results.items():
        c = _live(r.calls)
        hist = r.calls[r.calls.phase == "history"]
        acc = float(c.correct.mean())
        accepted = c.decision == "accept"
        out[name] = {
            "title": r.method.title,
            "queries": int(len(c)),
            "accuracy": acc,
            "accuracy_delta_pts": (acc - float(base.correct.mean())) * 100 if base is not None else None,
            "llm_share": float(c.llm_call.mean()),
            "escalated_share": float((c.decision == "escalate").mean()),
            "audited_share": float(c.audited.mean()),
            "cheap_share": float((accepted & ~c.audited).mean()),
            "cost_usd": float(c.cost_usd.sum()),
            "cost_per_1k_usd": float(c.cost_usd.sum() / len(c) * 1000),
            "cheap_errors_per_1k": float((accepted & ~c.correct).mean() * 1000),
            "disagree_teacher_per_1k": float((accepted & (c.final != c.teacher)).mean() * 1000),
            "route_accuracy": float((c.final.map(route) == c.gold.map(route)).mean()),
            "urgency_accuracy": float((c.final.map(urgency) == c.gold.map(urgency)).mean()),
            "launch_accuracy": float(c[c.is_launch].correct.mean()) if c.is_launch.any() else None,
            "oos_accuracy": float(c[c.gold == OOS].correct.mean()) if (c.gold == OOS).any() else None,
            "promotions": sum(e["kind"] == "promoted" for e in r.events),
            "onboarding_llm_calls": int(len(hist)),
            "onboarding_cost_usd": float(hist.cost_usd.sum()),
            "by_phase": {
                ph: {"accuracy": float(g.correct.mean()), "llm_share": float(g.llm_call.mean()),
                     "cheap_errors_per_1k": float(((g.decision == "accept") & ~g.correct).mean() * 1000)}
                for ph, g in c.groupby("phase", sort=False)
            },
        }
    return out


def _fmt_pct(x, d=1):
    return "–" if x is None else f"{x * 100:.{d}f}%"


def report_md(summary: dict, meta: dict) -> str:
    order = [m for m in ("llm_only", "static", "retrain", "global", "cortisol") if m in summary]
    lines = [
        f"# Replay report: {meta['run']}",
        "",
        f"Teacher: **{meta['teacher']}**. Stream: {meta['live']:,} live queries after {meta['history']:,} shadow-mode "
        f"queries (Banking77 + CLINC150 out-of-scope, seed {meta['seed']}). Accuracy is against the datasets' "
        "human labels; the system itself only ever sees the teacher's labels.",
        "",
    ]
    if meta["teacher"].startswith("simulated"):
        lines += ["> The teacher is SIMULATED (gold label, wrong mostly on ambiguous queries). These numbers show "
                  "the mechanism; they are not a measurement of any real LLM.", ""]
    lines += [
        "| Method | Accuracy | vs LLM only | LLM calls | Cost / 1k queries | Cheap-path errors / 1k | Launch queries | Off-topic queries | Retrains |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in order:
        s = summary[m]
        d = s["accuracy_delta_pts"]
        lines.append(
            f"| {s['title']} | {_fmt_pct(s['accuracy'])} | {'baseline' if m == 'llm_only' or d is None else f'{d:+.1f} pts'} | "
            f"{_fmt_pct(s['llm_share'], 0)} | ${s['cost_per_1k_usd']:.4f} | {s['cheap_errors_per_1k']:.1f} | "
            f"{_fmt_pct(s['launch_accuracy'])} | {_fmt_pct(s['oos_accuracy'])} | {s['promotions']} |")
    lines += ["", "## By phase (accuracy / LLM calls)", ""]
    phases = list(next(iter(summary.values()))["by_phase"].keys())
    lines.append("| Method | " + " | ".join(phases) + " |")
    lines.append("|---|" + "---|" * len(phases))
    for m in order:
        bp = summary[m]["by_phase"]
        lines.append(f"| {summary[m]['title']} | " + " | ".join(
            f"{_fmt_pct(bp[p]['accuracy'])} / {_fmt_pct(bp[p]['llm_share'], 0)}" for p in phases) + " |")
    lines += ["", "Cheap-path errors: answers the student gave (no LLM) that disagree with the human label. "
              "Cost uses the price profile in results.json; onboarding (shadow mode) cost is reported separately "
              "there.", ""]
    return "\n".join(lines)


def _bins(c: pd.DataFrame) -> dict:
    c = c.reset_index(drop=True)
    b = (np.arange(len(c)) // BIN)
    g = pd.DataFrame({
        "b": b, "llm": c.llm_call.astype(int), "ok": c.correct.astype(int),
        "acc": (c.decision == "accept").astype(int), "aud": c.audited.astype(int),
        "esc": (c.decision == "escalate").astype(int),
        "bad": ((c.decision == "accept") & ~c.correct).astype(int), "usd": c.cost_usd,
        "gs": c.glevel_scope, "gi": c.glevel_intent,
    }).groupby("b")
    s = g.sum(numeric_only=True)
    m = g.mean(numeric_only=True)
    return {
        "n": g.size().tolist(), "llm": s.llm.tolist(), "ok": s.ok.tolist(), "acc": s.acc.tolist(),
        "aud": s.aud.tolist(), "esc": s.esc.tolist(), "bad": s.bad.tolist(),
        "usd": [round(x, 6) for x in s.usd.tolist()],
        "gs": [None if np.isnan(x) else round(float(x), 3) for x in m.gs.tolist()],
        "gi": [None if np.isnan(x) else round(float(x), 3) for x in m.gi.tolist()],
    }


def dashboard_data(results: dict[str, RunResult], summary: dict, meta: dict, focus: str = "cortisol") -> dict:
    methods = {}
    for name, r in results.items():
        c = _live(r.calls)
        methods[name] = {"title": r.method.title, "bins": _bins(c),
                         "events": [e for e in r.events if e["kind"] != "shadow_rejected"]}
    data = {"meta": meta, "summary": summary, "bin": BIN, "methods": methods, "focus": focus}

    fr = results.get(focus)
    if fr is not None and fr.cells:
        c_all = fr.calls.reset_index(drop=True)
        cell = np.asarray(fr.cells["cell"])
        hist = c_all.phase == "history"
        live = ~hist
        order = fr.cells["order"]
        rows = []
        for k in order:
            hk = c_all[hist & (cell == k)]
            lk = c_all[live & (cell == k)]
            top = Counter(hk.teacher).most_common(1)
            rows.append({
                "cell": int(k), "label": top[0][0] if top else "",
                "launch_share": float(lk.is_launch.mean()) if len(lk) else 0.0,
                "oos_share": float((lk.gold == OOS).mean()) if len(lk) else 0.0,
                "n": int(len(lk)),
            })
        snaps = [s for s in fr.snapshots]
        data["heat"] = {
            "rows": rows,
            "t": [s["t"] for s in snaps],
            "intent": [[round(max(0.0, s["local_intent"][k]), 3) for k in order] for s in snaps],
            "scope": [[round(max(0.0, s["local_scope"][k]), 3) for k in order] for s in snaps],
        }
        lc = _live(c_all)
        samp = lc.iloc[::10]
        data["feed"] = [{
            "t": int(r.t), "text": r.text[:110], "d": r.decision, "a": bool(r.audited), "f": r.final,
            "g": r.gold, "ok": bool(r.correct), "route": route(r.final), "urg": urgency(r.final),
            "launch": bool(r.is_launch),
        } for r in samp.itertuples(index=False)]
    ref = _live(next(iter(results.values())).calls)
    data["phases"] = [{"name": ph, "start": int(g.t.min()), "end": int(g.t.max()) + 1}
                      for ph, g in ref.groupby("phase", sort=False)]
    data["launch_intents"] = list(LAUNCH_INTENTS)
    return data


def write_all(results: dict[str, RunResult], meta: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarise(results)
    (out_dir / "REPORT.md").write_text(report_md(summary, meta))
    data = dashboard_data(results, summary, meta)
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=2))
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    (out_dir / "replay.html").write_text(html)
    return summary
