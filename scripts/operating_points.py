"""Operating-point grid: collect, select the default, find the trade-off curve, draw it.

    python scripts/operating_points.py                      # seeds 0-2: table, docs/grid.json, docs/operating-points.svg
    python scripts/operating_points.py --confirm global tfidf 0.05 --seeds 3 4 5

Settings are (method, student, alpha_intent) with the TF-IDF map. Each cell is read from the run's
results.json (see scripts/grid.py for where runs live). Accuracy is "vs LLM only" in points; LLM calls
is the share of live queries that reached the LLM (escalations plus audits).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from grid import run_dir  # noqa: E402

METHODS = ("retrain", "global", "cortisol")
TITLES = {"retrain": "Static gate + retraining", "global": "Global cortisol + retraining", "cortisol": "Cortisol field"}
STUDENTS = ("tfidf", "minilm")
ALPHAS = (0.03, 0.05, 0.08)
BAR = -0.3          # accuracy vs LLM only must be at least this many points, on every selection seed
VARS = {"retrain": ("--s-retrain", "#eda100"), "global": ("--s-global", "#2a78d6"), "cortisol": ("--s-cortisol", "#eb6834")}


def cell(method, student, alpha, seed):
    p = Path(run_dir(student, alpha, seed)) / "results.json"
    s = json.loads(p.read_text())["summary"][method]
    return {"calls": s["llm_share"] * 100, "delta": s["accuracy_delta_pts"], "errors": s["cheap_errors_per_1k"]}


def collect(seeds, methods=METHODS, students=STUDENTS, alphas=ALPHAS):
    out = []
    for m in methods:
        for st in students:
            for a in alphas:
                per = {k: cell(m, st, a, k) for k in seeds}
                row = {"method": m, "student": st, "alpha": a, "seeds": per}
                for key in ("calls", "delta", "errors"):
                    v = [per[k][key] for k in seeds]
                    row[key] = {"mean": float(np.mean(v)), "min": float(min(v)), "max": float(max(v))}
                row["passes"] = all(per[k]["delta"] >= BAR for k in seeds)
                out.append(row)
    return out


def pareto(rows):
    """Settings no other setting beats on both fewer LLM calls and higher accuracy (means)."""
    for p in rows:
        p["frontier"] = not any(
            q is not p and q["calls"]["mean"] <= p["calls"]["mean"] and q["delta"]["mean"] >= p["delta"]["mean"]
            and (q["calls"]["mean"] < p["calls"]["mean"] or q["delta"]["mean"] > p["delta"]["mean"]) for q in rows)


def select(rows, methods=None):
    ok = [r for r in rows if r["passes"] and (methods is None or r["method"] in methods)]
    return min(ok, key=lambda r: (r["calls"]["mean"], r["errors"]["mean"])) if ok else None


def label(r):
    return f"{TITLES[r['method']]} / {'TF-IDF' if r['student'] == 'tfidf' else 'MiniLM'} / α {r['alpha']:g}"


def table(rows, default=None):
    f = lambda d, p=1, u="": f"{d['mean']:.{p}f}{u} [{d['min']:.{p}f}–{d['max']:.{p}f}]"   # noqa: E731
    lines = ["| Method | Student | α | LLM calls, mean [range] | Δ vs LLM only, pts, mean [range] | Cheap-path errors / 1k, mean [range] | Δ ≥ −0.3 on every seed | On curve |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["calls"]["mean"]):
        mark = " **← default**" if default is r else ""
        lines.append(f"| {TITLES[r['method']]} | {'TF-IDF' if r['student'] == 'tfidf' else 'MiniLM'} | {r['alpha']:g} | "
                     f"{f(r['calls'], 0, '%')} | {f(r['delta'], 2)} | {f(r['errors'])} | {'yes' if r['passes'] else 'no'} | "
                     f"{'yes' if r['frontier'] else 'no'}{mark} |")
    return "\n".join(lines)


def svg(rows, default=None, W=760, H=470):
    L, R, T, B = 62, 20, 22, 56
    xs = [r["calls"][k] for r in rows for k in ("min", "max")]
    ys = [r["delta"][k] for r in rows for k in ("min", "max")] + [BAR]
    x0, x1 = np.floor(min(xs) / 5) * 5 - 2, np.ceil(max(xs) / 5) * 5 + 2
    y0, y1 = np.floor(min(ys) * 2) / 2 - 0.25, np.ceil(max(ys) * 2) / 2 + 0.25
    X = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)          # noqa: E731
    Y = lambda v: T + (1 - (v - y0) / (y1 - y0)) * (H - T - B)    # noqa: E731
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" font-family="IBM Plex Mono, ui-monospace, Menlo, monospace" '
         f'aria-label="Operating points: LLM calls against accuracy versus LLM only, one point per setting, error bars over seeds">']
    ink3, line = "var(--ink-3, #6f797e)", "var(--line, #dde3df)"
    for v in np.arange(np.ceil(x0 / 10) * 10, x1 + 0.1, 10):
        o.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{T}" y2="{H - B}" stroke="{line}"/>'
                 f'<text x="{X(v):.1f}" y="{H - B + 16}" text-anchor="middle" font-size="11" fill="{ink3}">{v:.0f}%</text>')
    for v in np.arange(np.ceil(y0 * 2) / 2, y1 + 0.01, 0.5):
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{line}"/>'
                 f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" font-size="11" fill="{ink3}">{v:+.1f}</text>')
    o.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(BAR):.1f}" y2="{Y(BAR):.1f}" stroke="var(--bad, #d03b3b)" stroke-dasharray="5 4"/>'
             f'<text x="{W - R - 4}" y="{Y(BAR) - 5:.1f}" text-anchor="end" font-size="11" fill="var(--bad-ink, #a82a2a)">accuracy bar: −0.3 pts</text>')
    o.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 24}" text-anchor="middle" font-size="12" fill="var(--ink-2, #4d575c)">'
             f'← fewer LLM calls (share of queries)</text>')
    o.append(f'<text transform="translate(16 {(T + H - B) / 2:.0f}) rotate(-90)" text-anchor="middle" font-size="12" fill="var(--ink-2, #4d575c)">'
             f'accuracy vs LLM only (points; higher is better)</text>')
    front = sorted((r for r in rows if r["frontier"]), key=lambda r: r["calls"]["mean"])
    if len(front) > 1:
        d = "M" + " L".join(f"{X(r['calls']['mean']):.1f} {Y(r['delta']['mean']):.1f}" for r in front)
        o.append(f'<path d="{d}" fill="none" stroke="var(--ink-2, #4d575c)" stroke-width="1.2" stroke-dasharray="2 3"/>')
    for r in rows:
        var, fb = VARS[r["method"]]
        col = f"var({var}, {fb})"
        cx, cy = X(r["calls"]["mean"]), Y(r["delta"]["mean"])
        tip = (f"{label(r)}: {r['calls']['mean']:.0f}% LLM calls (range {r['calls']['min']:.0f}–{r['calls']['max']:.0f}), "
               f"{r['delta']['mean']:+.2f} pts vs LLM only (range {r['delta']['min']:+.2f} to {r['delta']['max']:+.2f}), "
               f"{r['errors']['mean']:.0f} cheap-path errors/1k" + (" · DEFAULT" if r is default else "") + (" · on the curve" if r["frontier"] else ""))
        o.append(f'<g><title>{tip}</title>'
                 f'<line x1="{X(r["calls"]["min"]):.1f}" x2="{X(r["calls"]["max"]):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" stroke="{col}" stroke-width="1.4" opacity=".55"/>'
                 f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{Y(r["delta"]["min"]):.1f}" y2="{Y(r["delta"]["max"]):.1f}" stroke="{col}" stroke-width="1.4" opacity=".55"/>')
        if r["frontier"] and r is not default:
            o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="11" fill="none" stroke="var(--ink-2, #4d575c)" stroke-width="1.4" stroke-dasharray="2 3"/>')
        if r is default:
            o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="12" fill="none" stroke="var(--ink, #14191b)" stroke-width="2"/>')
        if r["student"] == "tfidf":
            o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="{col}" stroke="var(--surface, #fff)" stroke-width="1.5"/>')
        else:
            o.append(f'<path d="M{cx:.1f} {cy - 7:.1f} L{cx + 7:.1f} {cy:.1f} L{cx:.1f} {cy + 7:.1f} L{cx - 7:.1f} {cy:.1f} Z" fill="{col}" stroke="var(--surface, #fff)" stroke-width="1.5"/>')
        o.append(f'<text x="{cx + 9:.1f}" y="{cy - 8:.1f}" font-size="10" fill="{ink3}">{r["alpha"]:g}</text></g>')
    if default is not None:
        o.append(f'<text x="{X(default["calls"]["mean"]):.1f}" y="{Y(default["delta"]["mean"]) + 28:.1f}" text-anchor="middle" font-size="11" '
                 f'font-weight="600" fill="var(--ink, #14191b)">default</text>')
    lx = L + 8
    for i, (m, t) in enumerate(TITLES.items()):
        var, fb = VARS[m]
        o.append(f'<circle cx="{lx + 6}" cy="{T + 8 + i * 17}" r="5" fill="var({var}, {fb})"/>'
                 f'<text x="{lx + 17}" y="{T + 12 + i * 17}" font-size="11" fill="var(--ink-2, #4d575c)">{t}</text>')
    o.append(f'<circle cx="{lx + 6}" cy="{T + 8 + 3 * 17}" r="5" fill="{ink3}"/><text x="{lx + 17}" y="{T + 12 + 3 * 17}" font-size="11" fill="var(--ink-2, #4d575c)">circle: TF-IDF student</text>')
    o.append(f'<path d="M{lx + 6} {T + 1 + 4 * 17} l6 7 l-6 7 l-6 -7 Z" fill="{ink3}"/><text x="{lx + 17}" y="{T + 12 + 4 * 17}" font-size="11" fill="var(--ink-2, #4d575c)">diamond: MiniLM student</text>')
    o.append(f'<text x="{lx}" y="{T + 12 + 5 * 17}" font-size="10" fill="{ink3}">labels: alpha_intent · bars: min–max over seeds 0–2 · dotted ring: on the trade-off curve</text>')
    o.append("</svg>")
    return "\n".join(o)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--confirm", nargs=3, metavar=("METHOD", "STUDENT", "ALPHA"))
    ap.add_argument("--out", default="docs")
    a = ap.parse_args()
    if a.confirm:
        m, st, al = a.confirm
        r = collect(a.seeds, (m,), (st,), (float(al),))[0]
        print(f"{label(r)} on seeds {a.seeds} (accuracy vs LLM only, LLM calls, cheap-path errors / 1k):")
        for k in a.seeds:
            c = r["seeds"][k]
            print(f"  seed {k}: {c['delta']:+.2f} pts, {c['calls']:.1f}% calls, {c['errors']:.1f} errors/1k")
        for key, name in (("delta", "pts vs LLM only"), ("calls", "% LLM calls"), ("errors", "cheap-path errors/1k")):
            d = r[key]
            print(f"  {name}: mean {d['mean']:.2f}, range {d['min']:.2f} to {d['max']:.2f}")
        return
    rows = collect(a.seeds)
    pareto(rows)
    best = select(rows)
    field_on_curve = any(r["frontier"] for r in rows if r["method"] == "cortisol")
    default = best if field_on_curve else select(rows, {"global"})
    print(table(rows, default))
    print(f"\nBest by the rule over all methods: {label(best) if best else 'none'}")
    print(f"Cortisol field on the trade-off curve: {field_on_curve}")
    print(f"Default: {label(default) if default else 'none'}")
    out = Path(a.out)
    (out / "grid.json").write_text(json.dumps({"seeds": a.seeds, "bar_pts": BAR, "default": None if default is None else
                                               {k: default[k] for k in ("method", "student", "alpha")},
                                               "field_on_curve": field_on_curve,
                                               "best_overall": None if best is None else {
                                                   **{k: best[k] for k in ("method", "student", "alpha")},
                                                   "calls": best["calls"]["mean"], "errors": best["errors"]["mean"]},
                                               "settings": rows}, indent=1))
    (out / "operating-points.svg").write_text(svg(rows, default))
    print(f"wrote {out / 'grid.json'} and {out / 'operating-points.svg'}")


if __name__ == "__main__":
    main()
