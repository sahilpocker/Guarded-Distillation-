"""Routing benchmark: which score should the risk-controlled gate use? (cached Haiku labels, no API calls)

    python scripts/routing.py grid [--workers 6]        # stage 1: alpha grid, all routers, all families, seeds 0-5
    python scripts/routing.py select                    # alpha per (router, family, budget) from seeds 0-2 -> runs/routing/select.json
    python scripts/routing.py report                    # stage 2 at the selected alpha on seeds 3-5, tables + docs/routing.json + plot

Families: banking (launch of 5 intents), credit_cards and travel (CLINC150 domain launch).
Routers: margin, entropy, maxprob, disagree (margin + previous-version disagreement), learned,
learned_sem (learned + semantic cortisol), oracle (escalate exactly the student's mistakes).
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

FAMILIES = {"banking": "grid-minilm-a03-s", "credit_cards": "clinc-credit_cards-s", "travel": "clinc-travel-s"}
ROUTERS = ["margin", "entropy", "maxprob", "disagree", "learned", "learned_sem", "oracle"]
ALPHAS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3]
BUDGETS = [40, 50, 60]
SEM = {"sem_gain": 0.7, "sem_hot": 0.4, "sem_qfit": 0.5}
OUT = Path("runs/routing")


def path(fam, router, alpha, seed):
    a = "na" if router == "oracle" else f"{alpha:g}"
    return OUT / f"{fam}-{router}-a{a}-s{seed}.json"


def npz_path(fam, router, alpha, seed):
    return path(fam, router, alpha, seed).with_suffix(".npz")


def run_one(fam, router, alpha, seed, need_npz=False):
    p = path(fam, router, alpha, seed)
    if p.exists() and (not need_npz or npz_path(fam, router, alpha, seed).exists()):
        return json.loads(p.read_text())
    import semantic as S
    from guarded_distillation import data as D, embed
    from guarded_distillation.cli import _load_run
    from guarded_distillation.engine import METHODS, EngineConfig, run_method
    from guarded_distillation.teacher import PRICES, CallCost
    run = Path(f"runs/{FAMILIES[fam]}{seed}")
    stream, tdf, meta = _load_run(run)
    pools = D.load_pools("data/raw")
    price = PRICES[meta.get("price", "haiku-5.5")]
    cost = np.array([CallCost(int(r.input_tokens), int(r.output_tokens), int(r.cache_read_tokens), int(r.cache_write_tokens)).usd(price)
                     for r in tdf.itertuples()])
    kw = {"seed": meta["seed"], "map_backend": "tfidf", "student_backend": "sbert", "alpha_intent": alpha if alpha else 0.03, "balance": True}
    method = "retrain"
    if router == "learned_sem":
        kw |= {"router": "learned", **SEM}
        method = "semantic"
    else:
        kw["router"] = router
    cfg = EngineConfig(**kw)
    embed.encode(stream.text.tolist())
    intents = tuple(meta.get("intents") or pools.intents)
    ref = run_method(stream, tdf, intents, METHODS["llm_only"], cfg, cost)
    res = run_method(stream, tdf, intents, METHODS[method], cfg, cost)
    m = S.metrics(res, ref)
    c = res.calls[res.calls.phase != "history"]
    m["leak_rate"] = float(((c.decision == "accept") & (c.final != c.teacher)).mean())
    m["leak_budget"] = cfg.alpha_scope + cfg.alpha_intent
    m["launch600"] = m["launch600"]
    out = {"family": fam, "router": router, "alpha": alpha, "seed": seed, **m}
    OUT.mkdir(parents=True, exist_ok=True)
    cc = res.calls[res.calls.phase != "history"].reset_index(drop=True)
    rr = ref.calls[ref.calls.phase != "history"].reset_index(drop=True)
    first = np.zeros(len(cc), bool)
    first[np.flatnonzero((cc.phase == "launch").to_numpy())[:600]] = True
    np.savez_compressed(npz_path(fam, router, alpha, seed), correct=cc.correct.to_numpy(), llm_correct=rr.correct.to_numpy(),
                        is_launch=cc.is_launch.to_numpy(), first=first, llm_call=cc.llm_call.to_numpy())
    p.write_text(json.dumps(out))
    return out


def load(fam, router, alpha, seeds):
    rows = [json.loads(path(fam, router, alpha, s).read_text()) for s in seeds if path(fam, router, alpha, s).exists()]
    return rows if len(rows) == len(seeds) else None


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def interp_alpha(curve, target):
    """curve: [(alpha, calls)] sorted by alpha (calls falls as alpha grows); log-linear interpolation of alpha at `target` calls."""
    pts = sorted(curve)
    for (a0, c0), (a1, c1) in zip(pts, pts[1:]):
        if (c0 - target) * (c1 - target) <= 0 and c0 != c1:
            t = (c0 - target) / (c0 - c1)
            return float(np.exp(np.log(a0) + t * (np.log(a1) - np.log(a0))))
    return None


def cmd_grid(a):
    jobs = [(f, r, al, s) for f in FAMILIES for r in ROUTERS for al in ([None] if r == "oracle" else ALPHAS) for s in range(6)]
    todo = [j for j in jobs if not path(*j).exists()]
    print(f"{len(jobs)} runs, {len(todo)} to do", flush=True)
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_one, *j): j for j in todo}
        for f in as_completed(futs):
            done += 1
            try:
                f.result()
            except Exception as e:
                print("FAILED", futs[f], repr(e)[:200], flush=True)
            if done % 20 == 0:
                print(f"{done}/{len(todo)}", flush=True)


def cmd_select(a):
    sel = {}
    for fam in FAMILIES:
        for r in ROUTERS:
            if r == "oracle":
                continue
            curve = []
            for al in ALPHAS:
                rows = load(fam, r, al, [0, 1, 2])
                if rows:
                    curve.append((al, mean(rows, "calls")))
            for b in BUDGETS:
                sel[f"{fam}|{r}|{b}"] = interp_alpha(curve, b)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "select.json").write_text(json.dumps(sel, indent=1))
    miss = [k for k, v in sel.items() if v is None]
    print(f"selected {len(sel) - len(miss)} alphas; unreachable budgets: {miss}")


def cmd_stage2(a):
    sel = json.loads((OUT / "select.json").read_text())
    jobs = [(f, r, al, s) for k, al in sel.items() if al for f, r, _ in [k.split("|")] for s in (3, 4, 5)]
    jobs = [(f, r, round(al, 5), s) for f, r, al, s in jobs]
    todo = [j for j in jobs if not path(*j).exists()]
    print(f"stage 2: {len(jobs)} runs, {len(todo)} to do", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(run_one, *j) for j in todo])):
            f.result()
            if i % 20 == 0:
                print(i, flush=True)


COLORS = {"margin": "#455765", "entropy": "#9a6fd1", "maxprob": "#c9a227", "disagree": "#d9822b", "learned": "#0b7a75",
          "learned_sem": "#1f5fbf", "oracle": "#c23b31"}
LABEL = {"margin": "Margin (current)", "entropy": "Entropy", "maxprob": "Max probability", "disagree": "Version disagreement + margin",
         "learned": "Learned router", "learned_sem": "Learned router + semantic cortisol", "oracle": "Oracle"}
FAM_TITLE = {"banking": "Banking77 launch (5 new intents)", "credit_cards": "CLINC150 domain launch: credit_cards",
             "travel": "CLINC150 domain launch: travel"}


def collect():
    """Everything the report needs, from the run files."""
    out = {"curves": {}, "budget": {}, "oracle": {}, "llm_only": {}}
    for fam in FAMILIES:
        base = [json.loads(path(fam, "margin", 0.03, s).read_text()) for s in (3, 4, 5)]
        out["llm_only"][fam] = {"accuracy": mean([{"v": r["acc"] - r["acc_delta"]} for r in base], "v"),
                                "launch600": mean([{"v": r["launch600"] - r["launch600_delta"]} for r in base], "v")}
        o = load(fam, "oracle", None, [3, 4, 5])
        out["oracle"][fam] = {k: mean(o, k) for k in ("calls", "acc", "launch600", "leak_rate")} if o else None
        for r in ROUTERS:
            if r == "oracle":
                continue
            pts = []
            for al in ALPHAS:
                rows = load(fam, r, al, [3, 4, 5])
                if rows:
                    pts.append({"alpha": al, "calls": mean(rows, "calls"), "acc": mean(rows, "acc"), "launch600": mean(rows, "launch600"),
                                "leak": mean(rows, "leak_rate")})
            out["curves"][f"{fam}|{r}"] = sorted(pts, key=lambda p: p["calls"])
    sel = json.loads((OUT / "select.json").read_text()) if (OUT / "select.json").exists() else {}
    for k, al in sel.items():
        fam, r, b = k.split("|")
        rows = load(fam, r, round(al, 5), [3, 4, 5]) if al else None
        if rows:
            out["budget"][k] = {"alpha": al, "calls": mean(rows, "calls"), "acc": mean(rows, "acc"), "acc_min": min(x["acc"] for x in rows),
                                "acc_max": max(x["acc"] for x in rows), "launch600": mean(rows, "launch600"),
                                "launch600_delta": mean(rows, "launch600_delta"), "leak": mean(rows, "leak_rate"), "budget": mean(rows, "leak_budget"),
                                "held": sum(x["leak_rate"] <= x["leak_budget"] for x in rows), "extra": mean(rows, "extra_calls_pct")}
    return out


def at_budget(fam, router, budget, key, seeds=(3, 4, 5)):
    """Per seed, interpolate `key` linearly along that seed's own calls-vs-metric curve at `budget`% calls;
    returns (mean, min, max) over seeds, or None when the curve does not reach the budget on some seed."""
    vals = []
    for sd in seeds:
        pts = sorted((r["calls"], r[key]) for al in ALPHAS for r in [json.loads(path(fam, router, al, sd).read_text())])
        v = None
        for (c0, y0), (c1, y1) in zip(pts, pts[1:]):
            if c0 <= budget <= c1 and c1 > c0:
                v = y0 + (y1 - y0) * (budget - c0) / (c1 - c0)
                break
        if v is None:
            return None
        vals.append(v)
    return float(np.mean(vals)), float(min(vals)), float(max(vals))


def ocl_points(fam):
    """Online Cascade Learning runs (scripts/ocl_adapter.py) for a family: {seed: [(calls, acc, launch600)]} sorted by calls."""
    out = {}
    for sd in (3, 4, 5):
        rows = [json.loads(f.read_text()) for f in Path("runs/ocl").glob(f"{fam}-s{sd}-mu*.json")]
        out[sd] = sorted((r["calls"], r["acc"], r["launch600"]) for r in rows)
    return out


def ocl_at_budget(fam, budget, idx, near=5.0, max_gap=25.0):
    """OCL's calls-vs-metric curve has a cliff (a tiny change in its cost coefficient flips it from deferring almost
    everything to deferring almost nothing), so a budget is only reported where a measured point lies within `near`
    calls or two neighbouring points are at most `max_gap` apart; otherwise None ("not reached")."""
    vals = []
    for sd, pts in ocl_points(fam).items():
        v = None
        close = [p for p in pts if abs(p[0] - budget) <= near]
        if close:
            p = min(close, key=lambda q: abs(q[0] - budget))
            v = p[idx]
        else:
            for p0, p1 in zip(pts, pts[1:]):
                if p0[0] <= budget <= p1[0] and 0 < p1[0] - p0[0] <= max_gap:
                    v = p0[idx] + (p1[idx] - p0[idx]) * (budget - p0[0]) / (p1[0] - p0[0])
                    break
        if v is None:
            return None
        vals.append(v)
    return float(np.mean(vals)), float(min(vals)), float(max(vals))


def svg_curves(d, W=1180, H=380):
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H + 84}" role="img" font-family="Familjen Grotesk, Helvetica, Arial, sans-serif" '
         f'aria-label="LLM calls against accuracy for each routing score, three streams">']
    pw, L, T, B = 370, 52, 34, 46
    for pi, fam in enumerate(FAMILIES):
        x0 = pi * (pw + 24) + L
        pts_all = [p for r in ROUTERS if r != "oracle" for p in d["curves"][f"{fam}|{r}"]]
        xs = [p["calls"] for p in pts_all] + [d["oracle"][fam]["calls"]]
        ys = [p["acc"] for p in pts_all] + [d["llm_only"][fam]["accuracy"], d["oracle"][fam]["acc"]]
        ax0, ax1 = max(10, np.floor(min(xs) / 10) * 10), min(100, np.ceil(max(xs) / 10) * 10)
        ay0, ay1 = np.floor(min(ys) * 2) / 2 - 0.5, np.ceil(max(ys) * 2) / 2 + 0.5
        X = lambda v: x0 + (v - ax0) / (ax1 - ax0) * (pw - L)              # noqa: E731
        Y = lambda v: T + (1 - (v - ay0) / (ay1 - ay0)) * (H - T - B)      # noqa: E731
        o.append(f'<text x="{x0 - L + 4}" y="20" font-size="13" font-weight="600" fill="#13202b">{FAM_TITLE[fam]}</text>')
        for v in np.arange(ax0, ax1 + 0.1, 10):
            o.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{T}" y2="{H - B}" stroke="#d3dce2"/><text x="{X(v):.1f}" y="{H - B + 15}" text-anchor="middle" font-size="11" fill="#455765">{v:.0f}%</text>')
        for v in np.arange(np.ceil(ay0), ay1 + 0.01, 1.0):
            o.append(f'<line x1="{x0}" x2="{x0 + pw - L}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#d3dce2"/><text x="{x0 - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" font-size="11" fill="#455765">{v:.0f}</text>')
        o.append(f'<text x="{(x0 + x0 + pw - L) / 2:.0f}" y="{H - 10}" text-anchor="middle" font-size="12" fill="#455765">LLM calls (share of queries)</text>')
        if pi == 0:
            o.append(f'<text transform="translate(12 {(T + H - B) / 2:.0f}) rotate(-90)" text-anchor="middle" font-size="12" fill="#455765">accuracy vs human labels (%)</text>')
        pts_o = ocl_points(fam)
        if all(pts_o.values()):
            mus = {}
            for sd, pts in pts_o.items():
                for c, a_, _ in pts:
                    mus.setdefault(round(c / 3), []).append((c, a_))
            for v in mus.values():
                c, a_ = np.mean([x[0] for x in v]), np.mean([x[1] for x in v])
                if ax0 <= c <= ax1 and a_ >= ay0:
                    o.append(f'<circle cx="{X(c):.1f}" cy="{Y(a_):.1f}" r="3.4" fill="none" stroke="#7b8791" stroke-width="1.5"><title>Online Cascade Learning: {c:.0f}% calls, {a_:.2f}%</title></circle>')
        la = d["llm_only"][fam]["accuracy"]
        o.append(f'<line x1="{x0}" x2="{x0 + pw - L}" y1="{Y(la):.1f}" y2="{Y(la):.1f}" stroke="#13202b" stroke-dasharray="4 4"/>'
                 f'<text x="{x0 + 4}" y="{Y(la) - 5:.1f}" text-anchor="start" font-size="10" fill="#13202b">LLM on every query</text>')
        for r in ROUTERS:
            if r == "oracle":
                ov = d["oracle"][fam]
                o.append(f'<path d="M{X(ov["calls"]):.1f} {Y(ov["acc"]) - 7:.1f} l7 12 h-14 Z" fill="{COLORS[r]}"><title>Oracle: {ov["calls"]:.0f}% calls, {ov["acc"]:.2f}%</title></path>')
                continue
            pts = d["curves"][f"{fam}|{r}"]
            dd = "M" + " L".join(f"{X(p['calls']):.1f} {Y(p['acc']):.1f}" for p in pts)
            o.append(f'<path d="{dd}" fill="none" stroke="{COLORS[r]}" stroke-width="2"/>')
            for p in pts:
                o.append(f'<circle cx="{X(p["calls"]):.1f}" cy="{Y(p["acc"]):.1f}" r="3.2" fill="{COLORS[r]}"><title>{LABEL[r]} (alpha {p["alpha"]:g}): {p["calls"]:.0f}% calls, {p["acc"]:.2f}%</title></circle>')
    items = [(COLORS[r], LABEL[r], "box") for r in ROUTERS] + [("#7b8791", "Online Cascade Learning (their code; points, no line: it has a cliff)", "dash")]
    lx, ly = 6, H + 30
    for col, text, kind in items:
        w = 28 + 7.0 * len(text) + 20
        if lx + w > W:
            lx, ly = 6, ly + 22
        if kind == "box":
            o.append(f'<rect x="{lx}" y="{ly}" width="12" height="12" rx="2" fill="{col}"/>')
        else:
            o.append(f'<circle cx="{lx + 6}" cy="{ly + 6}" r="3.6" fill="none" stroke="{col}" stroke-width="1.5"/>')
        o.append(f'<text x="{lx + 19}" y="{ly + 10}" font-size="12" fill="#13202b">{text}</text>')
        lx += w
    o.append("</svg>")
    return "\n".join(o)


def cmd_report(a):
    d = collect()
    Path("docs").mkdir(exist_ok=True)
    Path("docs/routing.json").write_text(json.dumps(d, indent=1))
    Path("docs/routing.svg").write_text(svg_curves(d), encoding="utf-8")
    lines = ["## Accuracy at matched LLM-call budgets (interpolated along each seed's own curve; mean [min-max] over seeds 3-5)\n"]
    for key, title in (("acc", "Overall accuracy vs human labels (%)"), ("launch600", "Launch-question accuracy in the first 600 launch-phase queries (%)")):
        for fam in FAMILIES:
            lo = d["llm_only"][fam]
            ref = lo["accuracy"] if key == "acc" else lo["launch600"]
            lines += [f"\n**{title}: {FAM_TITLE[fam]}** (LLM on every query: {ref:.2f}%)\n", "| Router | 40% calls | 50% calls | 60% calls |", "|---|---|---|---|"]
            if any(ocl_points(fam).values()):
                cells = []
                for b in BUDGETS:
                    v = ocl_at_budget(fam, b, 1 if key == "acc" else 2)
                    cells.append("not reached" if v is None else f"{v[0]:.2f} [{v[1]:.2f}-{v[2]:.2f}]")
                ocl_row = "| Online Cascade Learning (LR + LLM, their code) | " + " | ".join(cells) + " |"
            for r in ROUTERS:
                if r == "oracle":
                    o = d["oracle"][fam]
                    lines.append(f"| Oracle ({o['calls']:.0f}% calls) | {o[key]:.2f} | {o[key]:.2f} | {o[key]:.2f} |")
                    continue
                cells = []
                for b in BUDGETS:
                    v = at_budget(fam, r, b, key)
                    cells.append("not reached" if v is None else f"{v[0]:.2f} [{v[1]:.2f}-{v[2]:.2f}]")
                lines.append(f"| {LABEL[r]} | " + " | ".join(cells) + " |")
            if any(ocl_points(fam).values()):
                lines.append(ocl_row)
    lines.append("\n## Online Cascade Learning, measured points (their code, LR + LLM cascade; mean over seeds 3-5)")
    lines += ["\n| Stream | mu | LLM calls | Accuracy (LLM alone) | Launch accuracy, first 600 (LLM alone) |", "|---|---|---|---|---|"]
    import collections
    for fam in FAMILIES:
        byc = collections.defaultdict(list)
        for f in Path("runs/ocl").glob(f"{fam}-s[345]-mu*.json"):
            r = json.loads(f.read_text())
            byc[r["mu"]].append(r)
        for mu in sorted(byc):
            v = byc[mu]
            m = lambda k: float(np.mean([x[k] for x in v]))      # noqa: E731
            lines.append(f"| {FAM_TITLE[fam]} | {mu:g} | {m('calls'):.1f}% | {m('acc'):.2f}% ({m('llm_acc'):.2f}%) | {m('launch600'):.1f}% ({m('llm_launch600'):.1f}%) |")
    lines.append("\n## Gate settings chosen on seeds 0-2 and run on seeds 3-5 (calls land off the nominal budget)")
    for fam in FAMILIES:
        lo = d["llm_only"][fam]
        lines += [f"\n### {FAM_TITLE[fam]} (seeds 3-5; LLM on every query: {lo['accuracy']:.2f}% overall, {lo['launch600']:.1f}% launch questions in the first 600 launch-phase queries)\n",
                  "| Budget | Router | LLM calls | Accuracy | Launch accuracy, first 600 | Leak rate (budget) | Leak budget held |", "|---|---|---|---|---|---|---|"]
        for b in BUDGETS:
            for r in ROUTERS:
                if r == "oracle":
                    continue
                v = d["budget"].get(f"{fam}|{r}|{b}")
                if not v:
                    lines.append(f"| {b}% | {LABEL[r]} | not reachable | | | | |")
                    continue
                lines.append(f"| {b}% | {LABEL[r]} | {v['calls']:.1f}% | {v['acc']:.2f}% [{v['acc_min']:.2f}-{v['acc_max']:.2f}] | {v['launch600']:.1f}% | "
                             f"{v['leak'] * 100:.1f}% ({v['budget'] * 100:.1f}%) | {v['held']}/3 seeds |")
        o = d["oracle"][fam]
        lines.append(f"| any | Oracle | {o['calls']:.1f}% | {o['acc']:.2f}% | {o['launch600']:.1f}% | {o['leak_rate'] * 100:.1f}% | n/a |")
    Path("runs/routing/report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["grid", "select", "stage2", "report"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    {"grid": cmd_grid, "select": cmd_select, "stage2": cmd_stage2, "report": cmd_report}[a.cmd](a)
