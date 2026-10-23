"""gdistill bundle | serve | traffic."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd


def cmd_bundle(args) -> None:
    """Train students on the shadow-mode history of a seed's stream (teacher labels from the cache)."""
    from .. import data as D
    from ..teacher import LLMTeacher, build_system_prompt
    from .bundle import build_bundle
    pools = D.load_pools(args.raw)
    fewshot = D.load_fewshot(pools, 1)
    stream, _ = D.build_stream(pools, D.StreamConfig(seed=args.seed), fewshot)
    hist = stream[stream.phase == "history"]
    t = LLMTeacher("anthropic", "claude-haiku-5-5", build_system_prompt(pools.intents, fewshot), list(pools.intents) + [D.OOS],
                   cache_path="labels/claude-haiku-5-5.jsonl", readonly=True)
    recs = t.label_all(hist.text.tolist())
    labels = [r["label"] for r in recs]
    version = args.version or Path(args.out).name
    b = build_bundle(hist.text.tolist(), labels, list(pools.intents), version=version, seed=args.seed,
                     meta={"source": f"history of seed {args.seed} stream, Haiku labels", "n_history": len(hist)})
    b.save(args.out)
    print(f"{version}: {len(hist)} history queries, intent holdout acc {b.intent.holdout_acc:.1%}, "
          f"starting levels scope {b.q0_scope:.2f} intent {b.q0_intent:.2f} -> {args.out}")


def cmd_serve(args) -> None:
    import uvicorn
    from .app import make_app
    app = make_app(args.bundle)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_traffic(args) -> None:
    import httpx
    if args.stream:
        stream = pd.read_csv(Path(args.stream) / "stream.csv", keep_default_na=False)
    else:                                   # no run folder (CI): build the seed's stream from the datasets
        from .. import data as D
        pools = D.load_pools(args.raw)
        stream, _ = D.build_stream(pools, D.StreamConfig(seed=args.seed), D.load_fewshot(pools, 1))
    stream = stream[stream.phase != "history"]
    if getattr(args, "start", 0):
        stream = stream.iloc[args.start:]
    if args.limit:
        stream = stream.head(args.limit)
    gap = 1.0 / args.rate
    n = esc = 0
    t_start = time.time()
    with httpx.Client(timeout=120) as c:
        for r in stream.itertuples():
            t0 = time.time()
            try:
                out = c.post(args.url.rstrip("/") + "/triage", json={"text": r.text}).json()
            except Exception as e:                      # keep going: a sleeping free host wakes slowly
                print(f"request failed: {e}")
                time.sleep(2)
                continue
            n += 1
            esc += out.get("decision") == "escalate"
            if n % 50 == 0:
                print(f"{n} sent, {esc / n:.0%} escalated, {n / (time.time() - t_start):.1f} req/s")
            time.sleep(max(0.0, gap - (time.time() - t0)))
    m = httpx.get(args.url.rstrip("/") + "/metrics").json()
    print(json.dumps({k: m[k] for k in ("model_version", "decisions_total", "escalation_rate", "estimated_leak_rate", "leak_budget", "spend")}, indent=1))


def cmd_retrain(args) -> None:
    """Build a candidate bundle from the history plus the labels the service collected, gate it, and publish it."""
    import hashlib
    import subprocess
    import tarfile
    import httpx
    from .. import data as D
    from ..teacher import LLMTeacher, build_system_prompt
    from .bundle import Bundle, build_bundle
    from .evalbundle import eval_stream, evaluate, gate
    from .rollout import Registry, version_number
    pools = D.load_pools(args.raw)
    fewshot = D.load_fewshot(pools, 1)
    stream, _ = D.build_stream(pools, D.StreamConfig(seed=args.seed), fewshot)
    hist = stream[stream.phase == "history"]
    t = LLMTeacher("anthropic", "claude-haiku-5-5", build_system_prompt(pools.intents, fewshot), list(pools.intents) + [D.OOS],
                   cache_path="labels/claude-haiku-5-5.jsonl", readonly=True)
    texts = hist.text.tolist()
    labels = [r["label"] for r in t.label_all(texts)]
    seen = set(texts)
    extra = []
    if args.labels_file:
        extra = json.loads(Path(args.labels_file).read_text())["labels"]
    elif args.export_url and os.environ.get("ADMIN_TOKEN"):
        r = httpx.get(args.export_url.rstrip("/") + "/admin/export", headers={"x-admin-token": os.environ["ADMIN_TOKEN"]}, timeout=120)
        r.raise_for_status()
        extra = r.json()["labels"]
    ok = set(pools.intents) | {D.OOS}
    new = [(e["text"], e["label"]) for e in extra if e["text"] not in seen and e["label"] in ok]
    for x, y in dict(new).items():
        texts.append(x)
        labels.append(y)
    reg = Registry(args.repo) if args.repo else None
    rels = reg.releases() if reg else []
    current = reg.fetch(rels[0]) if rels else Bundle.load(args.current)
    n_next = max([version_number(current.version)] + [r["n"] for r in rels]) + 1
    cand = build_bundle(texts, labels, list(pools.intents), version=f"v{n_next}", seed=args.seed,
                        meta={"source": f"history + {len(new)} labels collected by the service", "n_service_labels": len(new)})
    live = eval_stream(args.eval_seed)
    trained_on = set(texts)                                   # streams of different seeds share one question pool:
    n_all = len(live)                                         # judge both models only on questions the candidate never saw
    live = live[~live.text.isin(trained_on)].reset_index(drop=True)
    cur_m, cand_m = evaluate(current, live), evaluate(cand, live)
    n_launch = int(live.is_launch.sum())
    rows = gate(cand_m, cur_m)
    passed = all(ok_ for *_, ok_ in rows)
    md = [f"## Retrain: candidate {cand.version} against {current.version} (eval stream seed {args.eval_seed}, cached Haiku labels)", "",
          f"Trained on {len(texts) - len(new)} history labels + {len(new)} labels collected by the service. Judged on the {len(live)} of {n_all} eval queries "
          f"the candidate never saw ({n_launch} new-product questions).", "",
          "| Check | Candidate | Limit | Result |", "|---|---|---|---|"]
    md += [f"| {n} | {v} | {lim} | {'pass' if ok_ else '**FAIL**'} |" for n, v, lim, ok_ in rows]
    md += ["", f"{current.version}: overall {cur_m['overall_delta']:+.2f} pts, launch {cur_m['launch_delta']:+.2f} pts, calls {cur_m['calls_pct']:.1f}%.",
           f"{cand.version}: overall {cand_m['overall_delta']:+.2f} pts, launch {cand_m['launch_delta']:+.2f} pts, calls {cand_m['calls_pct']:.1f}%.",
           "", "Promoted." if passed and args.publish else ("Gate passed; not published (dry run)." if passed else "Not promoted: the gate failed.")]
    text = "\n".join(md)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(text + "\n")
    out = Path(args.out) / cand.version
    cand.save(out)
    if not (passed and args.publish):
        return
    tar = Path(args.out) / f"bundle-{cand.version}.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        tf.add(out, arcname=cand.version)
    sha = hashlib.sha256(tar.read_bytes()).hexdigest()
    notes = f"sha256: {sha}\n\n```json\n{json.dumps({'candidate': cand_m, 'previous': cur_m, 'n_service_labels': len(new)}, indent=1)}\n```\n"
    notes_file = Path(args.out) / "notes.md"
    notes_file.write_text(notes)
    subprocess.run(["gh", "release", "create", f"model-{cand.version}", str(tar), "--title", f"Model {cand.version}", "--notes-file", str(notes_file),
                    *( ["--repo", args.repo] if args.repo else [])], check=True)
    print(f"published model-{cand.version}")
