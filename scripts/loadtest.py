"""Load test of the triage service (closed loop, several concurrency levels), run against a local server on this machine.

    python scripts/loadtest.py [--n 1500] [--levels 1 4 16 32] [--url http://host:port]   # no --url: start a local server

The server it starts is the production app (ONNX Runtime embeddings, models/v1, SQLite log in a temp dir) with live Claude calls
off and per-visitor limits lifted, so escalations are answered from the cached Haiku labels and nothing is spent. Requests are
questions from the seed-3 live stream (each wording once; the on-disk embedding cache the replays use is switched off, so every request runs the ONNX encoder).
Writes docs/loadtest.json. A free Render instance has 0.1 CPU, so its absolute numbers are far lower than a laptop's; the
results here are what one process on this machine does, and the Dockerfile's size (about 300 MiB resident) is what carries over.
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent


def start_server(port: int):
    code = ("import os,uvicorn;import guarded_distillation.embed as e;e._disk_loaded=True;"      # no precomputed vectors: every wording is embedded by the ONNX model
            "from guarded_distillation.serve.app import make_app;"
            f"uvicorn.run(make_app(None, live=False, db=os.path.join('{tempfile.mkdtemp()}','svc.db')), host='127.0.0.1', port={port}, log_level='warning')")
    env = dict(os.environ, GD_EMBED="onnx", REQ_PER_IP_PER_MIN="100000000", ANTHROPIC_API_KEY="", OMP_NUM_THREADS="1",
               MODEL_REGISTRY="", DATABASE_URL="", WEBDAV_URL="")
    p = subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for _ in range(120):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1).status_code == 200:
                return p
        except Exception:
            time.sleep(0.5)
    p.kill()
    raise SystemExit("server did not start: " + p.stderr.read().decode()[-600:])


async def level(url, texts, conc):
    lat, errs, esc = [], 0, 0
    q = asyncio.Queue()
    for t in texts:
        q.put_nowait(t)

    async def worker(c):
        nonlocal errs, esc
        while True:
            try:
                t = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            t0 = time.perf_counter()
            try:
                r = await c.post(url + "/triage", json={"text": t})
                if r.status_code != 200:
                    errs += 1
                else:
                    esc += r.json().get("decision") == "escalate"
            except Exception:
                errs += 1
            lat.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=60, limits=httpx.Limits(max_connections=conc + 4)) as c:
        await asyncio.gather(*[worker(c) for _ in range(conc)])
    wall = time.perf_counter() - t0
    a = np.array(lat)
    return {"concurrency": conc, "requests": len(texts), "errors": errs, "rps": round(len(texts) / wall, 1),
            "p50_ms": round(float(np.percentile(a, 50)), 1), "p95_ms": round(float(np.percentile(a, 95)), 1),
            "p99_ms": round(float(np.percentile(a, 99)), 1), "escalated": round(esc / max(1, len(texts)), 3)}


def rss_mib(pid):
    try:
        return round(int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)]).split()[0]) / 1024, 1)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 4, 16, 32])
    ap.add_argument("--url", default=None)
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--out", default="docs/loadtest.json")
    a = ap.parse_args()
    s = pd.read_csv(ROOT / "runs/grid-minilm-a03-s3/stream.csv", keep_default_na=False)
    texts = s[s.phase != "history"].text.drop_duplicates().tolist()
    proc = None if a.url else start_server(a.port)
    url = a.url or f"http://127.0.0.1:{a.port}"
    try:
        httpx.post(url + "/triage", json={"text": "warm up"}, timeout=120)           # first request loads the model
        rows, off = [], 0
        for conc in a.levels:
            batch = texts[off:off + a.n]
            off += a.n
            row = asyncio.run(level(url, batch, conc))
            rows.append(row)
            print(row, flush=True)
        out = {"url": "local" if proc else url, "machine": os.uname().machine, "rss_mib": rss_mib(proc.pid) if proc else None, "levels": rows}
    finally:
        if proc:
            proc.terminate()
    Path(ROOT / a.out).write_text(json.dumps(out, indent=1))
    print("resident memory (MiB):", out["rss_mib"])


if __name__ == "__main__":
    main()
