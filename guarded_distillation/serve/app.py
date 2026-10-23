"""FastAPI app: POST /triage, GET /metrics, GET /healthz. Admin routes (reload, export) need ADMIN_TOKEN."""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import monitor
from .bundle import Bundle
from .store import Store
from .teacher_access import DbLedger, TeacherAccess
from .triage import Triage


LANDING = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Guarded Distillation triage service</title>
<style>body{font:17px/1.55 Georgia,serif;max-width:640px;margin:40px auto;padding:0 18px;color:#13202b}h1{font:700 28px/1.1 system-ui,sans-serif}
a{color:#0b7a75}input{width:100%;padding:10px;font:inherit;border:1.5px solid #13202b;border-radius:8px;margin:6px 0}button{font:600 15px system-ui;padding:10px 18px;border-radius:999px;border:0;background:#13202b;color:#fff;cursor:pointer}
pre{background:#eef2f4;padding:12px;border-radius:8px;overflow:auto;font-size:14px}.s{color:#455765;font-size:15px}</style></head><body>
<h1>Guarded Distillation triage service</h1>
<p>A support-question router. A small model answers the questions it is sure about and hands the rest to Claude Haiku. Model {{VERSION}}.
This is a free server: it sleeps when idle and the first request can take up to a minute.</p>
<p>Try it: type a banking support question.</p>
<input id="q" maxlength="300" value="I have not received my card yet, where is it?" aria-label="Support question">
<button id="go">Route it</button><pre id="out">The answer appears here.</pre>
<p class="s">Endpoints: <a href="/metrics">/metrics</a> (rolling escalation rate, estimated leak rate, per-intent stress, spend),
<a href="/alerts">/alerts</a>, <a href="/healthz">/healthz</a>, <a href="/docs">/docs</a> (POST /triage). Project: <a href="https://github.com/sahilpocker/Guarded-Distillation-">source and results</a>,
<a href="https://guarded-distillation.vercel.app">the explainer</a>.</p>
<script>document.getElementById("go").onclick=async()=>{const o=document.getElementById("out");o.textContent="working...";
try{const r=await fetch("/triage",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({text:document.getElementById("q").value})});
o.textContent=JSON.stringify(await r.json(),null,2)}catch(e){o.textContent="request failed: "+e}};</script></body></html>"""


class Query(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


def make_app(bundle_dir: str | Path | None = None, db: str | None = None, cap_usd: float | None = None, live: bool | None = None) -> FastAPI:
    bundle_dir = bundle_dir or os.environ.get("GD_BUNDLE") or ("models/current" if Path("models/current/bundle.pkl").exists() else "models/v1")
    bundle = Bundle.load(bundle_dir)
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    clean = key.strip().strip("\"'").strip()
    sanitized = bool(key) and clean != key          # a pasted key often carries a newline, which breaks the request header
    if sanitized:
        os.environ["ANTHROPIC_API_KEY"] = clean
    cap = float(os.environ.get("DAILY_CAP_USD", 1.0)) if cap_usd is None else cap_usd
    live = (bool(os.environ.get("ANTHROPIC_API_KEY")) if live is None else live)
    store = Store(db or os.environ.get("GD_DB", "data/service.db"), url=os.environ.get("DATABASE_URL"))
    teacher = TeacherAccess(bundle.intents, cap_usd=cap, live=live, ledger=DbLedger(store, cap))
    tri = Triage(bundle, teacher, store)
    tri.blocked = set(json.loads(store.get_state("blocked") or "[]"))        # rolled-back versions stay rolled back after a restart
    want = store.get_state("model_version")
    if want and want != bundle.version and os.environ.get("MODEL_REGISTRY"):     # the version the last run was serving
        try:
            from .rollout import Registry, version_number
            reg = Registry(os.environ["MODEL_REGISTRY"])
            rel = next((r for r in reg.releases() if r["n"] == version_number(want)), None)
            if rel and want not in tri.blocked:
                tri.set_bundle(reg.fetch(rel))
        except Exception as e:
            store.add_event("registry_error", {"error": f"restoring {want}: {str(e)[:150]}"})
    store.set_state("model_version", tri.b.version)
    app = FastAPI(title="Guarded Distillation triage", version=bundle.version)
    if os.environ.get("MODEL_REGISTRY"):          # owner/repo whose releases hold model-vN bundles
        from .rollout import start_poller
        start_poller(tri, os.environ["MODEL_REGISTRY"], int(os.environ.get("REGISTRY_POLL_S", 900)))
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
    app.state.triage = tri

    def admin(token: str | None):
        want = os.environ.get("ADMIN_TOKEN")
        if not want or token != want:
            raise HTTPException(401, "admin token required")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home():
        return LANDING.replace("{{VERSION}}", tri.b.version)

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "model_version": tri.b.version, "live_api": teacher.live, "key_sanitized": sanitized}

    @app.post("/triage")
    def triage(q: Query):
        return tri.decide(q.text.strip())

    @app.get("/metrics")
    def metrics(window: int = 500):
        return monitor.compute(tri, min(max(window, 50), 5000))

    @app.get("/alerts")
    def alerts():
        m = monitor.compute(tri)
        return {"alerts": monitor.alerts(m), "model_version": m["model_version"], "window": m["window"]}

    @app.get("/admin/export")
    def export(x_admin_token: str | None = Header(default=None)):
        admin(x_admin_token)
        return {"labels": store.labels()}

    @app.post("/admin/reload")
    def reload(path: str, x_admin_token: str | None = Header(default=None)):
        admin(x_admin_token)
        from .rollout import swap
        return swap(tri, path, reason="admin")

    return app
