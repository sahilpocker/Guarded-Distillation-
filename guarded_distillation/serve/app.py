"""FastAPI app: POST /triage, GET /metrics, GET /healthz. Admin routes (reload, export) need ADMIN_TOKEN."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import monitor
from .bundle import Bundle
from .store import Store
from .teacher_access import TeacherAccess
from .triage import Triage


class Query(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


def make_app(bundle_dir: str | Path | None = None, db: str | None = None, cap_usd: float | None = None, live: bool | None = None) -> FastAPI:
    bundle_dir = bundle_dir or os.environ.get("GD_BUNDLE", "models/v1")
    bundle = Bundle.load(bundle_dir)
    cap = float(os.environ.get("DAILY_CAP_USD", 1.0)) if cap_usd is None else cap_usd
    live = (bool(os.environ.get("ANTHROPIC_API_KEY")) if live is None else live)
    teacher = TeacherAccess(bundle.intents, cap_usd=cap, live=live, ledger=os.environ.get("GD_LEDGER", "data/service_spend.json"))
    store = Store(db or os.environ.get("GD_DB", "data/service.db"))
    tri = Triage(bundle, teacher, store)
    app = FastAPI(title="Guarded Distillation triage", version=bundle.version)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
    app.state.triage = tri

    def admin(token: str | None):
        want = os.environ.get("ADMIN_TOKEN")
        if not want or token != want:
            raise HTTPException(401, "admin token required")

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "model_version": tri.b.version, "live_api": teacher.live}

    @app.post("/triage")
    def triage(q: Query):
        return tri.decide(q.text.strip())

    @app.get("/metrics")
    def metrics(window: int = 500):
        return monitor.compute(tri, min(max(window, 50), 5000))

    @app.get("/admin/export")
    def export(x_admin_token: str | None = Header(default=None)):
        admin(x_admin_token)
        return {"labels": store.labels()}

    @app.post("/admin/reload")
    def reload(path: str, x_admin_token: str | None = Header(default=None)):
        admin(x_admin_token)
        from .rollout import swap
        return swap(tri, path)

    return app
