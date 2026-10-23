"""Keep the service's SQLite file alive across restarts of a free host by replicating it to WebDAV (Koofr, Nextcloud, ...).

The working copy stays local (fast, ordinary SQLite). A snapshot is uploaded (gzip, decisions pruned to the newest
`keep`) every `interval` seconds while something changed, and immediately after anything that must not be lost: spend,
the model version, a rollback. On start the newest snapshot is restored before the store opens. The window of loss in a
crash is `interval` seconds of decisions; spend and model state are never older than the last paid call or promotion.

Koofr: WEBDAV_URL=https://app.koofr.net/dav/Koofr, WEBDAV_USER=<account email>, WEBDAV_PASSWORD=<app password from
Koofr > Preferences > Password > "App passwords">, WEBDAV_DIR=guarded-distillation.
"""
from __future__ import annotations

import gzip
import os
import shutil
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

import httpx


class WebDavReplica:
    def __init__(self, url: str, user: str, password: str, remote_dir: str = "guarded-distillation", interval: float = 60.0,
                 keep: int = 20000, name: str = "service.db.gz"):
        self.base = url.rstrip("/")
        self.dir = remote_dir.strip("/")
        self.auth = (user, password) if user else None
        self.interval, self.keep, self.name = interval, keep, name
        self.lock = threading.Lock()
        self.dirty = False
        self.last_push = 0.0
        self.last_error: str | None = None
        self.pushes = 0
        self.store = None

    @classmethod
    def from_env(cls) -> "WebDavReplica | None":
        e = os.environ.get
        if not e("WEBDAV_URL"):
            return None
        return cls(e("WEBDAV_URL"), e("WEBDAV_USER", "").strip(), e("WEBDAV_PASSWORD", "").strip(), e("WEBDAV_DIR", "guarded-distillation"),
                   float(e("WEBDAV_INTERVAL_S", 60)))

    def _url(self) -> str:
        return f"{self.base}/{self.dir}/{self.name}"

    def _client(self) -> httpx.Client:
        return httpx.Client(auth=self.auth, timeout=30, follow_redirects=True)

    def restore(self, local: str | Path) -> bool:
        """Download the newest snapshot to `local` if there is one. Never raises: a dead remote must not stop the service."""
        try:
            with self._client() as c:
                r = c.get(self._url())
            if r.status_code != 200:
                return False
            local = Path(local)
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(gzip.decompress(r.content))
            return True
        except Exception as e:
            self.last_error = f"restore: {e}"
            return False

    def _ensure_dir(self, c: httpx.Client) -> None:
        parts, acc = self.dir.split("/"), ""
        for p in parts:
            acc += "/" + p
            c.request("MKCOL", f"{self.base}{acc}")           # 201 created, 405 already exists: both fine

    def push(self) -> bool:
        if self.store is None:
            return False
        with self.lock:
            try:
                tmp = Path(tempfile.mkdtemp())
                snap = tmp / "snap.db"
                self.store.snapshot(snap, keep=self.keep)
                blob = gzip.compress(snap.read_bytes(), 6)
                with self._client() as c:
                    r = c.put(self._url(), content=blob)
                    if r.status_code in (404, 409):               # the folder is missing: create it and retry once
                        self._ensure_dir(c)
                        r = c.put(self._url(), content=blob)
                shutil.rmtree(tmp, ignore_errors=True)
                if r.status_code not in (200, 201, 204):
                    raise RuntimeError(f"PUT {r.status_code}")
                self.dirty, self.last_push, self.last_error = False, time.time(), None
                self.pushes += 1
                return True
            except Exception as e:
                self.last_error = f"push: {e}"
                return False

    def start(self, store) -> None:
        self.store = store
        store.on_critical = self.push                              # spend, model version, rollback: upload now
        store.on_change = lambda: setattr(self, "dirty", True)

        def loop():
            while True:
                time.sleep(self.interval)
                if self.dirty:
                    self.push()
        threading.Thread(target=loop, daemon=True, name="webdav-replica").start()
