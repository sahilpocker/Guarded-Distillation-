"""Model rollout: hot-swap a new bundle into the service, and roll back automatically if it is worse in production.

ROLLBACK RULE. After a new version has served 500 decisions, compare its audit disagreement rate (audited answers
that differ from the teacher) with the previous version's rate over that version's last 500 decisions. If the new
rate is at least 50% higher (against a floor of 2% for the old rate, so a quiet old model does not make every
fluctuation a regression) and at least 15 answers were audited, restore the previous bundle, block the bad version
and record a `rollback` event (which the monitor turns into an alert).

REGISTRY. GitHub releases tagged `model-vN`, each with `bundle-vN.tar.gz` and a `sha256: <hex>` line in the notes.
The service polls the releases of $MODEL_REGISTRY (owner/repo) and promotes the newest version it has not blocked.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from . import monitor
from .bundle import Bundle

WINDOW = 500
RISE = 1.5
FLOOR = 0.02
MIN_AUDITED = 15


def swap(tri, bundle: Bundle | str | Path, reason: str = "manual") -> dict:
    """Make `bundle` the live model, remembering the old one (and its audit disagreement rate) for rollback."""
    new = bundle if isinstance(bundle, Bundle) else Bundle.load(bundle)
    old = tri.b
    with tri.lock:
        prev_stats = monitor.audit_stats(tri.store.last(WINDOW, version=old.version))
        tri.prev = {"bundle": old, "rate": prev_stats["rate"], "audited": prev_stats["audited"]}
        tri.set_bundle(new)
        tri.n_cur, tri.checked = 0, False
    info = {"from": old.version, "to": new.version, "reason": reason, "previous_audit_rate": prev_stats["rate"],
            "previous_audited": prev_stats["audited"]}
    tri.store.add_event("promoted", info)
    return info


def maybe_rollback(tri) -> dict | None:
    """Called after every decision; acts once, when the current version reaches WINDOW decisions."""
    if tri.prev is None or tri.checked or tri.n_cur < WINDOW:
        return None
    tri.checked = True
    cur = monitor.audit_stats(tri.store.last(WINDOW, version=tri.b.version))
    base = tri.prev["rate"] if tri.prev["rate"] is not None else 0.0
    limit = RISE * max(base, FLOOR)
    worse = cur["audited"] >= MIN_AUDITED and cur["rate"] is not None and cur["rate"] >= limit
    detail = {"version": tri.b.version, "previous": tri.prev["bundle"].version, "audit_rate": cur["rate"], "audited": cur["audited"],
              "previous_rate": tri.prev["rate"], "limit": limit}
    tri.store.add_event("audit_check", {**detail, "worse": worse})
    if not worse:
        return None
    bad = tri.b.version
    with tri.lock:
        tri.set_bundle(tri.prev["bundle"])
        tri.blocked.add(bad)
        tri.prev, tri.n_cur = None, 0
    tri.store.add_event("rollback", {**detail, "restored": tri.b.version})
    return detail


class Registry:
    def __init__(self, repo: str, cache: str | Path | None = None):
        self.repo, self.cache = repo, Path(cache or tempfile.gettempdir()) / "gd-models"

    def _get(self, url: str) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": "guarded-distillation", "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=60) as r:      # noqa: S310 (fixed github.com URLs)
            return r.read()

    def releases(self) -> list[dict]:
        rows = json.loads(self._get(f"https://api.github.com/repos/{self.repo}/releases?per_page=20"))
        out = []
        for r in rows:
            m = re.fullmatch(r"model-v(\d+)", r.get("tag_name", ""))
            asset = next((a for a in r.get("assets", []) if a["name"].startswith("bundle-v")), None)
            sha = re.search(r"sha256:\s*([0-9a-f]{64})", r.get("body") or "")
            if m and asset and sha and not r.get("draft"):
                out.append({"n": int(m.group(1)), "tag": r["tag_name"], "url": asset["browser_download_url"], "sha256": sha.group(1), "body": r.get("body", "")})
        return sorted(out, key=lambda x: -x["n"])

    def fetch(self, rel: dict) -> Bundle:
        blob = self._get(rel["url"])
        if hashlib.sha256(blob).hexdigest() != rel["sha256"]:
            raise ValueError(f"{rel['tag']}: sha256 mismatch")
        dest = self.cache / rel["tag"]
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
            tf.extractall(dest, filter="data")
        return Bundle.load(next(dest.rglob("bundle.pkl")).parent)


def version_number(v: str) -> int:
    m = re.search(r"(\d+)$", v)
    return int(m.group(1)) if m else 0


def poll_once(tri, reg: Registry) -> dict:
    cur = version_number(tri.b.version)
    for rel in reg.releases():
        if rel["n"] <= cur:
            break
        if f"v{rel['n']}" in tri.blocked:
            continue
        return swap(tri, reg.fetch(rel), reason=f"registry {rel['tag']}")
    return {"updated": False, "current": tri.b.version}


def start_poller(tri, repo: str, every_s: int = 900) -> threading.Thread:
    reg = Registry(repo)

    def loop():
        time.sleep(20)
        while True:
            try:
                poll_once(tri, reg)
            except Exception as e:                      # a registry outage must never take the service down
                tri.store.add_event("registry_error", {"error": str(e)[:200]})
            time.sleep(every_s)
    t = threading.Thread(target=loop, daemon=True, name="registry-poller")
    t.start()
    return t
