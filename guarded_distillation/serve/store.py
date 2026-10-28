"""Decision log, events, spend ledger and small state, in Postgres (DATABASE_URL) or a local SQLite file.

Every decision is one JSON line in the `decisions` table. The same store holds the daily spend (so the cap survives a
restart), the model version in use and the versions that were rolled back. On a free host with an ephemeral disk, point
DATABASE_URL at a free Postgres (Neon, Supabase): a restart then keeps the log, the spend and the model version.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path


class Store:
    def __init__(self, path: str | Path = "data/service.db", url: str | None = None):
        self.url = url if url and url.startswith(("postgres://", "postgresql://")) else None
        self.lock = threading.RLock()
        self.pg = bool(self.url)
        self.on_change = None          # replication hooks (serve/replicate.py)
        self.on_critical = None
        self._connect(path)
        self._schema()

    # ------------------------------------------------------------------ plumbing
    def _connect(self, path) -> None:
        self._path = path
        if self.pg:
            import psycopg
            self.db = psycopg.connect(self.url, autocommit=True, connect_timeout=15)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(str(path), check_same_thread=False)

    def _x(self, sql: str, args: tuple = (), fetch: str | None = None):
        """Run one statement; reconnect once if a serverless Postgres dropped the idle connection."""
        if self.pg:
            sql = sql.replace("?", "%s")
        for attempt in (0, 1):
            try:
                with self.lock:
                    cur = self.db.cursor() if self.pg else self.db.cursor()
                    cur.execute(sql, args)
                    out = cur.fetchall() if fetch == "all" else (cur.fetchone() if fetch == "one" else None)
                    if not self.pg:
                        self.db.commit()
                    cur.close()
                    return out
            except Exception as e:
                if self.pg and attempt == 0 and e.__class__.__module__.startswith("psycopg"):
                    try:
                        self._connect(self._path)
                        continue
                    except Exception:
                        pass
                raise

    def _schema(self) -> None:
        if self.pg:
            ddl = ["create table if not exists decisions (id bigserial primary key, ts double precision, version text, data jsonb)",
                   "create table if not exists events (id bigserial primary key, ts double precision, kind text, data jsonb)",
                   "create table if not exists spend (day text, model text, usd double precision, primary key (day, model))",
                   "create table if not exists state (k text primary key, v text)",
                   "create index if not exists d_ver on decisions(version, id)"]
        else:
            ddl = ["create table if not exists decisions (id integer primary key autoincrement, ts real, version text, data text)",
                   "create table if not exists events (id integer primary key autoincrement, ts real, kind text, data text)",
                   "create table if not exists spend (day text, model text, usd real, primary key (day, model))",
                   "create table if not exists state (k text primary key, v text)",
                   "create index if not exists d_ver on decisions(version, id)"]
        for s in ddl:
            self._x(s)

    @staticmethod
    def _j(v):
        return v if isinstance(v, dict) else json.loads(v)

    # ------------------------------------------------------------------ log
    def _changed(self, critical: bool = False) -> None:
        if self.on_change:
            self.on_change()
        if critical and self.on_critical:
            try:
                self.on_critical()
            except Exception:
                pass

    def snapshot(self, dest: str | Path, keep: int = 20000) -> None:
        """A consistent copy of the SQLite file with the decision log pruned to the newest `keep` rows."""
        if self.pg:
            raise RuntimeError("snapshots are for the SQLite store")
        with self.lock:
            out = sqlite3.connect(str(dest))
            self.db.backup(out)
            out.execute("delete from decisions where id <= (select coalesce(max(id), 0) - ? from decisions)", (keep,))
            out.commit()
            out.execute("vacuum")
            out.close()

    def add_decision(self, version: str, rec: dict) -> None:
        self._x("insert into decisions(ts, version, data) values (?,?,?)", (time.time(), version, json.dumps(rec)))
        self._changed()

    def add_event(self, kind: str, data: dict) -> None:
        self._x("insert into events(ts, kind, data) values (?,?,?)", (time.time(), kind, json.dumps(data)))
        self._changed(critical=kind in ("promoted", "rollback"))

    def last(self, n: int, version: str | None = None) -> list[dict]:
        q, args = "select id, ts, version, data from decisions", ()
        if version:
            q, args = q + " where version=?", (version,)
        rows = self._x(q + " order by id desc limit ?", (*args, n), "all")
        return [{"id": r[0], "ts": r[1], "version": r[2], **self._j(r[3])} for r in rows][::-1]

    def count(self, version: str | None = None) -> int:
        if version:
            return self._x("select count(*) from decisions where version=?", (version,), "one")[0]
        return self._x("select count(*) from decisions", (), "one")[0]

    def events(self, kind: str | None = None, limit: int = 50) -> list[dict]:
        rows = self._x("select ts, kind, data from events " + ("where kind=? " if kind else "") + "order by id desc limit ?",   # nosec B608 (values are bound)
                       (kind, limit) if kind else (limit,), "all")
        return [{"ts": r[0], "kind": r[1], **self._j(r[2])} for r in rows]

    def labels(self, limit: int = 100000) -> list[dict]:
        """Teacher-seen answers (escalations and audits) for retraining."""
        if self.pg:
            cond = "data->>'teacher_label' is not null and data->>'source' in ('cache','api')"
        else:
            cond = "json_extract(data,'$.teacher_label') is not null and json_extract(data,'$.source') in ('cache','api')"
        rows = self._x(f"select data from decisions where {cond} order by id desc limit ?", (limit,), "all")   # nosec B608 (cond is a constant)
        return [{"text": d["text"], "label": d["teacher_label"]} for d in (self._j(r[0]) for r in rows)]

    # ------------------------------------------------------------------ spend and state
    def spend_add(self, day: str, model: str, usd: float) -> None:
        self._x("insert into spend(day, model, usd) values (?,?,?) on conflict (day, model) do update set usd = spend.usd + excluded.usd",
                (day, model, usd))
        self._changed(critical=True)

    def spend_today(self, day: str) -> float:
        r = self._x("select coalesce(sum(usd), 0) from spend where day=?", (day,), "one")
        return float(r[0])

    def spend_total(self) -> float:
        return float(self._x("select coalesce(sum(usd), 0) from spend", (), "one")[0])

    def get_state(self, k: str) -> str | None:
        r = self._x("select v from state where k=?", (k,), "one")
        return r[0] if r else None

    def set_state(self, k: str, v: str) -> None:
        self._x("insert into state(k, v) values (?,?) on conflict (k) do update set v = excluded.v", (k, v))
        self._changed(critical=True)
