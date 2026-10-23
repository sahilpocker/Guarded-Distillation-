"""SQLite store: every decision as one JSON line, plus an events log (promotions, rollbacks, alerts)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path


class Store:
    def __init__(self, path: str | Path = "data/service.db"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript("""
              create table if not exists decisions (id integer primary key autoincrement, ts real, version text, data text);
              create table if not exists events (id integer primary key autoincrement, ts real, kind text, data text);
              create index if not exists d_ts on decisions(ts);
            """)

    def add_decision(self, version: str, rec: dict) -> int:
        with self.lock:
            cur = self.db.execute("insert into decisions(ts, version, data) values (?,?,?)", (time.time(), version, json.dumps(rec)))
            self.db.commit()
            return cur.lastrowid

    def add_event(self, kind: str, data: dict) -> None:
        with self.lock:
            self.db.execute("insert into events(ts, kind, data) values (?,?,?)", (time.time(), kind, json.dumps(data)))
            self.db.commit()

    def last(self, n: int, version: str | None = None) -> list[dict]:
        q, args = "select id, ts, version, data from decisions", ()
        if version:
            q, args = q + " where version=?", (version,)
        with self.lock:
            rows = self.db.execute(q + " order by id desc limit ?", (*args, n)).fetchall()
        return [{"id": r[0], "ts": r[1], "version": r[2], **json.loads(r[3])} for r in rows][::-1]

    def count(self, version: str | None = None) -> int:
        with self.lock:
            if version:
                return self.db.execute("select count(*) from decisions where version=?", (version,)).fetchone()[0]
            return self.db.execute("select count(*) from decisions").fetchone()[0]

    def events(self, kind: str | None = None, limit: int = 50) -> list[dict]:
        with self.lock:
            rows = self.db.execute("select ts, kind, data from events " + ("where kind=? " if kind else "") + "order by id desc limit ?",
                                   ((kind, limit) if kind else (limit,))).fetchall()
        return [{"ts": r[0], "kind": r[1], **json.loads(r[2])} for r in rows]

    def labels(self, limit: int = 100000) -> list[dict]:
        """Teacher-seen answers (escalations and audits) for retraining."""
        with self.lock:
            rows = self.db.execute("select data from decisions where json_extract(data,'$.teacher_label') is not null "
                                   "and json_extract(data,'$.source') in ('cache','api') order by id desc limit ?", (limit,)).fetchall()
        return [{"text": d["text"], "label": d["teacher_label"]} for d in (json.loads(r[0]) for r in rows)]
