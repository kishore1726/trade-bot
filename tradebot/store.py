"""SQLite state store: JSON key/value, append-only event log, Telegram outbox.

Shared by two processes (daily job + Telegram bot): WAL mode and a busy timeout.
Outside `transaction()` every write commits immediately. Callers convert dates to
ISO strings themselves (plain json).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, text TEXT NOT NULL, sent INTEGER NOT NULL DEFAULT 0);
"""


def _now() -> str:
    return datetime.now(IST).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: autocommit; transactions are explicit BEGIN/COMMIT.
        self._conn = sqlite3.connect(str(path), timeout=10, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    def get(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row[0])

    def set(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT INTO kv(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )

    def log_event(self, kind: str, payload: dict) -> None:
        self._conn.execute(
            "INSERT INTO events(ts, kind, payload) VALUES(?, ?, ?)", (_now(), kind, json.dumps(payload))
        )

    def events(self, kind: str | None = None) -> list[dict]:
        sql, args = "SELECT id, ts, kind, payload FROM events", ()
        if kind is not None:
            sql, args = sql + " WHERE kind=?", (kind,)
        rows = self._conn.execute(sql + " ORDER BY id", args).fetchall()
        return [{"id": i, "ts": t, "kind": k, "payload": json.loads(p)} for i, t, k, p in rows]

    def outbox_add(self, text: str) -> None:
        self._conn.execute("INSERT INTO outbox(ts, text) VALUES(?, ?)", (_now(), text))

    def outbox_pending(self) -> list[tuple[int, str]]:
        rows = self._conn.execute("SELECT id, text FROM outbox WHERE sent=0 ORDER BY id").fetchall()
        return [(i, t) for i, t in rows]

    def outbox_mark_sent(self, id: int) -> None:
        self._conn.execute("UPDATE outbox SET sent=1 WHERE id=?", (id,))
