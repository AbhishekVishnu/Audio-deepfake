"""SQLite-backed history of analyses."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "history.db"
KEEP_LAST = 50


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init() -> None:
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                source TEXT NOT NULL,
                filename TEXT,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                fake_prob REAL NOT NULL,
                duration REAL,
                chunks_json TEXT NOT NULL,
                models_json TEXT NOT NULL
            )"""
        )


def save(*, source: str, filename: str | None, label: str, confidence: float,
         fake_prob: float, duration: float, chunks: list, models: list[str]) -> int:
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO analyses (ts, source, filename, label, confidence, fake_prob,
                                     duration, chunks_json, models_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (int(time.time()), source, filename, label, confidence, fake_prob, duration,
             json.dumps(chunks), json.dumps(models)),
        )
        new_id = int(cur.lastrowid)
        c.execute(
            """DELETE FROM analyses WHERE id NOT IN (
                 SELECT id FROM analyses ORDER BY id DESC LIMIT ?
               )""",
            (KEEP_LAST,),
        )
    return new_id


def list_recent(limit: int = 20) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["chunks"] = json.loads(d.pop("chunks_json"))
        d["models"] = json.loads(d.pop("models_json"))
        out.append(d)
    return out


def clear() -> None:
    with _conn() as c:
        c.execute("DELETE FROM analyses")
