"""Tiny SQLite store that records every uploaded image (data saved for retraining)."""
import sqlite3
import time
from contextlib import contextmanager

from .config import DB_PATH


@contextmanager
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        con.execute("""CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            label TEXT NOT NULL,
            path TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending')""")
        yield con
        con.commit()
    finally:
        con.close()


def add_upload(filename, label, path):
    with _conn() as con:
        con.execute("INSERT INTO uploads (filename, label, path, uploaded_at) VALUES (?,?,?,?)",
                    (filename, label, str(path), time.strftime("%Y-%m-%d %H:%M:%S")))


def pending_uploads():
    with _conn() as con:
        return [dict(r) for r in con.execute("SELECT * FROM uploads WHERE status='pending'")]


def mark_ingested(ids):
    with _conn() as con:
        con.executemany("UPDATE uploads SET status='ingested' WHERE id=?", [(i,) for i in ids])


def upload_summary():
    with _conn() as con:
        rows = con.execute("SELECT status, label, COUNT(*) n FROM uploads GROUP BY status, label")
        return [dict(r) for r in rows]
