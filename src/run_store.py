"""
Persistent history of uploaded eDNA files, their analysis runs, and chatbot
conversations -- one SQLite file, standard library only.

Why SQLite rather than JSON files: run status is written by a background
worker while API requests read it, and the duplicate check must be atomic
(two uploads of the same file arriving together must not both start a run).
SQLite gives both for free and needs no server.

Duplicate detection keys on TWO hashes of every upload:

  * raw_sha256     -- the bytes exactly as uploaded. A client can compute this
                      before uploading (POST /uploads/check) and skip sending
                      a file the server has already analysed.
  * content_sha256 -- the bytes after gunzip. The same reads uploaded once
                      gzipped and once plain are the same data; this catches
                      that case, which raw_sha256 alone would miss.

A file counts as already analysed only for the same marker: running one file
against two reference databases is a legitimate second analysis, not a
redundant upload. A FAILED run never blocks a retry.
"""
import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    raw_sha256      TEXT NOT NULL,
    content_sha256  TEXT NOT NULL,
    marker          TEXT NOT NULL,
    filename        TEXT NOT NULL,
    size_bytes      INTEGER NOT NULL,
    file_format     TEXT NOT NULL,
    stored_path     TEXT NOT NULL,
    status          TEXT NOT NULL,          -- queued | running | done | failed
    progress_done   INTEGER NOT NULL DEFAULT 0,
    progress_total  INTEGER NOT NULL DEFAULT 0,
    error           TEXT,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT,
    results_path    TEXT,
    summary_json    TEXT
);
CREATE INDEX IF NOT EXISTS runs_raw ON runs(raw_sha256, marker);
CREATE INDEX IF NOT EXISTS runs_content ON runs(content_sha256, marker);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id          TEXT PRIMARY KEY,
    run_id      TEXT,
    created_at  TEXT NOT NULL,
    provider    TEXT                        -- "gemini" | "anthropic"; history is stored in that provider's format
);
CREATE TABLE IF NOT EXISTS chat_messages (
    session_id  TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,              -- JSON: a string or a list of content blocks
    created_at  TEXT NOT NULL,
    PRIMARY KEY (session_id, seq)
);
"""

ACTIVE_STATUSES = ("queued", "running", "done")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        # Serialises check-then-insert so concurrent duplicate uploads can't
        # both pass the duplicate check.
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(_SCHEMA)
            # Databases created before chat sessions recorded their provider.
            cols = {r["name"] for r in c.execute("PRAGMA table_info(chat_sessions)")}
            if "provider" not in cols:
                c.execute("ALTER TABLE chat_sessions ADD COLUMN provider TEXT")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------- runs ---

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> Dict[str, Any]:
        run = dict(row)
        summary = run.pop("summary_json")
        run["summary"] = json.loads(summary) if summary else None
        return run

    def find_duplicate(self, marker: Optional[str], raw_sha256: Optional[str] = None,
                       content_sha256: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Most recent non-failed run of the same data against the same marker.
        marker=None (or "auto": the reference set is detected from the reads,
        deterministically) matches a run of the same data whatever its marker."""
        clauses, params = [], []
        if raw_sha256:
            clauses.append("raw_sha256 = ?")
            params.append(raw_sha256)
        if content_sha256:
            clauses.append("content_sha256 = ?")
            params.append(content_sha256)
        if not clauses:
            return None
        any_marker = marker in (None, "auto")
        q = (f"SELECT * FROM runs WHERE ({' OR '.join(clauses)})" + ("" if any_marker else " AND marker = ?") +
             f" AND status IN ({','.join('?' * len(ACTIVE_STATUSES))}) ORDER BY created_at DESC LIMIT 1")
        with self._conn() as c:
            row = c.execute(q, (*params, *(() if any_marker else (marker,)), *ACTIVE_STATUSES)).fetchone()
        return self._row_to_run(row) if row else None

    def create_run_unless_duplicate(self, **fields) -> (Dict[str, Any], bool):
        """Returns (run, created). When the same data was already analysed
        against the same marker, returns that existing run and created=False."""
        with self._lock:
            dup = self.find_duplicate(fields["marker"], fields["raw_sha256"], fields["content_sha256"])
            if dup:
                return dup, False
            run_id = uuid.uuid4().hex[:12]
            row = {"id": run_id, "status": "queued", "created_at": now_iso(), **fields}
            cols = ", ".join(row)
            with self._conn() as c:
                c.execute(f"INSERT INTO runs ({cols}) VALUES ({', '.join('?' * len(row))})", tuple(row.values()))
            return self.get_run(run_id), True

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return self._row_to_run(row) if row else None

    def list_runs(self, marker: Optional[str] = None, status: Optional[str] = None,
                  limit: int = 50, offset: int = 0) -> (List[Dict[str, Any]], int):
        where, params = [], []
        if marker:
            where.append("marker = ?")
            params.append(marker)
        if status:
            where.append("status = ?")
            params.append(status)
        w = f"WHERE {' AND '.join(where)}" if where else ""
        with self._conn() as c:
            total = c.execute(f"SELECT COUNT(*) FROM runs {w}", params).fetchone()[0]
            rows = c.execute(f"SELECT * FROM runs {w} ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                             (*params, limit, offset)).fetchall()
        return [self._row_to_run(r) for r in rows], total

    def update_run(self, run_id: str, **fields) -> None:
        if "summary" in fields:
            fields["summary_json"] = json.dumps(fields.pop("summary"))
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE runs SET {sets} WHERE id = ?", (*fields.values(), run_id))

    def delete_run(self, run_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM runs WHERE id = ?", (run_id,))

    def other_runs_using_file(self, stored_path: str, exclude_run_id: str) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM runs WHERE stored_path = ? AND id != ?",
                             (stored_path, exclude_run_id)).fetchone()[0]

    def fail_interrupted_runs(self) -> int:
        """A run still queued/running at startup was cut off by a restart --
        its worker is gone. Mark it failed so it stops blocking a retry,
        rather than leaving it to look 'in progress' forever."""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE runs SET status = 'failed', error = 'interrupted by a server restart -- upload again to retry', "
                "finished_at = ? WHERE status IN ('queued', 'running')", (now_iso(),))
            return cur.rowcount

    # ------------------------------------------------------------- chat ---

    def create_chat_session(self, run_id: Optional[str], provider: str) -> Dict[str, Any]:
        sid = uuid.uuid4().hex[:12]
        created = now_iso()
        with self._conn() as c:
            c.execute("INSERT INTO chat_sessions (id, run_id, created_at, provider) VALUES (?, ?, ?, ?)",
                      (sid, run_id, created, provider))
        return {"id": sid, "run_id": run_id, "created_at": created, "provider": provider}

    def get_chat_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    def get_chat_messages(self, session_id: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT role, content, created_at FROM chat_messages WHERE session_id = ? ORDER BY seq",
                             (session_id,)).fetchall()
        return [{"role": r["role"], "content": json.loads(r["content"]), "created_at": r["created_at"]} for r in rows]

    def append_chat_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Append-only by design: the model's thinking blocks are only valid
        in the exact conversation that produced them, so history is never
        edited or reordered once stored."""
        created = now_iso()
        with self._conn() as c:
            start = c.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM chat_messages WHERE session_id = ?",
                              (session_id,)).fetchone()[0]
            c.executemany(
                "INSERT INTO chat_messages (session_id, seq, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                [(session_id, start + i, m["role"], json.dumps(m["content"]), created) for i, m in enumerate(messages)])
