from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

DEFAULT_SESSION_DB_PATH = "session_data/sessions.db"


class SqliteSessionStore:
    """Thread-safe, LRU tahliyeli ve SQLite tabanlı kalıcı oturum deposu."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        max_sessions: int = 100,
        default_max_turns: int = 5,
    ) -> None:
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int) or max_sessions < 1:
            raise ValueError("max_sessions en az 1 olan bir tam sayı olmalıdır.")
        if (
            isinstance(default_max_turns, bool)
            or not isinstance(default_max_turns, int)
            or default_max_turns < 1
        ):
            raise ValueError("default_max_turns en az 1 olan bir tam sayı olmalıdır.")

        self.db_path = str(db_path)
        self.max_sessions = max_sessions
        self.default_max_turns = default_max_turns
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.db_path != ":memory:":
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
            conn.row_factory = sqlite3.Row

            conn.execute("PRAGMA foreign_keys = ON;")
            if self.db_path != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("PRAGMA synchronous = NORMAL;")

            self._init_schema(conn)
            self._conn = conn

        return self._conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    max_turns INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    access_seq INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session_order
                ON messages(session_id, id)
                """
            )

    def _next_access_seq(self, conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(access_seq), 0) + 1 AS next_seq FROM sessions"
        ).fetchone()
        return int(row["next_seq"])

    def get_history(self, session_id: str) -> list[dict]:
        with self._lock:
            conn = self._ensure_conn()
            row = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return []

            with conn:
                next_seq = self._next_access_seq(conn)
                conn.execute(
                    "UPDATE sessions SET access_seq = ? WHERE session_id = ?",
                    (next_seq, session_id),
                )

            msg_rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            return [{"role": r["role"], "content": r["content"]} for r in msg_rows]

    def record_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        now = time.time()
        with self._lock:
            conn = self._ensure_conn()
            with conn:
                row = conn.execute(
                    "SELECT max_turns FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()

                if row is None:
                    count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
                    while count >= self.max_sessions:
                        victim = conn.execute(
                            """
                            SELECT session_id FROM sessions
                            ORDER BY access_seq ASC, updated_at ASC
                            LIMIT 1
                            """
                        ).fetchone()
                        if victim is None:
                            break
                        conn.execute(
                            "DELETE FROM sessions WHERE session_id = ?",
                            (victim["session_id"],),
                        )
                        count -= 1

                    max_turns = self.default_max_turns
                    next_seq = self._next_access_seq(conn)
                    conn.execute(
                        """
                        INSERT INTO sessions (session_id, max_turns, created_at, updated_at, access_seq)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (session_id, max_turns, now, now, next_seq),
                    )
                else:
                    max_turns = int(row["max_turns"])
                    next_seq = self._next_access_seq(conn)
                    conn.execute(
                        "UPDATE sessions SET updated_at = ?, access_seq = ? WHERE session_id = ?",
                        (now, next_seq, session_id),
                    )

                conn.execute(
                    """
                    INSERT INTO messages (session_id, role, content, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, "user", user_message, now),
                )
                conn.execute(
                    """
                    INSERT INTO messages (session_id, role, content, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, "assistant", assistant_message, now),
                )

                max_messages = max_turns * 2
                msg_count = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0]
                excess = msg_count - max_messages
                if excess > 0:
                    conn.execute(
                        """
                        DELETE FROM messages
                        WHERE id IN (
                            SELECT id FROM messages
                            WHERE session_id = ?
                            ORDER BY id ASC
                            LIMIT ?
                        )
                        """,
                        (session_id, excess),
                    )

    def get_session(self, session_id: str) -> dict | None:
        with self._lock:
            conn = self._ensure_conn()
            row = conn.execute(
                "SELECT max_turns FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None

            with conn:
                next_seq = self._next_access_seq(conn)
                conn.execute(
                    "UPDATE sessions SET access_seq = ? WHERE session_id = ?",
                    (next_seq, session_id),
                )

            msg_rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            messages = [{"role": r["role"], "content": r["content"]} for r in msg_rows]
            return {
                "session_id": session_id,
                "turn_count": len(messages) // 2,
                "max_turns": int(row["max_turns"]),
                "messages": messages,
            }

    def list_sessions(self, limit: int = 50) -> list[dict]:
        if limit < 1:
            return []
        with self._lock:
            conn = self._ensure_conn()
            rows = conn.execute(
                """
                SELECT
                    s.session_id,
                    s.max_turns,
                    s.created_at,
                    s.updated_at,
                    (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.session_id) AS msg_count,
                    (
                        SELECT m.content FROM messages m
                        WHERE m.session_id = s.session_id AND m.role = 'user'
                        ORDER BY m.id ASC
                        LIMIT 1
                    ) AS first_user_msg
                FROM sessions s
                ORDER BY s.access_seq DESC, s.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            results = []
            for row in rows:
                raw_preview = row["first_user_msg"] or ""
                preview = " ".join(str(raw_preview).split())[:80]
                results.append(
                    {
                        "session_id": row["session_id"],
                        "turn_count": int(row["msg_count"]) // 2,
                        "max_turns": int(row["max_turns"]),
                        "created_at": float(row["created_at"]),
                        "updated_at": float(row["updated_at"]),
                        "preview": preview,
                    }
                )
            return results

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            conn = self._ensure_conn()
            with conn:
                cursor = conn.execute(
                    "DELETE FROM sessions WHERE session_id = ?",
                    (session_id,),
                )
                return cursor.rowcount > 0

    def clear_all(self) -> None:
        with self._lock:
            conn = self._ensure_conn()
            with conn:
                conn.execute("DELETE FROM messages")
                conn.execute("DELETE FROM sessions")

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
