import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone


def init_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS todos (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id    INTEGER NOT NULL,
                title      TEXT    NOT NULL,
                done       INTEGER NOT NULL DEFAULT 0,
                remind_at  TEXT,
                reminded   INTEGER NOT NULL DEFAULT 0,
                created_at TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id    INTEGER NOT NULL,
                content    TEXT    NOT NULL,
                type       TEXT    NOT NULL,
                created_at TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id    INTEGER NOT NULL,
                role       TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                created_at TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS journal (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id    INTEGER NOT NULL,
                content    TEXT    NOT NULL,
                embedding  TEXT    NOT NULL,
                created_at TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.commit()


@contextmanager
def _conn(db_path: str):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ── Todos (unified with reminders) ────────────────────────────────────────────

def create_todo(
    db_path: str,
    chat_id: int,
    title: str,
    remind_at: datetime | None = None,
) -> int:
    remind_at_str = (
        remind_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        if remind_at is not None
        else None
    )
    with _conn(db_path) as con:
        cur = con.execute(
            "INSERT INTO todos (chat_id, title, remind_at) VALUES (?, ?, ?)",
            (chat_id, title, remind_at_str),
        )
        return cur.lastrowid


def list_open_todos(db_path: str, chat_id: int) -> list[dict]:
    with _conn(db_path) as con:
        rows = con.execute(
            """
            SELECT id, title, created_at, remind_at
            FROM todos
            WHERE chat_id = ? AND done = 0
            ORDER BY created_at ASC
            """,
            (chat_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def complete_todo(db_path: str, todo_id: int) -> bool:
    with _conn(db_path) as con:
        cur = con.execute(
            "UPDATE todos SET done = 1 WHERE id = ? AND done = 0",
            (todo_id,),
        )
        return cur.rowcount > 0


def get_due_reminders(db_path: str) -> list[dict]:
    """Return all todos with a remind_at that haven't been reminded yet and aren't done."""
    with _conn(db_path) as con:
        rows = con.execute(
            """
            SELECT id, chat_id, title, remind_at
            FROM todos
            WHERE remind_at IS NOT NULL AND reminded = 0 AND done = 0
            """,
        ).fetchall()
        return [dict(r) for r in rows]


def mark_reminded(db_path: str, todo_id: int) -> None:
    with _conn(db_path) as con:
        con.execute("UPDATE todos SET reminded = 1 WHERE id = ?", (todo_id,))


# ── Notes ─────────────────────────────────────────────────────────────────────

def save_note(db_path: str, chat_id: int, content: str, note_type: str) -> int:
    with _conn(db_path) as con:
        cur = con.execute(
            "INSERT INTO notes (chat_id, content, type) VALUES (?, ?, ?)",
            (chat_id, content, note_type),
        )
        return cur.lastrowid


def search_notes(db_path: str, chat_id: int, query: str) -> list[dict]:
    with _conn(db_path) as con:
        rows = con.execute(
            "SELECT id, content, type, created_at FROM notes "
            "WHERE chat_id = ? AND content LIKE ? "
            "ORDER BY created_at DESC LIMIT 10",
            (chat_id, f"%{query}%"),
        ).fetchall()
        return [dict(r) for r in rows]


# ── Journal ───────────────────────────────────────────────────────────────────

def save_journal_entry(db_path: str, chat_id: int, content: str, embedding: list) -> int:
    with _conn(db_path) as con:
        cur = con.execute(
            "INSERT INTO journal (chat_id, content, embedding) VALUES (?, ?, ?)",
            (chat_id, content, json.dumps(embedding)),
        )
        return cur.lastrowid


def get_journal_entries(db_path: str, chat_id: int, since_days: int | None = None) -> list[dict]:
    with _conn(db_path) as con:
        if since_days is not None:
            rows = con.execute(
                "SELECT id, content, embedding, created_at FROM journal "
                "WHERE chat_id = ? AND created_at >= datetime('now', ?) "
                "ORDER BY created_at DESC",
                (chat_id, f"-{since_days} days"),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT id, content, embedding, created_at FROM journal "
                "WHERE chat_id = ? ORDER BY created_at DESC",
                (chat_id,),
            ).fetchall()
        return [dict(r) for r in rows]


# ── History ───────────────────────────────────────────────────────────────────

def append_history(db_path: str, chat_id: int, role: str, content: str) -> None:
    with _conn(db_path) as con:
        con.execute(
            "INSERT INTO history (chat_id, role, content) VALUES (?, ?, ?)",
            (chat_id, role, content),
        )


def load_history(db_path: str, chat_id: int, limit: int) -> list[dict]:
    """Return the most recent `limit` user+assistant pairs as [{role, content}]."""
    with _conn(db_path) as con:
        rows = con.execute(
            """
            SELECT role, content FROM (
                SELECT id, role, content FROM history
                WHERE chat_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) ORDER BY id ASC
            """,
            (chat_id, limit * 2),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]
