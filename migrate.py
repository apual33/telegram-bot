#!/usr/bin/env python3
"""
One-time migration: merge reminders + todos into a unified todos table.
Run ONCE against bot.db before starting the updated bot.
"""
import shutil
import sqlite3
from pathlib import Path

DB = Path(__file__).parent / "bot.db"
BACKUP = Path(__file__).parent / "bot.db.backup"


def migrate() -> None:
    if not DB.exists():
        print(f"ERROR: {DB} not found.")
        return

    # 1. Back up
    shutil.copy2(DB, BACKUP)
    print(f"Backed up {DB} → {BACKUP}")

    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row

        existing = {
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        reminders: list = []
        old_todos: list = []

        if "reminders" in existing:
            reminders = con.execute("SELECT * FROM reminders").fetchall()
            print(f"  reminders: {len(reminders)} rows")

        if "todos" in existing:
            old_todos = con.execute("SELECT * FROM todos").fetchall()
            print(f"  todos (old): {len(old_todos)} rows")
            con.execute("ALTER TABLE todos RENAME TO todos_old")

        # 2. Create new unified todos table
        con.execute("""
            CREATE TABLE todos (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id    INTEGER NOT NULL,
                title      TEXT    NOT NULL,
                done       INTEGER NOT NULL DEFAULT 0,
                remind_at  TEXT,
                reminded   INTEGER NOT NULL DEFAULT 0,
                created_at TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)

        # 3. Reminders → new todos (remind_at = scheduled_for, reminded = 0)
        reminder_to_todo: dict[int, int] = {}
        for r in reminders:
            cur = con.execute(
                "INSERT INTO todos (chat_id, title, done, remind_at, reminded, created_at) "
                "VALUES (?, ?, 0, ?, 0, ?)",
                (r["chat_id"], r["message"], r["scheduled_for"], r["created_at"]),
            )
            reminder_to_todo[r["id"]] = cur.lastrowid

        # 4. Old todos → new todos
        for t in old_todos:
            rid = t["reminder_id"] if "reminder_id" in t.keys() else None
            if rid is not None and rid in reminder_to_todo:
                # Linked to a migrated reminder — propagate done flag only
                if t["done"]:
                    con.execute(
                        "UPDATE todos SET done = 1 WHERE id = ?",
                        (reminder_to_todo[rid],),
                    )
            else:
                # Standalone todo without a reminder
                con.execute(
                    "INSERT INTO todos (chat_id, title, done, remind_at, reminded, created_at) "
                    "VALUES (?, ?, ?, NULL, 0, ?)",
                    (t["chat_id"], t["title"], t["done"], t["created_at"]),
                )

        # 5. Drop old tables
        refreshed = {
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "todos_old" in refreshed:
            con.execute("DROP TABLE todos_old")
        if "reminders" in existing:
            con.execute("DROP TABLE reminders")

        con.commit()

    # Report
    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT id, chat_id, title, done, remind_at, reminded FROM todos ORDER BY id"
        ).fetchall()

    print(f"\nDone — {len(rows)} todos in new table:")
    for r in rows:
        remind = f"  remind_at={r['remind_at']}" if r["remind_at"] else ""
        print(f"  [{r['id']}] {r['title']!r}  done={r['done']} reminded={r['reminded']}{remind}")


def migrate_journal_to_notes() -> None:
    """
    Migration: consolidate journal into notes.
    - Backs up bot.db to bot.db.backup
    - Copies all journal rows into notes with type='note'
    - Drops the journal table
    """
    if not DB.exists():
        print(f"ERROR: {DB} not found.")
        return

    shutil.copy2(DB, BACKUP)
    print(f"Backed up {DB} → {BACKUP}")

    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row

        existing = {
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        if "journal" not in existing:
            print("journal table does not exist — nothing to migrate.")
            return

        rows = con.execute(
            "SELECT chat_id, content, created_at FROM journal"
        ).fetchall()
        print(f"  journal: {len(rows)} rows to migrate")

        for r in rows:
            con.execute(
                "INSERT INTO notes (chat_id, content, type, created_at) VALUES (?, ?, 'note', ?)",
                (r["chat_id"], r["content"], r["created_at"]),
            )

        con.execute("DROP TABLE journal")
        con.commit()

    with sqlite3.connect(DB) as con:
        count = con.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    print(f"Done — notes table now has {count} rows. journal table dropped.")


if __name__ == "__main__":
    migrate_journal_to_notes()
