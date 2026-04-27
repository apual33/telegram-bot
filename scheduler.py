import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Bot

import database

_TZ = ZoneInfo("Europe/Berlin")
logger = logging.getLogger(__name__)


class ReminderScheduler:
    def __init__(self, bot: Bot, db_path: str) -> None:
        self._bot = bot
        self._db_path = db_path
        self._scheduler = AsyncIOScheduler()

    def start(self) -> None:
        self._scheduler.start()

    def stop(self) -> None:
        self._scheduler.shutdown(wait=False)

    def add_reminder(self, todo_id: int, chat_id: int, title: str, run_at: datetime) -> None:
        if run_at.tzinfo is None:
            run_at = run_at.replace(tzinfo=timezone.utc)
        self._scheduler.add_job(
            self._fire,
            trigger="date",
            run_date=run_at,
            args=[todo_id, chat_id, title],
            id=f"reminder_{todo_id}",
            replace_existing=True,
        )
        logger.info("Scheduled reminder for todo %d at %s", todo_id, run_at)

    def remove_reminder(self, todo_id: int) -> None:
        job_id = f"reminder_{todo_id}"
        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)
            logger.info("Cancelled reminder for todo %d", todo_id)

    def add_daily_digest(self, chat_id: int, db_path: str) -> None:
        self._scheduler.add_job(
            self._send_daily_digest,
            trigger="cron",
            hour=8,
            minute=0,
            timezone=_TZ,
            args=[chat_id, db_path],
            id="daily_digest",
            replace_existing=True,
        )
        logger.info("Scheduled daily digest for chat %d at 08:00 Europe/Berlin", chat_id)

    async def _send_daily_digest(self, chat_id: int, db_path: str) -> None:
        todos = database.list_open_todos(db_path, chat_id)
        if not todos:
            await self._bot.send_message(chat_id=chat_id, text="✅ Keine offenen To-Dos — guten Morgen!")
            return

        await self._bot.send_message(
            chat_id=chat_id,
            text=database.format_todo_list(todos),
            parse_mode="Markdown",
        )

    async def _fire(self, todo_id: int, chat_id: int, title: str) -> None:
        await self._bot.send_message(
            chat_id=chat_id,
            text=f"⏰ *Erinnerung:* {title}",
            parse_mode="Markdown",
        )
        database.mark_reminded(self._db_path, todo_id)
        # Save to history so the AI has todo_id in context for snooze/delete replies
        history_text = f"⏰ Erinnerung: {title} [todo_id:{todo_id}]"
        database.append_history(self._db_path, chat_id, "assistant", history_text)
