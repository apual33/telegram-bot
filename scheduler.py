import logging
import random
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from anthropic import AsyncAnthropic
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Bot

import calendar_integration
import database
from config import Config

_TZ = ZoneInfo("Europe/Berlin")
logger = logging.getLogger(__name__)


class ReminderScheduler:
    def __init__(self, bot: Bot, db_path: str) -> None:
        self._bot = bot
        self._db_path = db_path
        self._scheduler = AsyncIOScheduler()
        self._config: Config | None = None

    def set_config(self, config: Config) -> None:
        self._config = config

    def set_anthropic_client(self, client: AsyncAnthropic) -> None:
        self._anthropic = client

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
            hour=6,
            minute=0,
            timezone=_TZ,
            args=[chat_id, db_path],
            id="daily_digest",
            replace_existing=True,
        )
        logger.info("Scheduled daily digest for chat %d at 08:00 Europe/Berlin", chat_id)

    async def _send_daily_digest(self, chat_id: int, db_path: str) -> None:
        parts: list[str] = []

        # ── Calendar events ────────────────────────────────────────────────────
        if self._config and self._config.google_credentials_file:
            try:
                events = await calendar_integration.fetch_today_events_async(
                    self._config.google_token_file,
                    self._config.google_calendar_id,
                )
                parts.append(calendar_integration.format_events(events))
            except RuntimeError as exc:
                if str(exc) == "no_token":
                    logger.warning("Google Calendar token missing — notifying user")
                    await self._bot.send_message(
                        chat_id=chat_id,
                        text="⚠️ Google Calendar nicht verbunden. Bitte /auth\\_calendar ausführen.",
                        parse_mode="Markdown",
                    )
                else:
                    logger.exception("Calendar fetch failed")
            except Exception:
                logger.exception("Calendar fetch failed")

        # ── To-do list ─────────────────────────────────────────────────────────
        todos = database.list_open_todos(db_path, chat_id)
        if todos:
            parts.append(database.format_todo_list(todos))
        else:
            parts.append("✅ Keine offenen To-Dos")

        # ── Daily impulse ─────────────────────────────────────────────────────
        impulse = await self._generate_daily_impulse()
        if impulse:
            parts.append(f"🌱 Impuls für heute:\n{impulse}")

        digest_text = "\n\n".join(parts)
        await self._bot.send_message(
            chat_id=chat_id,
            text=digest_text,
        )

        # Save impulse to history so the AI can respond to follow-ups
        if impulse:
            database.append_history(db_path, chat_id, "assistant", digest_text)

    async def _generate_daily_impulse(self) -> str | None:
        """Generate a daily reflection question or challenge via Claude Sonnet."""
        if not hasattr(self, "_anthropic") or not self._anthropic:
            logger.warning("Anthropic client not set — skipping daily impulse")
            return None

        category = random.choice(["reflection", "challenge"])
        if category == "reflection":
            category_instruction = (
                "Generiere eine tiefe, persönliche Reflexionsfrage. "
                "Die Frage soll zum Nachdenken anregen und ehrliche Selbstreflexion fördern. "
                "Beispiel-Niveau (aber nicht kopieren): 'Wann hast du zuletzt etwas getan, das dich wirklich stolz gemacht hat?'"
            )
        else:
            category_instruction = (
                "Generiere eine konkrete kleine Tagesaufgabe/Challenge. "
                "Sie soll entweder aus der Komfortzone holen oder eine Beziehung stärken. "
                "Beispiel-Niveau (aber nicht kopieren): 'Schreib heute jemandem eine Nachricht, dem du lange nicht geschrieben hast.'"
            )

        try:
            response = await self._anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                system=(
                    "Du bist ein einfühlsamer persönlicher Coach. "
                    "Generiere entweder eine tiefe Reflexionsfrage oder eine konkrete Tagesaufgabe. "
                    "Keine Klischees. Keine oberflächlichen Fragen. "
                    "Sei kreativ, spezifisch und persönlich. "
                    "Antworte nur mit dem Impuls selbst, kein Intro, keine Erklärung."
                ),
                messages=[{"role": "user", "content": category_instruction}],
            )
            return response.content[0].text.strip()
        except Exception:
            logger.exception("Failed to generate daily impulse")
            return None

    async def _fire(self, todo_id: int, chat_id: int, title: str) -> None:
        msg = await self._bot.send_message(
            chat_id=chat_id,
            text=f"⏰ *Erinnerung:* {title}",
            parse_mode="Markdown",
        )
        database.set_todo_message_id(self._db_path, todo_id, msg.message_id)
        database.mark_reminded(self._db_path, todo_id)
        # Save to history so the AI has todo_id in context for snooze/delete replies
        history_text = f"⏰ Erinnerung: {title} [todo_id:{todo_id}]"
        database.append_history(self._db_path, chat_id, "assistant", history_text)
