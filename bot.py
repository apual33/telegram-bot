import asyncio
import logging
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import ai
import calendar_integration
import database
import research
import voice as voice_module
from config import Config
from scheduler import ReminderScheduler

logger = logging.getLogger(__name__)
_TZ = ZoneInfo("Europe/Berlin")

# ── Core pipeline ─────────────────────────────────────────────────────────────

async def _pipeline(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    chat_id = update.effective_chat.id
    anthropic_client: AsyncAnthropic = context.bot_data["anthropic_client"]
    db_path: str = context.bot_data["db_path"]
    sched: ReminderScheduler = context.bot_data["scheduler"]
    max_pairs: int = context.bot_data["history_max_pairs"]
    config: Config = context.bot_data["config"]

    # Load recent history from DB and append current user message
    history = database.load_history(db_path, chat_id, max_pairs)
    working = [*history, {"role": "user", "content": text}]

    await update.message.chat.send_action("typing")

    try:
        reply = await ai.run(chat_id, working, anthropic_client, db_path, sched, config)
    except Exception:
        logger.exception("AI pipeline error for chat %d", chat_id)
        await update.message.reply_text("Sorry, something went wrong. Please try again.")
        return

    # Persist the exchange to DB (never trimmed)
    database.append_history(db_path, chat_id, "user", text)
    database.append_history(db_path, chat_id, "assistant", reply)

    await update.message.reply_text(reply)


# ── Research pipeline ─────────────────────────────────────────────────────────

async def _research_pipeline(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    chat_id = update.effective_chat.id
    client: AsyncAnthropic = context.bot_data["anthropic_client"]
    config: Config = context.bot_data["config"]

    if not config.gmail_address or not config.report_email:
        await update.message.reply_text(
            "Research feature not configured. Please set GMAIL_ADDRESS, GMAIL_APP_PASSWORD, and REPORT_EMAIL in .env."
        )
        return

    topic = research.extract_topic(text)
    await update.message.reply_text(f"Starting research on: {topic}\nI'll email the report to {config.report_email} when done.")

    async def _do_research() -> None:
        try:
            report_text = await research.run(text, client)
            safe_topic = re.sub(r"[^\w\s-]", "", topic)[:40].strip().replace(" ", "_")
            filename = f"report_{safe_topic}_{datetime.now(_TZ).strftime('%Y%m%d')}.docx"
            docx_buf = research.build_docx(topic, report_text)
            await asyncio.get_event_loop().run_in_executor(
                None, research.send_email, topic, docx_buf, filename, config
            )
            await context.bot.send_message(chat_id, f"Done! Report on \"{topic}\" sent to {config.report_email}.")
        except asyncio.TimeoutError:
            logger.error("Research timed out for chat %d", chat_id)
            await context.bot.send_message(chat_id, "Sorry, the research request timed out (no response from the API after 120 s). Please try again.")
        except Exception:
            logger.exception("Research pipeline error for chat %d", chat_id)
            await context.bot.send_message(chat_id, "Sorry, something went wrong during research.")

    asyncio.create_task(_do_research())


# ── Handlers ──────────────────────────────────────────────────────────────────

_COMPLETION_KEYWORDS = (
    "ist erledigt",
    "ist fertig",
    "habe erledigt",
    "kannst du löschen",
    "bitte löschen",
)


async def handle_calendar_code(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.bot_data["config"]
    code = update.message.text.strip()
    try:
        calendar_integration.exchange_code(code, config.google_token_file)
    except Exception:
        logger.exception("Failed to exchange Google auth code")
        await update.message.reply_text(
            "Ungültiger Code oder Fehler beim Austausch. Bitte /auth\\_calendar erneut versuchen.",
            parse_mode="Markdown",
        )
        return
    context.bot_data.pop("awaiting_calendar_code", None)
    await update.message.reply_text("✅ Google Calendar erfolgreich verbunden!")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.bot_data.get("awaiting_calendar_code"):
        await handle_calendar_code(update, context)
        return

    text = update.message.text
    lower = text.lower()
    if any(kw in lower for kw in _COMPLETION_KEYWORDS):
        text = "WICHTIG: Rufe ZUERST list_todos auf, dann complete_todo. Keine Ausnahmen.\n" + text
    if research.is_research_request(text):
        await _research_pipeline(update, context, text)
    else:
        await _pipeline(update, context, text)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    openai_client: AsyncOpenAI = context.bot_data["openai_client"]
    voice = update.message.voice

    tg_file = await context.bot.get_file(voice.file_id)

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        await tg_file.download_to_drive(tmp_path)
        await update.message.chat.send_action("typing")
        text = await voice_module.transcribe(tmp_path, openai_client)
    except Exception:
        logger.exception("Voice transcription error")
        await update.message.reply_text("Sorry, I couldn't transcribe that voice message.")
        return
    finally:
        tmp_path.unlink(missing_ok=True)

    await update.message.reply_text(f'_Heard:_ "{text}"', parse_mode="Markdown")
    await _pipeline(update, context, text)


async def handle_auth_calendar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.bot_data["config"]
    if not config.google_credentials_file:
        await update.message.reply_text(
            "GOOGLE\\_CREDENTIALS\\_FILE ist nicht konfiguriert. Bitte credentials.json herunterladen und Pfad in .env setzen.",
            parse_mode="Markdown",
        )
        return

    try:
        auth_url = calendar_integration.get_auth_url(config.google_credentials_file)
    except Exception:
        logger.exception("Failed to generate Google auth URL")
        await update.message.reply_text("Fehler beim Generieren der Auth-URL. Bitte Logs prüfen.")
        return

    context.bot_data["awaiting_calendar_code"] = True
    await update.message.reply_text(
        f"Öffne diesen Link, autorisiere den Zugriff, und schicke mir den angezeigten Code zurück:\n\n{auth_url}"
    )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    db_path: str = context.bot_data["db_path"]
    action, reminder_id_str = query.data.split(":", 1)
    reminder_id = int(reminder_id_str)

    if action == "done":
        database.complete_todo(db_path, reminder_id)
        await query.edit_message_text(
            (query.message.text or "") + "\n\n✅ Done!"
        )
    # "notyet" → do nothing; message and buttons stay as-is


# ── Application lifecycle ─────────────────────────────────────────────────────

async def _post_init(app: Application) -> None:
    config: Config = app.bot_data["config"]

    database.init_db(config.db_path)
    app.bot_data["db_path"] = config.db_path
    app.bot_data["history_max_pairs"] = config.history_max_pairs
    app.bot_data["anthropic_client"] = AsyncAnthropic(api_key=config.anthropic_api_key)
    app.bot_data["openai_client"] = AsyncOpenAI(api_key=config.openai_api_key)

    sched = ReminderScheduler(app.bot, config.db_path)
    sched.set_config(config)
    sched.start()
    app.bot_data["scheduler"] = sched

    # Reschedule any reminders that survived a restart
    pending = database.get_due_reminders(config.db_path)
    now = datetime.now(timezone.utc)
    rescheduled = 0
    for r in pending:
        run_at = datetime.fromisoformat(r["remind_at"]).replace(tzinfo=timezone.utc)
        if run_at > now:
            sched.add_reminder(r["id"], r["chat_id"], r["title"], run_at)
            rescheduled += 1
    logger.info("Rescheduled %d pending reminder(s)", rescheduled)

    # Schedule daily to-do digest at 08:00 Europe/Berlin
    if config.telegram_chat_id:
        sched.add_daily_digest(config.telegram_chat_id, config.db_path)

    # Backfill embeddings for existing notes in the background
    asyncio.create_task(ai.backfill_note_embeddings(config.db_path, config.voyage_api_key))


async def _post_shutdown(app: Application) -> None:
    sched: ReminderScheduler | None = app.bot_data.get("scheduler")
    if sched:
        sched.stop()


def build_app(config: Config) -> Application:
    app = (
        ApplicationBuilder()
        .token(config.telegram_token)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )
    app.bot_data["config"] = config

    app.add_handler(CommandHandler("auth_calendar", handle_auth_calendar))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(CallbackQueryHandler(handle_callback))

    return app
