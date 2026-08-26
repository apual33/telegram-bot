import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx
from anthropic import AsyncAnthropic
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Bot

import calendar_integration
import database
from config import Config

_TZ = ZoneInfo("Europe/Berlin")
logger = logging.getLogger(__name__)

# ── Weather (Open-Meteo, no API key required) ─────────────────────────────────
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
_LEIPZIG_LAT = 51.3397
_LEIPZIG_LON = 12.3731

# WMO weather codes → German description
_WEATHER_CODES = {
    0: "klar",
    1: "überwiegend klar",
    2: "teils bewölkt",
    3: "bedeckt",
    45: "neblig",
    48: "Nebel mit Reifbildung",
    51: "leichter Nieselregen",
    53: "Nieselregen",
    55: "starker Nieselregen",
    56: "gefrierender Nieselregen",
    57: "starker gefrierender Nieselregen",
    61: "leichter Regen",
    63: "Regen",
    65: "starker Regen",
    66: "gefrierender Regen",
    67: "starker gefrierender Regen",
    71: "leichter Schneefall",
    73: "Schneefall",
    75: "starker Schneefall",
    77: "Schneegriesel",
    80: "leichte Schauer",
    81: "Schauer",
    82: "starke Schauer",
    85: "leichte Schneeschauer",
    86: "starke Schneeschauer",
    95: "Gewitter",
    96: "Gewitter mit leichtem Hagel",
    99: "Gewitter mit starkem Hagel",
}


def _weather_description(code: int) -> str:
    return _WEATHER_CODES.get(code, "wechselhaft")


def _weather_emoji(code: int) -> str:
    if code in (0, 1):
        return "☀️"
    if code == 2:
        return "🌤"
    if code == 3:
        return "☁️"
    if code in (45, 48):
        return "🌫"
    if code in (71, 73, 75, 77, 85, 86):
        return "🌨"
    if code in (80, 81, 82):
        return "🌦"
    if code in (95, 96, 99):
        return "⛈"
    return "🌧"


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

        # ── Weather ───────────────────────────────────────────────────────────
        weather = await self._generate_weather_summary()
        if weather:
            parts.append(weather)

        digest_text = "\n\n".join(parts)
        await self._bot.send_message(
            chat_id=chat_id,
            text=digest_text,
        )

        # Save to history so the AI can respond to follow-ups
        database.append_history(db_path, chat_id, "assistant", digest_text)

    async def _fetch_weather(self) -> dict | None:
        """Fetch today's Leipzig forecast from Open-Meteo (no API key needed)."""
        try:
            async with httpx.AsyncClient() as http:
                r = await http.get(
                    _WEATHER_URL,
                    params={
                        "latitude": _LEIPZIG_LAT,
                        "longitude": _LEIPZIG_LON,
                        "current": "temperature_2m,weather_code",
                        "hourly": "weather_code,precipitation_probability,precipitation",
                        "daily": "temperature_2m_max",
                        "timezone": "Europe/Berlin",
                        "forecast_days": 1,
                    },
                    timeout=10,
                )
                r.raise_for_status()
                return r.json()
        except Exception:
            logger.exception("Open-Meteo request failed")
            return None

    async def _generate_weather_summary(self) -> str | None:
        """One-line German weather summary for Leipzig."""
        data = await self._fetch_weather()
        if not data:
            return None

        try:
            current = data["current"]
            hourly = data["hourly"]
            temp_now = round(current["temperature_2m"])
            temp_max = round(data["daily"]["temperature_2m_max"][0])

            # Only the rest of the day is interesting — the digest fires in the morning.
            now_hour = datetime.now(_TZ).hour
            slots: list[tuple[int, int, int, float]] = []
            for i, stamp in enumerate(hourly["time"]):
                hour = int(stamp[11:13])
                if now_hour <= hour <= 22:
                    slots.append(
                        (
                            hour,
                            int(hourly["weather_code"][i]),
                            int(hourly["precipitation_probability"][i] or 0),
                            float(hourly["precipitation"][i] or 0.0),
                        )
                    )
        except (KeyError, IndexError, TypeError, ValueError):
            logger.exception("Unexpected Open-Meteo response shape")
            return None

        if not slots:
            slots = [(now_hour, int(current.get("weather_code", 3)), 0, 0.0)]

        # Emoji reflects the most notable condition of the remaining day.
        emoji = _weather_emoji(max(code for _, code, _, _ in slots))
        temps = f"Jetzige Temperatur {temp_now} °C, maximal heute {temp_max} °C."

        description = await self._describe_weather(slots)
        if not description:
            dominant = _weather_description(max(code for _, code, _, _ in slots))
            description = f"Heute überwiegend {dominant}."

        return f"{emoji} {description} {temps}"

    async def _describe_weather(self, slots: list[tuple[int, int, int, float]]) -> str | None:
        """Turn the hourly forecast into one short natural German sentence."""
        if not getattr(self, "_anthropic", None):
            logger.warning("Anthropic client not set — using template weather text")
            return None

        forecast_block = "\n".join(
            f"{hour:02d}:00 — {_weather_description(code)}, "
            f"Regenwahrscheinlichkeit {prob} %, Niederschlag {precip} mm"
            for hour, code, prob, precip in slots
        )

        try:
            response = await self._anthropic.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=128,
                system=(
                    "Du fasst Wettervorhersagen zusammen. Schreibe EINEN kurzen, "
                    "natürlichen deutschen Satz über den Tagesverlauf in Leipzig. "
                    "Fasse den Tag in Abschnitten zusammen (vormittags, mittags, "
                    "nachmittags, abends) und nenne nur das, was wirklich auffällt. "
                    "Höchstens 20 Wörter. "
                    "Beispiel-Stil (nicht kopieren): 'Bis mittags bewölkt aber trocken, "
                    "nachmittags starker Regen erwartet.' "
                    "Keine Temperaturangaben, kein Emoji, kein Intro, keine Uhrzeiten "
                    "in Ziffern. Antworte nur mit dem Satz."
                ),
                messages=[{"role": "user", "content": forecast_block}],
            )
            return response.content[0].text.strip()
        except Exception:
            logger.exception("Failed to generate weather description")
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
