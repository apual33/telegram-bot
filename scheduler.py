import asyncio
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
_AACHEN_LAT = 50.7753
_AACHEN_LON = 6.0839
_WEATHER_ATTEMPTS = 3
_WEATHER_BACKOFF = 2.0  # seconds, multiplied by the attempt number
# Shown instead of silently dropping the section when the forecast is unreachable.
_WEATHER_UNAVAILABLE = "🌡 Wetterdaten für Aachen sind gerade nicht abrufbar."

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


def _num(*candidates: object) -> float | None:
    """First usable number among the candidates, else None.

    Open-Meteo can return null for individual values while a model run is being
    refreshed — around 04:00 UTC, which is exactly when the 06:00 digest fires.
    """
    for value in candidates:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


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
        """Fetch today's Aachen forecast from Open-Meteo (no API key needed).

        Retries a few times: a single transient network hiccup on the host used
        to make the whole weather section vanish from the digest without a trace.
        """
        params = {
            "latitude": _AACHEN_LAT,
            "longitude": _AACHEN_LON,
            "current": "temperature_2m,weather_code",
            "hourly": "temperature_2m,weather_code,precipitation_probability,precipitation",
            "daily": "temperature_2m_max",
            "timezone": "Europe/Berlin",
            "forecast_days": 1,
        }
        timeout = httpx.Timeout(20.0, connect=10.0)
        last_error = "unknown"

        for attempt in range(1, _WEATHER_ATTEMPTS + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as http:
                    r = await http.get(_WEATHER_URL, params=params)
                if r.status_code != 200:
                    # Open-Meteo explains refusals (rate limit, bad params) in the body.
                    last_error = f"HTTP {r.status_code}: {r.text[:200]}"
                else:
                    data = r.json()
                    if data.get("error"):
                        last_error = f"API error: {data.get('reason')}"
                    else:
                        if attempt > 1:
                            logger.info("Open-Meteo succeeded on attempt %d", attempt)
                        return data
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            logger.warning(
                "Open-Meteo attempt %d/%d failed — %s", attempt, _WEATHER_ATTEMPTS, last_error
            )
            if attempt < _WEATHER_ATTEMPTS:
                await asyncio.sleep(_WEATHER_BACKOFF * attempt)

        logger.error(
            "Open-Meteo unreachable after %d attempts — last error: %s",
            _WEATHER_ATTEMPTS,
            last_error,
        )
        return None

    async def _generate_weather_summary(self) -> str:
        """One-line German weather summary for Aachen.

        Tolerates gaps: individual null values are skipped rather than
        discarding the whole forecast, and each temperature has a fallback
        source. Always returns a line, so a failure is visible in the digest
        instead of silently removing the section.
        """
        data = await self._fetch_weather()
        if not data:
            return _WEATHER_UNAVAILABLE

        current = data.get("current") or {}
        hourly = data.get("hourly") or {}
        daily = data.get("daily") or {}

        times = hourly.get("time") or []
        codes = hourly.get("weather_code") or []
        probs = hourly.get("precipitation_probability") or []
        precs = hourly.get("precipitation") or []
        hour_temps = hourly.get("temperature_2m") or []

        def at(seq: list, i: int) -> object:
            return seq[i] if 0 <= i < len(seq) else None

        # Only the rest of the day is interesting — the digest fires in the morning.
        now_hour = datetime.now(_TZ).hour
        slots: list[tuple[int, int, int, float]] = []
        gaps = 0
        for i, stamp in enumerate(times):
            try:
                hour = int(str(stamp)[11:13])
            except (TypeError, ValueError):
                gaps += 1
                continue
            if not now_hour <= hour <= 22:
                continue
            code = _num(at(codes, i))
            if code is None:
                gaps += 1
                continue
            slots.append(
                (
                    hour,
                    int(code),
                    int(_num(at(probs, i)) or 0),
                    _num(at(precs, i)) or 0.0,
                )
            )
        if gaps:
            logger.warning("Open-Meteo: skipped %d hourly entry/entries without data", gaps)

        current_code = _num(current.get("weather_code"))
        temp_now = _num(current.get("temperature_2m"), at(hour_temps, now_hour))
        temp_max = _num(
            at(daily.get("temperature_2m_max") or [], 0),
            max((t for t in hour_temps if isinstance(t, (int, float))), default=None),
        )

        if not slots and current_code is None and temp_now is None and temp_max is None:
            logger.error("Open-Meteo returned no usable values — keys: %s", sorted(data.keys()))
            return _WEATHER_UNAVAILABLE

        # Emoji reflects the most notable condition of the remaining day.
        notable = max((code for _, code, _, _ in slots), default=None)
        if notable is None:
            notable = int(current_code) if current_code is not None else 3
        emoji = _weather_emoji(notable)

        if temp_now is not None and temp_max is not None:
            temps = f"Jetzige Temperatur {round(temp_now)} °C, maximal heute {round(temp_max)} °C."
        elif temp_now is not None:
            temps = f"Jetzige Temperatur {round(temp_now)} °C."
        elif temp_max is not None:
            temps = f"Maximal heute {round(temp_max)} °C."
        else:
            temps = ""

        description = await self._describe_weather(slots) if slots else None
        if not description:
            description = f"Heute überwiegend {_weather_description(notable)}."

        return f"{emoji} {description} {temps}".strip()

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
                    "natürlichen deutschen Satz über den Tagesverlauf in Aachen. "
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
