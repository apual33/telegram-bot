"""Google Calendar integration — OAuth2 flow and event fetching."""
import asyncio
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
_TZ = ZoneInfo("Europe/Berlin")

# Held between get_auth_url() and exchange_code() so the same flow object
# (and its internal state) is reused for the token exchange.
_pending_flow: Flow | None = None


# ── OAuth — OOB (Desktop app) flow ────────────────────────────────────────────

def get_auth_url(credentials_file: str) -> str:
    """
    Build and return the Google consent URL using the OOB redirect.
    The same flow object is stored in _pending_flow for exchange_code() to reuse.
    No PKCE — OOB Desktop app credentials do not support it.
    """
    global _pending_flow
    flow = Flow.from_client_secrets_file(credentials_file, scopes=_SCOPES)
    flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
    auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
    _pending_flow = flow
    return auth_url


def exchange_code(code: str, token_file: str) -> None:
    """
    Exchange the code the user pasted from the Google consent page.
    Must be called after get_auth_url() — reuses the stored flow object.
    """
    global _pending_flow
    if _pending_flow is None:
        raise RuntimeError("No pending OAuth flow — call get_auth_url() first.")
    _pending_flow.fetch_token(code=code)
    _save_token(_pending_flow.credentials, token_file)
    logger.info("Google Calendar token saved to %s", token_file)
    _pending_flow = None


def _load_credentials(token_file: str) -> Credentials | None:
    """Load and refresh credentials from token_file. Returns None if unavailable."""
    import os
    if not os.path.exists(token_file):
        return None
    creds = Credentials.from_authorized_user_file(token_file, _SCOPES)
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_token(creds, token_file)
        except Exception:
            logger.warning("Failed to refresh Google Calendar token")
            return None
    return creds if creds.valid else None


def _save_token(creds: Credentials, token_file: str) -> None:
    with open(token_file, "w") as f:
        f.write(creds.to_json())


# ── Event fetching ─────────────────────────────────────────────────────────────

def fetch_today_events(token_file: str, calendar_id: str, date: str | None = None) -> list[dict]:
    """
    Return events for the given ISO date (YYYY-MM-DD, Europe/Berlin) as a list of
    dicts with keys 'start' (HH:MM or 'ganztägig') and 'summary'.
    Defaults to today if date is None. Raises RuntimeError("no_token") if not authorised.
    """
    from datetime import date as date_type
    creds = _load_credentials(token_file)
    if creds is None:
        raise RuntimeError("no_token")

    if date:
        day = datetime.fromisoformat(date).replace(tzinfo=_TZ)
    else:
        day = datetime.now(_TZ)
    day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day.replace(hour=23, minute=59, second=59, microsecond=0)

    time_min = day_start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    time_max = day_end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    try:
        service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        result = (
            service.events()
            .list(
                calendarId=calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
    except HttpError as exc:
        logger.error("Google Calendar API error: %s", exc)
        raise

    events = []
    for item in result.get("items", []):
        start = item["start"]
        if "dateTime" in start:
            dt = datetime.fromisoformat(start["dateTime"]).astimezone(_TZ)
            start_str = dt.strftime("%H:%M")
        else:
            start_str = "ganztägig"
        events.append({"start": start_str, "summary": item.get("summary", "(kein Titel)")})

    return events


async def fetch_today_events_async(token_file: str, calendar_id: str, date: str | None = None) -> list[dict]:
    """Async wrapper — runs the blocking API call in a thread pool."""
    return await asyncio.get_event_loop().run_in_executor(
        None, fetch_today_events, token_file, calendar_id, date
    )


def format_events(events: list[dict], date_label: str = "heute") -> str:
    if not events:
        return f"📅 {date_label.capitalize()} keine Termine"
    lines = [f"📅 *Termine {date_label}:*"]
    for e in events:
        lines.append(f"• {e['start']} — {e['summary']}")
    return "\n".join(lines)
