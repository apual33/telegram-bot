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

# Stored between get_auth_url() and exchange_code() so the PKCE code verifier
# that was generated for the auth URL is reused when fetching the token.
_pending_flow: Flow | None = None


# ── OAuth helpers ──────────────────────────────────────────────────────────────

def get_auth_url(credentials_file: str, token_file: str) -> str:
    """Return the OAuth consent URL (PKCE/S256). Raises if credentials_file is missing."""
    global _pending_flow
    flow = Flow.from_client_secrets_file(credentials_file, scopes=_SCOPES)
    flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
    auth_url, _ = flow.authorization_url(
        prompt="consent",
        access_type="offline",
        autogenerate_code_verifier=True,
    )
    _pending_flow = flow
    return auth_url


def exchange_code(code: str, credentials_file: str, token_file: str) -> None:
    """Exchange an auth code for a token and persist it to token_file."""
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

def fetch_today_events(token_file: str, calendar_id: str) -> list[dict]:
    """
    Return today's events as a list of dicts with keys 'start' (HH:MM or 'ganztägig')
    and 'summary'. Raises RuntimeError if the token is missing/invalid.
    """
    creds = _load_credentials(token_file)
    if creds is None:
        raise RuntimeError("no_token")

    now_berlin = datetime.now(_TZ)
    day_start = now_berlin.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = now_berlin.replace(hour=23, minute=59, second=59, microsecond=0)

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


async def fetch_today_events_async(token_file: str, calendar_id: str) -> list[dict]:
    """Async wrapper — runs the blocking API call in a thread pool."""
    return await asyncio.get_event_loop().run_in_executor(
        None, fetch_today_events, token_file, calendar_id
    )


def format_events(events: list[dict]) -> str:
    if not events:
        return "📅 Heute keine Termine"
    lines = ["📅 *Termine heute:*"]
    for e in events:
        lines.append(f"• {e['start']} — {e['summary']}")
    return "\n".join(lines)
