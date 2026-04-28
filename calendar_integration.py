"""Google Calendar integration — OAuth2 flow and event fetching."""
import asyncio
import logging
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
_TZ = ZoneInfo("Europe/Berlin")
_CALLBACK_PORT = 8080
_CALLBACK_TIMEOUT = 300  # seconds to wait for the user to complete the browser flow


# ── OAuth — local callback server ──────────────────────────────────────────────

def start_auth_flow(credentials_file: str, token_file: str) -> tuple[str, "asyncio.Future[None]"]:
    """
    Start a temporary HTTP server on localhost:8080 and return (auth_url, done_future).

    The caller should:
      1. Send auth_url to the user immediately.
      2. Await done_future (with a timeout) to know when the token has been saved.

    The server shuts itself down after the first callback.
    Raises TimeoutError (from the caller's await) if no callback arrives in time.
    """
    flow = Flow.from_client_secrets_file(credentials_file, scopes=_SCOPES)
    flow.redirect_uri = f"http://localhost:{_CALLBACK_PORT}"

    auth_url, _ = flow.authorization_url(
        prompt="consent",
        access_type="offline",
        autogenerate_code_verifier=True,
    )

    loop = asyncio.get_event_loop()
    # Resolved with None on success, set_exception on error
    done_future: asyncio.Future[None] = loop.create_future()

    class _CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            qs = parse_qs(urlparse(self.path).query)
            code = qs.get("code", [None])[0]
            error = qs.get("error", [None])[0]

            if code:
                status, body = 200, b"Authorized! You can close this tab."
            else:
                status = 400
                body = f"Auth error: {error or 'no_code'}. You can close this tab.".encode()

            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

            # Exchange the code and save the token, then signal the future
            def _finish() -> None:
                if done_future.done():
                    return
                try:
                    if not code:
                        raise RuntimeError(error or "no_code")
                    flow.fetch_token(code=code)
                    _save_token(flow.credentials, token_file)
                    logger.info("Google Calendar token saved to %s", token_file)
                    done_future.set_result(None)
                except Exception as exc:
                    done_future.set_exception(exc)

            loop.call_soon_threadsafe(_finish)
            # Ask the server to stop after this request
            threading.Thread(target=server.shutdown, daemon=True).start()

        def log_message(self, fmt: str, *args: object) -> None:
            pass

    server = HTTPServer(("localhost", _CALLBACK_PORT), _CallbackHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("OAuth callback server listening on port %d", _CALLBACK_PORT)

    return auth_url, done_future


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
