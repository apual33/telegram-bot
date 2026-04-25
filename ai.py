import asyncio
import json
import logging
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

from anthropic import AsyncAnthropic

import database
from config import Config
from scheduler import ReminderScheduler

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-5"
_EMBED_MODEL = "voyage-3"
_TZ = ZoneInfo("Europe/Berlin")

# web_search_20250305 is a server-side tool — Anthropic executes it internally.
# The other three are client-side tools that this code executes.
def _build_tools(now_iso: str, tomorrow_iso: str, tz_offset: str) -> list:
    """Return the tool list with a fresh example date/time for remind_at."""
    return [
    {
        "name": "create_todo",
        "description": (
            "Create a to-do item for the user. "
            "Optionally set remind_at to also schedule a timed reminder. "
            "Use for any task the user wants to remember, with or without a specific time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short description of the task.",
                },
                "remind_at": {
                    "type": "string",
                    "description": (
                        f"ISO 8601 datetime with UTC offset for an absolute reminder time. "
                        f"Use ONLY for clock-time requests like 'at 10:00' or 'tomorrow at 9'. "
                        f"Example: {tomorrow_iso}T10:00:00{tz_offset}. "
                        "Use pre-resolved dates from the system prompt for heute/morgen/übermorgen. "
                        "Do NOT use for relative expressions like 'in 5 minutes' — use remind_in_minutes instead."
                    ),
                },
                "remind_in_minutes": {
                    "type": "integer",
                    "description": (
                        "Use for relative reminders like 'in 5 minutes' or 'in 2 hours'. "
                        "Pass the total number of minutes from now. "
                        "Examples: 'in 5 minutes' → 5, 'in 2 hours' → 120, 'in 1.5 hours' → 90. "
                        "The server will compute the exact time. Do not use together with remind_at."
                    ),
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "save_note",
        "description": "Save any text the user wants to store: notes, thoughts, ideas, reflections, follow-ups, references, or tasks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The content to save."},
                "type": {
                    "type": "string",
                    "enum": ["note", "followup", "reference", "task"],
                    "description": "Category: 'note' for general thoughts/ideas/reflections, 'followup' for follow-ups, 'reference' for links/references, 'task' for actionable items.",
                },
            },
            "required": ["content", "type"],
        },
    },
    {
        "name": "search_notes",
        "description": "Search the user's saved notes, thoughts, tasks, and references.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "research",
        "description": (
            "Search the web and return a concise, structured answer. "
            "Use for factual lookups, lists (e.g. VCs in a city, restaurants in an area), "
            "or any question that benefits from current web data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query to run."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_todos",
        "description": "Return all open (not yet done) to-dos for the user.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "complete_todo",
        "description": (
            "Mark a to-do as done. Call list_todos first if you do not know the id. "
            "The result includes the 'title' of what was actually marked done — always "
            "confirm this exact title back to the user so they can catch any mistakes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "todo_id": {
                    "type": "integer",
                    "description": "The id of the to-do to mark as done.",
                },
            },
            "required": ["todo_id"],
        },
    },
    {
        "name": "snooze_todo",
        "description": (
            "Postpone a reminder by adding minutes to its scheduled time. "
            "Use when the user says 'verschieben', 'später', 'nicht jetzt', or similar after a reminder fires. "
            "The todo_id is visible in the conversation history as [todo_id:X] on the reminder message."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "todo_id": {
                    "type": "integer",
                    "description": "The id of the todo to snooze. Read it from [todo_id:X] in the reminder message.",
                },
                "minutes": {
                    "type": "integer",
                    "description": (
                        "Minutes to add to the current remind_at. "
                        "'um einen Tag' → 1440, 'morgen' → 1440, '2 Stunden' → 120, "
                        "'eine Stunde' → 60, '30 Minuten' → 30. "
                        "If the user just says 'verschieben' with no duration, use 1440 (one day)."
                    ),
                },
            },
            "required": ["todo_id", "minutes"],
        },
    },
    {
        "name": "send_email",
        "description": (
            "Send an email on behalf of the user via their configured Gmail account. "
            "Use this when the user asks to send, write, or draft+send an email."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address.",
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line.",
                },
                "body": {
                    "type": "string",
                    "description": "Plain-text email body.",
                },
            },
            "required": ["to", "subject", "body"],
        },
    },
]  # end of _build_tools return value

_CLIENT_TOOLS = {"create_todo", "list_todos", "complete_todo", "snooze_todo", "save_note", "search_notes", "research", "send_email"}

HAIKU_MODEL = "claude-haiku-4-5-20251001"


async def _embed(text: str, voyage_api_key: str, input_type: str = "document") -> list[float]:
    import voyageai
    vo = voyageai.Client(api_key=voyage_api_key)
    result = await asyncio.to_thread(vo.embed, [text], model=_EMBED_MODEL, input_type=input_type)
    return result.embeddings[0]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def backfill_note_embeddings(db_path: str, voyage_api_key: str) -> None:
    """Generate and store embeddings for all notes that don't have one yet."""
    if not voyage_api_key:
        logger.warning("VOYAGE_API_KEY not set — skipping embedding backfill")
        return
    notes = database.get_notes_without_embeddings(db_path)
    if not notes:
        return
    logger.info("Backfilling embeddings for %d note(s)…", len(notes))
    for note in notes:
        try:
            vec = await _embed(note["content"], voyage_api_key, input_type="document")
            database.update_note_embedding(db_path, note["id"], vec)
        except Exception:
            logger.exception("Failed to embed note id=%s during backfill", note["id"])
    logger.info("Embedding backfill complete")


def _fmt_berlin(dt: datetime) -> str:
    """Return a human-readable date/time string derived from the actual datetime value."""
    local = dt.astimezone(_TZ)
    return local.strftime("%A, %d. %B %Y, %H:%M")


_DE_WEEKDAYS = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
_DE_MONTHS = ["Januar", "Februar", "März", "April", "Mai", "Juni",
               "Juli", "August", "September", "Oktober", "November", "Dezember"]


def _system_prompt(report_email: str = "") -> str:
    now = datetime.now(_TZ)
    weekday_de = _DE_WEEKDAYS[now.weekday()]
    month_de = _DE_MONTHS[now.month - 1]
    today_de = f"{weekday_de}, {now.day}. {month_de} {now.year}"
    tz_offset = now.strftime("%z")  # e.g. +0200
    tz_offset_iso = tz_offset[:3] + ":" + tz_offset[3:]  # e.g. +02:00
    tomorrow = now + timedelta(days=1)
    tomorrow_iso = tomorrow.strftime("%Y-%m-%d")
    day_after_iso = (now + timedelta(days=2)).strftime("%Y-%m-%d")
    email_hint = (
        f" The user's own email address is {report_email} — use it as the default recipient"
        " unless they specify a different address."
        if report_email
        else ""
    )
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%S") + tz_offset_iso
    return (
        "CRITICAL RULE: You have tools available. You MUST call the appropriate tool whenever the user "
        "asks you to save, create, or schedule anything. NEVER describe what you would do — always call "
        "the tool immediately. Responding with plain text instead of a tool call when the user wants an "
        "action taken is a failure.\n\n"
        f"You are a personal assistant. Heute ist {today_de}, {now.strftime('%H:%M')} Uhr (UTC-Offset: {tz_offset_iso}).\n"
        f"Aktuelle Zeit (ISO 8601): {now_iso}\n"
        f"Vorberechnete Daten — heute: {now.strftime('%Y-%m-%d')}, morgen: {tomorrow_iso}, übermorgen: {day_after_iso}.\n"
        f"When creating reminders: for relative expressions ('in 5 minutes', 'in 2 hours') use "
        f"remind_in_minutes (e.g. 5 or 120) — the server computes the exact time. "
        f"For absolute times ('at 10:00', 'morgen um 9') use remind_at as a full ISO 8601 datetime "
        f"with UTC offset, e.g. {tomorrow_iso}T10:00:00{tz_offset_iso}.\n\n"
        "Tools and when to call them (mandatory — use the tool, do not just talk about it):\n"
        "- create_todo: call this for ANY request to remember, save, or be reminded of something. "
        "Pass remind_in_minutes for relative times ('in 5 minutes' → 5), remind_at for absolute times.\n"
        "- list_todos: call this ALWAYS when asked for todos/tasks — never answer from memory\n"
        "  - List EXACTLY the items in 'todos', state count as EXACTLY the 'count' field from the result\n"
        "- complete_todo: call when user says they finished a task; use list_todos first if you need the id. "
        "The result contains 'title' — always repeat this exact title in your confirmation ('Habe \\'X\\' als erledigt markiert ✓')\n"
        "- snooze_todo: call when user replies to a reminder with 'verschieben', 'später', 'nicht jetzt', "
        "'noch nicht', or similar. Read the todo_id from [todo_id:X] in the reminder message in history. "
        "When the user says this, first reply offering two options: 'Verschieben' and 'Löschen'. "
        "When they choose 'verschieben' (or specify a duration), call snooze_todo with the id and minutes. "
        "When they choose 'löschen', call complete_todo with the id.\n"
        "- save_note / search_notes: for short notes, references, follow-ups\n"
        "- save_note / search_notes: for notes, thoughts, ideas, reflections, references, follow-ups\n"
        "- research: for web lookups; return the result as-is\n"
        f"- send_email: for sending emails.{email_hint}\n\n"
        "When confirming a reminder, always use the 'scheduled_for_display' field from the tool result "
        "for the time string — never compute or name weekdays or times yourself.\n"
        "Always confirm actions you take. Be concise and friendly."
    )


async def run(
    chat_id: int,
    messages: list[dict],
    client: AsyncAnthropic,
    db_path: str,
    sched: ReminderScheduler,
    config: Config | None = None,
) -> str:
    """
    Run the agentic loop against the Anthropic API.

    `messages` is used as the working message list for this call; it is mutated
    in-place so that tool_use / tool_result rounds are preserved within a single
    request cycle. The caller owns history management across calls.
    """
    now = datetime.now(_TZ)
    tz_offset = now.strftime("%z")
    tz_offset_iso = tz_offset[:3] + ":" + tz_offset[3:]
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%S") + tz_offset_iso
    tomorrow_iso = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    tools = _build_tools(now_iso, tomorrow_iso, tz_offset_iso)

    while True:
        logger.info(
            "Sending API request | model=%s | tools=%d | messages=%d | tool_names=%s",
            MODEL, len(tools), len(messages), [t["name"] for t in tools],
        )
        response = await client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=_system_prompt(config.report_email if config else ""),
            tools=tools,
            messages=messages,
        )

        # Serialize all content blocks to plain dicts so they can be re-sent
        # to the API in the next loop iteration without Pydantic object issues.
        assistant_content = [block.model_dump() for block in response.content]
        messages.append({"role": "assistant", "content": assistant_content})

        text_parts = [b.text for b in response.content if b.type == "text"]

        logger.info(
            "API response | stop_reason=%s | blocks=%s",
            response.stop_reason,
            [{"type": b.type, "name": getattr(b, "name", None)} for b in response.content],
        )

        if response.stop_reason != "tool_use":
            # end_turn, or web search completed entirely server-side
            logger.info("Returning early: stop_reason=%s (no tool execution)", response.stop_reason)
            return "\n".join(text_parts) or "(no response)"

        # Only client-side tool_use blocks require action.
        # Server-side web search uses type "server_tool_use", not "tool_use".
        client_calls = [
            b for b in response.content
            if b.type == "tool_use" and b.name in _CLIENT_TOOLS
        ]

        all_tool_blocks = [b for b in response.content if b.type == "tool_use"]
        logger.info(
            "Tool blocks in response: total=%d client=%d names=%s",
            len(all_tool_blocks),
            len(client_calls),
            [b.name for b in all_tool_blocks],
        )

        if not client_calls:
            # stop_reason was "tool_use" but only server tools fired; return text
            logger.warning("stop_reason=tool_use but no client-side tool calls found — returning text without execution")
            return "\n".join(text_parts) or "(no response)"

        tool_results = []
        for call in client_calls:
            logger.info("Executing tool: %s | inputs=%s", call.name, call.input)
            result = await _execute(call.name, call.input, chat_id, db_path, sched, config, client)
            logger.info("Tool result: %s | result=%s", call.name, result)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": json.dumps(result),
            })
        messages.append({"role": "user", "content": tool_results})


async def _execute(
    name: str,
    inputs: dict,
    chat_id: int,
    db_path: str,
    sched: ReminderScheduler,
    config: Config | None = None,
    client: AsyncAnthropic | None = None,
) -> dict:
    logger.info("_execute called | tool=%s | chat_id=%d | db_path=%s | inputs=%s", name, chat_id, db_path, inputs)
    try:
        return await _execute_inner(name, inputs, chat_id, db_path, sched, config, client)
    except Exception:
        logger.exception("_execute FAILED | tool=%s | chat_id=%d | inputs=%s", name, chat_id, inputs)
        return {"ok": False, "error": "internal error — check server logs"}


async def _execute_inner(
    name: str,
    inputs: dict,
    chat_id: int,
    db_path: str,
    sched: ReminderScheduler,
    config: Config | None = None,
    client: AsyncAnthropic | None = None,
) -> dict:
    if name == "create_todo":
        remind_at = None
        if inputs.get("remind_in_minutes") is not None:
            remind_at = datetime.now(timezone.utc) + timedelta(minutes=int(inputs["remind_in_minutes"]))
            logger.info("create_todo: remind_in_minutes=%s → remind_at=%s", inputs["remind_in_minutes"], remind_at)
        elif inputs.get("remind_at"):
            remind_at = datetime.fromisoformat(inputs["remind_at"])
            logger.info("create_todo: remind_at (parsed)=%s", remind_at)
        logger.info("create_todo: calling database.create_todo | db=%s | chat_id=%d | title=%r | remind_at=%s", db_path, chat_id, inputs["title"], remind_at)
        tid = database.create_todo(db_path, chat_id, inputs["title"], remind_at)
        logger.info("create_todo: database.create_todo returned id=%s", tid)
        if remind_at:
            sched.add_reminder(tid, chat_id, inputs["title"], remind_at)
        result: dict = {"ok": True, "id": tid}
        if remind_at:
            result["scheduled_for_display"] = _fmt_berlin(remind_at)
        return result

    if name == "save_note":
        vec = None
        if config and config.voyage_api_key:
            try:
                vec = await _embed(inputs["content"], config.voyage_api_key, input_type="document")
            except Exception:
                logger.warning("Failed to generate embedding for note — saving without")
        nid = database.save_note(db_path, chat_id, inputs["content"], inputs["type"], embedding=vec)
        return {"ok": True, "id": nid}

    if name == "search_notes":
        query = inputs["query"]
        logger.info("search_notes | chat_id=%d | query=%r | voyage_key_set=%s", chat_id, query, bool(config and config.voyage_api_key))
        if config and config.voyage_api_key:
            try:
                query_vec = await _embed(query, config.voyage_api_key, input_type="query")
                logger.info("search_notes | query embedding generated | dims=%d", len(query_vec))
                notes = database.get_notes_with_embeddings(db_path, chat_id)
                logger.info("search_notes | notes with embeddings for chat_id=%d: %d", chat_id, len(notes))
                if notes:
                    scored = sorted(
                        notes,
                        key=lambda n: _cosine_similarity(query_vec, json.loads(n["embedding"])),
                        reverse=True,
                    )
                    top_scores = [(round(_cosine_similarity(query_vec, json.loads(n["embedding"])), 4), n["id"]) for n in scored[:5]]
                    logger.info("search_notes | top scores (score, id): %s", top_scores)
                    results = [
                        {k: v for k, v in n.items() if k != "embedding"}
                        for n in scored[:5]
                    ]
                    return {"results": results, "count": len(results)}
                else:
                    logger.warning("search_notes | no notes with embeddings found for chat_id=%d — falling back", chat_id)
            except Exception:
                logger.exception("search_notes | semantic search failed — falling back to text search")
        results = database.search_notes_text(db_path, chat_id, query)
        logger.info("search_notes | text search fallback returned %d result(s)", len(results))
        return {"results": results, "count": len(results)}

    if name == "list_todos":
        todos = database.list_open_todos(db_path, chat_id)
        for todo in todos:
            if todo.get("remind_at"):
                todo["scheduled_for_display"] = _fmt_berlin(
                    datetime.fromisoformat(todo["remind_at"]).replace(tzinfo=timezone.utc)
                )
        return {"todos": todos, "count": len(todos)}

    if name == "complete_todo":
        todo = database.complete_todo(db_path, inputs["todo_id"])
        if todo:
            logger.info("complete_todo | marked done | id=%d | title=%r", todo["id"], todo["title"])
            return {"ok": True, "id": todo["id"], "title": todo["title"]}
        else:
            logger.warning("complete_todo | not found or already done | id=%d", inputs["todo_id"])
            return {"ok": False, "error": f"Todo {inputs['todo_id']} not found or already done"}

    if name == "snooze_todo":
        result = database.snooze_todo(db_path, inputs["todo_id"], inputs["minutes"])
        if result is None:
            return {"ok": False, "error": "Todo not found or has no reminder set"}
        new_dt = datetime.fromisoformat(result["remind_at"]).replace(tzinfo=timezone.utc)
        sched.add_reminder(inputs["todo_id"], chat_id, result["title"], new_dt)
        return {"ok": True, "scheduled_for_display": _fmt_berlin(new_dt)}

    if name == "research":
        import httpx
        query = inputs["query"]
        if not config or not config.serper_api_key:
            return {"result": "Web search not configured (SERPER_API_KEY missing). Please inform the user."}
        try:
            async with httpx.AsyncClient() as http:
                r = await http.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": config.serper_api_key},
                    json={"q": query, "num": 8},
                    timeout=10,
                )
                r.raise_for_status()
                results = r.json().get("organic", [])
        except Exception as exc:
            logger.error("Serper search failed for query %r: %s", query, exc)
            return {"result": f"Web search failed: {exc}. Please inform the user and do not retry."}

        snippets = "\n\n".join(
            f"{r['title']}\n{r.get('snippet', '')}" for r in results if r.get("snippet")
        )
        if not snippets:
            logger.warning("Serper returned no usable results for query %r", query)
            return {"result": "Web search returned no results. Please inform the user and do not retry."}

        haiku_response = await client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Search results:\n{snippets}\n\n"
                    "Extract and list the relevant items concisely. "
                    "Use a simple bullet list. No preamble."
                ),
            }],
        )
        return {"result": haiku_response.content[0].text}

    if name == "send_email":
        if not config or not config.gmail_address or not config.gmail_app_password:
            return {"ok": False, "error": "Email not configured (GMAIL_ADDRESS / GMAIL_APP_PASSWORD missing)."}
        try:
            msg = MIMEMultipart()
            msg["From"] = config.gmail_address
            msg["To"] = inputs["to"]
            msg["Subject"] = inputs["subject"]
            msg.attach(MIMEText(inputs["body"], "plain"))
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as smtp:
                smtp.starttls()
                smtp.login(config.gmail_address, config.gmail_app_password)
                smtp.sendmail(config.gmail_address, inputs["to"], msg.as_string())
            return {"ok": True, "to": inputs["to"], "subject": inputs["subject"]}
        except Exception as exc:
            logger.exception("send_email tool failed")
            return {"ok": False, "error": str(exc)}

    return {"error": f"unknown tool: {name}"}
