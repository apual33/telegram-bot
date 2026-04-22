import json
import logging
import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

from anthropic import AsyncAnthropic

import database
from config import Config
from scheduler import ReminderScheduler

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"
_TZ = ZoneInfo("Europe/Berlin")

# web_search_20250305 is a server-side tool — Anthropic executes it internally.
# The other three are client-side tools that this code executes.
def _build_tools(tomorrow_iso: str, tz_offset: str) -> list:
    """Return the tool list with a fresh example date for remind_at."""
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
                        f"ISO 8601 datetime with UTC offset for when to send a reminder. "
                        f"Example for tomorrow at 10:00: {tomorrow_iso}T10:00:00{tz_offset}. "
                        "Always include the UTC offset. "
                        "IMPORTANT: Use ONLY the pre-resolved dates from the system prompt "
                        "(heute/today, morgen/tomorrow, übermorgen). "
                        "Never compute or guess dates yourself. "
                        "Omit entirely if no reminder is needed."
                    ),
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "save_note",
        "description": "Save a note, task, follow-up, or reference for the user.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The content to save."},
                "type": {
                    "type": "string",
                    "enum": ["note", "followup", "reference", "task"],
                    "description": "Category for this entry.",
                },
            },
            "required": ["content", "type"],
        },
    },
    {
        "name": "search_notes",
        "description": "Search the user's saved notes, tasks, and references.",
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
        "description": "Mark a to-do as done. Call list_todos first if you do not know the id.",
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
        "name": "save_journal",
        "description": (
            "Save a journal entry — a thought, idea, observation, or reflection from the user. "
            "Use this for longer, reflective content. "
            "For short actionable items (tasks, follow-ups, references) use save_note instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The full text of the journal entry.",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "search_journal",
        "description": (
            "Search the user's journal entries using semantic similarity. "
            "Use when the user asks what they wrote about a topic, or asks for a summary of their thoughts on something. "
            "Optionally filter to a recent time window."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic or question to search for.",
                },
                "since_days": {
                    "type": "integer",
                    "description": (
                        "Only search entries from the last N days. "
                        "Use 7 for 'last week', 30 for 'last month'. Omit to search all entries."
                    ),
                },
            },
            "required": ["query"],
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

_CLIENT_TOOLS = {"create_todo", "list_todos", "complete_todo", "save_note", "search_notes", "research", "save_journal", "search_journal", "send_email"}

HAIKU_MODEL = "claude-haiku-4-5-20251001"


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
    return (
        f"You are a personal assistant. Heute ist {today_de} (UTC-Offset: {tz_offset_iso}).\n"
        f"Vorberechnete Daten — heute: {now.strftime('%Y-%m-%d')}, morgen: {tomorrow_iso}, übermorgen: {day_after_iso}.\n"
        f"When creating reminders, always resolve relative day references (morgen, tomorrow, etc.) "
        f"to the concrete date from the list above. remind_at must always be a full ISO 8601 "
        f"datetime with UTC offset, e.g. {tomorrow_iso}T10:00:00{tz_offset_iso}.\n\n"
        "You can:\n"
        "- Manage to-dos and reminders: use create_todo to add tasks (pass remind_at for timed reminders), "
        "list_todos to show open tasks, complete_todo to mark done\n"
        "  - If the user wants a reminder, pass remind_at to create_todo — no separate reminder tool needed\n"
        "  - ALWAYS call list_todos when the user asks for their todos or task list — never answer from memory or context\n"
        "  - When presenting list_todos results: list EXACTLY the items in 'todos' — no additions, no omissions. "
        "State the count as EXACTLY the 'count' field from the tool result. Never compute or guess the count yourself.\n"
        "  - If the user says they completed a task, call list_todos if needed to find the id, then call complete_todo\n"
        "  - IMPORTANT: when confirming a reminder or listing todos, always use the 'scheduled_for_display' "
        "field from the tool result for the date string. Never compute or name weekdays yourself — "
        "the weekday is derived server-side from the actual date.\n"
        "- Research topics: use the research tool for web lookups and return the result as-is\n"
        "- Save and retrieve notes: use save_note and search_notes\n"
        "- Keep a personal journal: use save_journal for thoughts, ideas, observations, and reflections\n"
        "  - Decide based on length and intent: short actionable items → save_note; longer thoughts/ideas/reflections → save_journal\n"
        "  - Use search_journal when the user asks what they wrote about a topic or wants a summary of their thoughts\n"
        "  - After saving a journal entry, confirm briefly: 'Notiz gespeichert ✓'\n"
        f"- Send emails: use send_email with recipient, subject, and body.{email_hint}\n"
        "- Hold a general conversation\n\n"
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
    tomorrow_iso = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    tools = _build_tools(tomorrow_iso, tz_offset_iso)

    while True:
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

        if response.stop_reason != "tool_use":
            # end_turn, or web search completed entirely server-side
            return "\n".join(text_parts) or "(no response)"

        # Only client-side tool_use blocks require action.
        # Server-side web search uses type "server_tool_use", not "tool_use".
        client_calls = [
            b for b in response.content
            if b.type == "tool_use" and b.name in _CLIENT_TOOLS
        ]

        if not client_calls:
            # stop_reason was "tool_use" but only server tools fired; return text
            return "\n".join(text_parts) or "(no response)"

        tool_results = []
        for call in client_calls:
            result = await _execute(call.name, call.input, chat_id, db_path, sched, config, client)
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
    logger.info("Tool call: %s | chat_id=%d | db=%s", name, chat_id, db_path)
    if name == "create_todo":
        remind_at = None
        if inputs.get("remind_at"):
            remind_at = datetime.fromisoformat(inputs["remind_at"])
        tid = database.create_todo(db_path, chat_id, inputs["title"], remind_at)
        if remind_at:
            sched.add_reminder(tid, chat_id, inputs["title"], remind_at)
        result: dict = {"ok": True, "id": tid}
        if remind_at:
            result["scheduled_for_display"] = _fmt_berlin(remind_at)
        return result

    if name == "save_note":
        nid = database.save_note(db_path, chat_id, inputs["content"], inputs["type"])
        return {"ok": True, "id": nid}

    if name == "search_notes":
        results = database.search_notes(db_path, chat_id, inputs["query"])
        return {"results": results, "count": len(results)}

    if name == "list_todos":
        todos = database.list_open_todos(db_path, chat_id)
        for todo in todos:
            if todo.get("remind_at"):
                todo["scheduled_for_display"] = _fmt_berlin(
                    datetime.fromisoformat(todo["remind_at"])
                )
        return {"todos": todos, "count": len(todos)}

    if name == "complete_todo":
        updated = database.complete_todo(db_path, inputs["todo_id"])
        return {"ok": updated}

    if name == "save_journal":
        import embeddings as emb_module
        vec = await emb_module.encode(inputs["content"], client)
        jid = database.save_journal_entry(db_path, chat_id, inputs["content"], vec)
        return {"ok": True, "id": jid}

    if name == "search_journal":
        import embeddings as emb_module
        since_days = inputs.get("since_days")
        entries = database.get_journal_entries(db_path, chat_id, since_days=since_days)
        if not entries:
            return {"results": [], "count": 0}
        results = await emb_module.top_k(inputs["query"], entries, client, k=5)
        return {"results": results, "count": len(results)}

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
