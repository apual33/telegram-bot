import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

_DEFAULT_DB = str(Path(__file__).parent / "bot.db")


@dataclass
class Config:
    telegram_token: str
    anthropic_api_key: str
    openai_api_key: str
    db_path: str = _DEFAULT_DB
    history_max_pairs: int = 20  # user+assistant pairs to keep per chat
    gmail_address: str = ""
    gmail_app_password: str = ""
    report_email: str = ""
    telegram_chat_id: int = 0  # owner chat_id for proactive messages (daily digest)
    serper_api_key: str = ""
    voyage_api_key: str = ""

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        return cls(
            telegram_token=os.environ["TELEGRAM_BOT_TOKEN"],
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            db_path=os.getenv("DB_PATH", _DEFAULT_DB),
            gmail_address=os.getenv("GMAIL_ADDRESS", ""),
            gmail_app_password=os.getenv("GMAIL_APP_PASSWORD", ""),
            report_email=os.getenv("REPORT_EMAIL", ""),
            telegram_chat_id=int(os.getenv("TELEGRAM_CHAT_ID", "0")),
            serper_api_key=os.getenv("SERPER_API_KEY", ""),
            voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
        )
