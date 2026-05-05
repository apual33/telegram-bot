import logging

from telegram import Update

from config import Config
from bot import build_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)


def main() -> None:
    config = Config.from_env()
    app = build_app(config)
    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
