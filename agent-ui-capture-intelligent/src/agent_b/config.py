"""Configuration for Agent B."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from a .env file when present.
load_dotenv()

DATASET_ROOT = Path("datasets")

PLAYWRIGHT_STORAGE = {
    "linear": Path("secrets/linear-storage.json"),
    "notion": Path("secrets/notion-storage.json"),
}

OPENAI_MODEL = "gpt-4o-mini"

APP_LAUNCH_URLS = {
    "linear": "https://linear.app/login",
    "notion": "https://www.notion.so/login",
}

USER_DATA_DIR = Path("profiles/default")

DEFAULT_BROWSER = os.getenv("AGENT_BROWSER", "firefox").lower()

PLAYWRIGHT_CHANNEL = os.getenv("PLAYWRIGHT_CHANNEL", "chrome")

_default_chrome_path = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

PLAYWRIGHT_EXECUTABLE = os.getenv("PLAYWRIGHT_EXECUTABLE")
if not PLAYWRIGHT_EXECUTABLE and _default_chrome_path.exists():
    PLAYWRIGHT_EXECUTABLE = str(_default_chrome_path)


def get_openai_api_key() -> str | None:
    """Return the OpenAI API key or None when it is not configured."""
    return os.getenv("OPENAI_API_KEY") or None
