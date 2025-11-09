import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_b.executor import _normalize_expect_selector


def test_normalize_expect_selector_rejects_sentence():
    assert _normalize_expect_selector("Identify any new UI elements related to project creation.") is None


def test_normalize_expect_selector_keeps_valid_selector():
    assert _normalize_expect_selector("text=Create project") == "text=Create project"
