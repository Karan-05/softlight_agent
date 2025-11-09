import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_b.experience_store import ExperienceStore
from agent_b.models import PlannerAction


def test_experience_store_add_and_retrieve():
    with tempfile.TemporaryDirectory() as tmp:
        store_path = Path(tmp) / "experiences.jsonl"
        store = ExperienceStore(store_path)

        actions = [
            PlannerAction(action="click", selector="button:has-text('Create')"),
            PlannerAction(action="wait_for", selector="text='Created'"),
            PlannerAction(action="done"),
        ]
        store.add("linear", "Create a project", actions)

        retrieved = store.retrieve("linear", "Create new project")
        assert retrieved is not None
        assert retrieved.actions[0].selector == "button:has-text('Create')"


def test_experience_store_skips_capture_only_sequences():
    with tempfile.TemporaryDirectory() as tmp:
        store_path = Path(tmp) / "experiences.jsonl"
        store = ExperienceStore(store_path)

        actions = [PlannerAction(action="capture", capture_name="ambiguous")]
        store.add("linear", "Capture only", actions)

        assert not store_path.exists() or store_path.read_text().strip() == ""
