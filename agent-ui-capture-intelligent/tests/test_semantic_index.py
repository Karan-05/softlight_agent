import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_b.semantic_index import SemanticIndex
from agent_b.models import PlannerAction


def test_semantic_index_retrieves_similar_plan():
    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "semantic.jsonl"
        index = SemanticIndex(index_path)
        actions = [
            PlannerAction(action="click", selector="button:has-text('Create')"),
            PlannerAction(action="wait_for", selector="text='Created'"),
        ]
        index.add("linear", "Create a project", actions)

        retrieved = index.retrieve("linear", "Create new project")
        assert retrieved is not None
        assert retrieved.actions[0].selector == "button:has-text('Create')"


def test_semantic_index_respects_app_scope():
    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "semantic.jsonl"
        index = SemanticIndex(index_path)
        actions = [PlannerAction(action="click", selector="button:has-text('Create')")]
        index.add("linear", "Create a project", actions)

        retrieved = index.retrieve("notion", "Create new project")
        assert retrieved is None
