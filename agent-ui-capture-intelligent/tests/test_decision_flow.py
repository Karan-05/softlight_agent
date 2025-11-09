from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_b.heuristic_planner import HeuristicPlanner  # noqa: E402
from agent_b.hierarchical_planner import HierarchicalPlanner  # noqa: E402
from agent_b.models import PlannerAction, UIElement, UISnapshot  # noqa: E402


def _snapshot_with_clickables(elements: list[UIElement]) -> UISnapshot:
    return UISnapshot(url="https://example.test", clickables=elements)


def test_heuristic_prefers_modal_actions() -> None:
    planner = HeuristicPlanner()
    modal_clickable = UIElement(
        text="Create",
        selector="text=Create",
        role="button",
        kind="clickable",
        metadata={"ownerDialog": {"label": "Create item"}},
    )
    page_button = UIElement(
        text="Create",
        selector="text=Create (page)",
        role="button",
        kind="clickable",
    )
    snapshot = UISnapshot(
        url="https://example.test",
        clickables=[modal_clickable, page_button],
        modals=[UIElement(text="Create item", selector="role=dialog[name=/Create/]", kind="modal")],
    )

    action = planner.plan_next("Create a new record", snapshot, [], avoid_selectors=None)

    assert action.action == "click"
    assert action.selector == modal_clickable.selector


@pytest.mark.asyncio
async def test_hierarchical_defaults_to_heuristic_without_llm() -> None:
    heuristic = HeuristicPlanner()
    planner = HierarchicalPlanner(llm_planner=None, heuristic_planner=heuristic)
    snapshot = _snapshot_with_clickables(
        [
            UIElement(text="Add", selector="text=Add", role="button", kind="clickable"),
        ]
    )

    action, source = await planner.plan_next("Add a new entry", snapshot, [])

    assert action.action == "click"
    assert action.selector == "text=Add"
    assert source == "heuristic"


@pytest.mark.asyncio
async def test_hierarchical_replays_primed_actions_before_planning() -> None:
    heuristic = HeuristicPlanner()
    planner = HierarchicalPlanner(llm_planner=None, heuristic_planner=heuristic)
    primed = PlannerAction(action="click", selector="text=Next", reason="Replayed step")
    planner.prime_actions([primed], source="memory")

    snapshot = _snapshot_with_clickables(
        [
            UIElement(text="Next", selector="text=Next", role="button", kind="clickable"),
            UIElement(text="Continue", selector="text=Continue", role="button", kind="clickable"),
        ]
    )

    action, source = await planner.plan_next("Continue the flow", snapshot, [])

    assert action.selector == "text=Next"
    assert source == "memory"
