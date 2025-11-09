"""Core data models for Agent B."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class UIElement(BaseModel):
    text: Optional[str] = None
    role: Optional[str] = None
    selector: Optional[str] = None
    kind: Optional[str] = None  # "clickable" | "input" | "modal" | "primary" | "breadcrumb" | "overlay"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UISnapshot(BaseModel):
    url: str
    title: Optional[str] = None
    clickables: List[UIElement] = Field(default_factory=list)
    inputs: List[UIElement] = Field(default_factory=list)
    modals: List[UIElement] = Field(default_factory=list)
    primary_actions: List[UIElement] = Field(default_factory=list)
    overlays: List[UIElement] = Field(default_factory=list)
    breadcrumbs: List[UIElement] = Field(default_factory=list)
    detected_app: Optional[str] = None


class PlannerAction(BaseModel):
    action: Literal["goto", "click", "fill", "wait_for", "capture", "press", "done"]
    selector: Optional[str] = None
    url: Optional[str] = None
    value: Optional[str] = None
    capture_name: Optional[str] = None
    reason: Optional[str] = None
    expect: Optional[str] = None
    success_hint: Optional[str] = None


class AgentState(BaseModel):
    task: str
    app: Optional[str] = None
    history: List[PlannerAction] = Field(default_factory=list)
