"""Task graph orchestrator for Agent B."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional


logger = logging.getLogger(__name__)

TaskCallable = Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, Any]]], Dict[str, Any]]


@dataclass
class TaskNode:
    """A single node in the task DAG."""

    id: str
    label: str
    tool_name: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self, suffix: Optional[str] = None) -> TaskNode:
        node_id = f"{self.id}{suffix}" if suffix else self.id
        return TaskNode(
            id=node_id,
            label=self.label,
            tool_name=self.tool_name,
            inputs=dict(self.inputs),
            depends_on=[f"{dep}{suffix}" if suffix else dep for dep in self.depends_on],
            metadata=dict(self.metadata),
        )


class TaskGraph:
    """Directed acyclic graph of tasks."""

    def __init__(self) -> None:
        self.nodes: Dict[str, TaskNode] = {}

    def __len__(self) -> int:
        return len(self.nodes)

    def add_node(self, node: TaskNode) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Duplicate node id {node.id}")
        if node.id in node.depends_on:
            raise ValueError(f"Node {node.id} cannot depend on itself")
        self.nodes[node.id] = node

    def get_node(self, node_id: str) -> TaskNode:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"Node {node_id} not found in graph") from exc

    def merge(self, other: TaskGraph, *, suffix: Optional[str] = None) -> None:
        for node in other.topological_order():
            self.add_node(node.copy(suffix))

    def copy(self) -> TaskGraph:
        graph = TaskGraph()
        for node in self.nodes.values():
            graph.add_node(node.copy())
        return graph

    def topological_order(self) -> List[TaskNode]:
        indegree: Dict[str, int] = {node_id: 0 for node_id in self.nodes}
        children: Dict[str, List[str]] = defaultdict(list)

        for node in self.nodes.values():
            for dep in node.depends_on:
                if dep not in self.nodes:
                    raise ValueError(f"Dependency {dep} missing for node {node.id}")
                indegree[node.id] += 1
                children[dep].append(node.id)

        queue: deque[str] = deque(node_id for node_id, degree in indegree.items() if degree == 0)
        order: List[TaskNode] = []

        while queue:
            current = queue.popleft()
            order.append(self.nodes[current])
            for child in children[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.nodes):
            raise ValueError("Graph has a cycle")
        return order


class ToolRegistry:
    """Mapping of tool names to callables."""

    def __init__(self) -> None:
        self.tools: Dict[str, TaskCallable] = {}

    def register(self, name: str, func: TaskCallable) -> None:
        if name in self.tools:
            raise ValueError(f"Tool {name} already registered")
        self.tools[name] = func

    def get(self, name: str) -> TaskCallable:
        if name not in self.tools:
            raise KeyError(f"Tool {name} not found")
        return self.tools[name]


class TaskStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


@dataclass
class TaskOutcome:
    """Execution result for a single node."""

    node: TaskNode
    status: TaskStatus
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    attempts: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "id": self.node.id,
            "label": self.node.label,
            "tool": self.node.tool_name,
            "status": self.status.value,
            "attempts": self.attempts,
        }
        if self.output is not None:
            payload["output"] = self.output
        if self.error:
            payload["error"] = self.error
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass
class FailureResolution:
    """Directive returned by the failure handler."""

    action: str
    reason: Optional[str] = None
    graph: Optional["TaskGraph"] = None
    max_retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def retry(cls, *, reason: Optional[str] = None, max_retries: int = 1, metadata: Optional[Dict[str, Any]] = None) -> "FailureResolution":
        return cls(action="retry", reason=reason, max_retries=max_retries, metadata=metadata or {})

    @classmethod
    def skip(cls, *, reason: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> "FailureResolution":
        return cls(action="skip", reason=reason, metadata=metadata or {})

    @classmethod
    def abort(cls, *, reason: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> "FailureResolution":
        return cls(action="abort", reason=reason, metadata=metadata or {})

    @classmethod
    def replan(cls, graph: "TaskGraph", *, reason: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> "FailureResolution":
        return cls(action="replan", graph=graph, reason=reason, metadata=metadata or {})


FailureHandler = Callable[[TaskNode, Exception, Dict[str, Any], Dict[str, TaskOutcome]], Optional[FailureResolution]]


@dataclass
class GraphExecutionResult:
    """Summary of a graph execution run."""

    graph: TaskGraph
    outcomes: Dict[str, TaskOutcome] = field(default_factory=dict)
    outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    failed: List[str] = field(default_factory=list)
    aborted: bool = False
    next_graph: Optional[TaskGraph] = None
    replans: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return not self.failed and not self.aborted


class GraphExecutor:
    """Executes a task graph using registered tools."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(
        self,
        graph: TaskGraph,
        state: Dict[str, Any],
        *,
        continue_on_error: bool = True,
        failure_handler: Optional[FailureHandler] = None,
    ) -> GraphExecutionResult:
        outputs: Dict[str, Dict[str, Any]] = {}
        outcomes: Dict[str, TaskOutcome] = {}
        failed_nodes: List[str] = []
        aborted = False
        next_graph: Optional[TaskGraph] = None

        order = graph.topological_order()

        for node in order:
            if node.id in outcomes:
                continue

            if aborted or next_graph:
                outcomes[node.id] = TaskOutcome(node=node, status=TaskStatus.SKIPPED, error="execution stopped early", attempts=0)
                continue

            blocked = False
            for dep in node.depends_on:
                dep_outcome = outcomes.get(dep)
                if dep_outcome and dep_outcome.status is not TaskStatus.SUCCESS:
                    blocked = True
                    break
            if blocked:
                outcomes[node.id] = TaskOutcome(node=node, status=TaskStatus.BLOCKED, error="blocked by dependency failure", attempts=0)
                continue

            tool = self.registry.get(node.tool_name)
            inputs = dict(node.inputs)
            attempts = 0
            attempt_cap = 1

            while True:
                attempts += 1
                try:
                    output = tool(state, inputs, outputs)
                    if output is not None:
                        outputs[node.id] = output
                    outcomes[node.id] = TaskOutcome(node=node, status=TaskStatus.SUCCESS, output=output, attempts=attempts)
                    break
                except Exception as exc:
                    resolution = failure_handler(node, exc, state, outcomes) if failure_handler else None
                    if resolution and resolution.action == "retry":
                        attempt_cap = max(attempt_cap, 1 + max(resolution.max_retries, 0))
                        if attempts < attempt_cap:
                            logger.debug("Retrying node %s (attempt %s/%s)", node.id, attempts + 1, attempt_cap)
                            continue

                    message = resolution.reason if resolution and resolution.reason else str(exc)
                    outcomes[node.id] = TaskOutcome(
                        node=node,
                        status=TaskStatus.FAILED,
                        error=message,
                        attempts=attempts,
                        metadata=resolution.metadata if resolution else {},
                    )
                    failed_nodes.append(node.id)
                    logger.warning("Task %s failed: %s", node.id, message)

                    if resolution and resolution.action == "replan" and resolution.graph:
                        next_graph = resolution.graph
                    elif resolution and resolution.action == "abort":
                        aborted = True
                    elif not continue_on_error:
                        aborted = True
                    break

            if (aborted or next_graph) and continue_on_error:
                continue
            if aborted or next_graph:
                break

        if aborted or next_graph:
            for node in order:
                if node.id in outcomes:
                    continue
                reason = "aborted" if aborted else "replan requested"
                outcomes[node.id] = TaskOutcome(node=node, status=TaskStatus.SKIPPED, error=reason, attempts=0)

        return GraphExecutionResult(
            graph=graph,
            outcomes=outcomes,
            outputs=outputs,
            failed=failed_nodes,
            aborted=aborted,
            next_graph=next_graph,
        )


@dataclass
class TaskGoal:
    """Normalized goal information passed to task planners."""

    query: str
    app_hint: Optional[str] = None
    affordances: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_context(cls, context: Mapping[str, Any]) -> "TaskGoal":
        query = str(context.get("query") or context.get("task") or "").strip()
        app = context.get("app") or context.get("app_hint")
        affordances = dict(context.get("affordances") or {})
        return cls(query=query, app_hint=app, affordances=affordances, context=dict(context))


class BaseTaskPlanner(ABC):
    """Base interface for planners that generate DAGs."""

    name: str = "planner"

    @abstractmethod
    def can_plan(self, goal: TaskGoal) -> bool:
        raise NotImplementedError

    @abstractmethod
    def plan(self, goal: TaskGoal) -> TaskGraph:
        raise NotImplementedError

    def on_failure(
        self,
        goal: TaskGoal,
        *,
        failing_node: TaskNode,
        error: Exception,
        outcomes: Dict[str, TaskOutcome],
        state: Dict[str, Any],
    ) -> Optional[FailureResolution]:
        return None


@dataclass
class PlannerResult:
    """Wrapper for a planned graph."""

    graph: TaskGraph
    goal: TaskGoal
    planner: Optional[BaseTaskPlanner] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BuilderRegistration:
    builder: Callable[[Dict[str, Any]], TaskGraph]
    predicate: Optional[Callable[[TaskGoal], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    """Builds task graphs based on user requests."""

    def __init__(self) -> None:
        self._planners: List[BaseTaskPlanner] = []
        self._builders: Dict[str, BuilderRegistration] = {}

    def register_planner(self, planner: BaseTaskPlanner) -> None:
        self._planners.append(planner)

    def register_builder(
        self,
        key: str,
        builder: Callable[[Dict[str, Any]], TaskGraph],
        *,
        predicate: Optional[Callable[[TaskGoal], bool]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._builders[key] = BuilderRegistration(builder=builder, predicate=predicate, metadata=metadata or {})

    def build(self, context: Dict[str, Any]) -> PlannerResult:
        goal = TaskGoal.from_context(context)

        for planner in self._planners:
            try:
                if planner.can_plan(goal):
                    graph = planner.plan(goal)
                    if graph:
                        logger.debug("Planner %s produced graph with %s nodes", planner.name, len(graph))
                        return PlannerResult(graph=graph, goal=goal, planner=planner)
            except Exception as exc:
                logger.warning("Planner %s failed to build graph: %s", getattr(planner, "name", planner), exc)

        for key, registration in self._builders.items():
            predicate = registration.predicate
            try:
                if predicate is None or predicate(goal):
                    graph = registration.builder(dict(context))
                    if graph:
                        logger.debug("Builder %s produced graph with %s nodes", key, len(graph))
                        return PlannerResult(graph=graph, goal=goal, metadata={"builder": key, **registration.metadata})
            except Exception as exc:
                logger.warning("Builder %s failed to build graph: %s", key, exc)

        raise ValueError("No planner or builder could produce a graph for the provided context")

    def run(
        self,
        context: Dict[str, Any],
        registry: ToolRegistry,
        state: Dict[str, Any],
        *,
        executor: Optional[GraphExecutor] = None,
        max_replans: int = 1,
        continue_on_error: bool = True,
    ) -> GraphExecutionResult:
        plan = self.build(context)
        exec_instance = executor or GraphExecutor(registry)

        aggregated_outcomes: Dict[str, TaskOutcome] = {}
        aggregated_outputs: Dict[str, Dict[str, Any]] = {}
        aggregated_failed: List[str] = []
        replans = 0
        metadata: Dict[str, Any] = dict(plan.metadata)

        current_plan = plan
        last_result: Optional[GraphExecutionResult] = None

        while True:
            def _handle_failure(
                node: TaskNode,
                error: Exception,
                _state: Dict[str, Any],
                outcomes: Dict[str, TaskOutcome],
            ) -> Optional[FailureResolution]:
                if current_plan.planner:
                    try:
                        resolution = current_plan.planner.on_failure(
                            current_plan.goal,
                            failing_node=node,
                            error=error,
                            outcomes=outcomes,
                            state=state,
                        )
                        if resolution:
                            return resolution
                    except Exception as exc:
                        logger.warning("Planner %s.on_failure raised %s", current_plan.planner.name, exc)
                return None

            result = exec_instance.execute(
                current_plan.graph,
                state,
                continue_on_error=continue_on_error,
                failure_handler=_handle_failure,
            )

            aggregated_outcomes.update(result.outcomes)
            aggregated_outputs.update(result.outputs)
            for node_id in result.failed:
                if node_id not in aggregated_failed:
                    aggregated_failed.append(node_id)
            last_result = result

            if result.next_graph and replans < max_replans:
                replans += 1
                current_plan = PlannerResult(
                    graph=result.next_graph,
                    goal=plan.goal,
                    planner=plan.planner,
                    metadata={"replan": replans, **plan.metadata},
                )
                continue
            break

        if last_result is None:
            raise RuntimeError("Graph execution did not run")

        return GraphExecutionResult(
            graph=last_result.graph,
            outcomes=aggregated_outcomes,
            outputs=aggregated_outputs,
            failed=aggregated_failed,
            aborted=last_result.aborted,
            next_graph=last_result.next_graph,
            replans=replans,
            metadata=metadata,
        )


_GLOBAL_ORCHESTRATOR = Orchestrator()


def get_orchestrator() -> Orchestrator:
    return _GLOBAL_ORCHESTRATOR


def build_graph(context: Dict[str, Any]) -> TaskGraph:
    """Convenience hook for CLI usage."""
    plan = _GLOBAL_ORCHESTRATOR.build(context)
    return plan.graph


def run_graph(
    graph: TaskGraph,
    registry: ToolRegistry,
    state: Dict[str, Any],
    *,
    continue_on_error: bool = True,
    failure_handler: Optional[FailureHandler] = None,
) -> GraphExecutionResult:
    executor = GraphExecutor(registry)
    return executor.execute(graph, state, continue_on_error=continue_on_error, failure_handler=failure_handler)


__all__ = [
    "TaskCallable",
    "TaskNode",
    "TaskGraph",
    "TaskStatus",
    "TaskOutcome",
    "FailureResolution",
    "GraphExecutionResult",
    "ToolRegistry",
    "GraphExecutor",
    "TaskGoal",
    "BaseTaskPlanner",
    "PlannerResult",
    "Orchestrator",
    "build_graph",
    "get_orchestrator",
    "run_graph",
]
