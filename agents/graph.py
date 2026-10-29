"""
Graph — LangGraph StateGraph that wires all agent nodes together.

Flow:
  Planner → Coder (loops through sub-tasks) → Analyst → Critic
  Critic → END (PASS) | Coder (retry, max 2) | HumanInput (escalate)
  HumanInput → Planner (re-plan with clarification)

Uses MemorySaver checkpointer for pause/resume support.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.analyst import analyst_node
from agents.coder import coder_node
from agents.critic import critic_node
from agents.human_input import human_input_node
from agents.planner import planner_node
from agents.state import AgentState
from core.datasets import DatasetRegistry


# ── Routing Functions ───────────────────────────────────────────────


def route_after_coder(state: AgentState) -> str:
    """After Coder: continue to next sub-task or move to Analyst.

    If there are more sub-tasks, loop back to Coder.
    If all sub-tasks are done, proceed to Analyst.
    """
    sub_tasks = state.get("sub_tasks", [])
    current_idx = state.get("current_task_index", 0)

    if current_idx < len(sub_tasks):
        return "coder"  # More tasks to process
    return "analyst"  # All tasks done


def route_after_critic(state: AgentState) -> str:
    """After Critic: end, retry, or escalate.

    - PASS (verified_insight is set) → END
    - FAIL + retries remaining → Coder (retry)
    - FAIL + no retries / ambiguous → HumanInput (escalate)
    """
    # If Critic set verified_insight, we're done
    if state.get("verified_insight"):
        return "end"

    # If Critic flagged for human input
    if state.get("needs_human_input"):
        return "human_input"

    # Otherwise it's a retry (critic_feedback was set)
    return "coder"


# ── Graph Construction ──────────────────────────────────────────────


def build_graph(
    df=None,
    *,
    registry: DatasetRegistry | None = None,
    dataset_id: str | None = None,
) -> StateGraph:
    """Construct and compile the multi-agent analysis graph.

    Parameters
    ----------
    df : pd.DataFrame, optional
        The DataFrame to make available to the Coder node.
        Passed via closure to bypass LangGraph state serialization.
    registry, dataset_id : optional
        A DatasetRegistry and the ID of the dataset being analysed (give
        both or neither). They let the Coder answer common sub-tasks with
        deterministic core tools instead of generated code. Without ``df``,
        the sandbox gets a copy of the registered dataset.

    Returns a compiled graph with MemorySaver checkpointer.
    """
    if (registry is None) != (dataset_id is None):
        raise ValueError("registry and dataset_id must be given together")
    if registry is not None:
        registry.get(dataset_id)  # fail fast on an invalid or unknown ID
        if df is None:
            df = registry.get_dataframe(dataset_id)

    workflow = StateGraph(AgentState)

    # Wrap coder_node so the DataFrame (and registry) are injected via
    # closure. coder_node accepts them as keyword args; the @log_node
    # decorator forwards **kwargs so they reach the original function.
    def _coder_with_df(state: dict[str, Any]) -> dict[str, Any]:
        return coder_node(state, df=df, registry=registry, dataset_id=dataset_id)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("coder", _coder_with_df)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("human_input", human_input_node)

    # Set entry point
    workflow.set_entry_point("planner")

    # Edges
    workflow.add_edge("planner", "coder")

    # Coder → next task or Analyst
    workflow.add_conditional_edges(
        "coder",
        route_after_coder,
        {
            "coder": "coder",
            "analyst": "analyst",
        },
    )

    # Analyst → Critic (always)
    workflow.add_edge("analyst", "critic")

    # Critic → END / Coder (retry) / HumanInput (escalate)
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "end": END,
            "coder": "coder",
            "human_input": "human_input",
        },
    )

    # HumanInput → Planner (re-plan with clarification)
    workflow.add_edge("human_input", "planner")

    # Compile with checkpointer for pause/resume
    checkpointer = MemorySaver()
    compiled = workflow.compile(checkpointer=checkpointer)

    return compiled


def create_initial_state(
    user_query: str,
    dataframe_profile: str,
) -> dict[str, Any]:
    """Create the initial state dict for a new graph invocation.

    Parameters
    ----------
    user_query : str
        The user's natural-language question.
    dataframe_profile : str
        Compact text profile of the uploaded dataset.

    Returns
    -------
    dict — ready to pass to ``graph.invoke(state, config)``.
    """
    return {
        "user_query": user_query,
        "dataframe_profile": dataframe_profile,
        "sub_tasks": [],
        "current_task_index": 0,
        "code_outputs": [],
        "draft_insight": "",
        "verified_insight": "",
        "critic_feedback": "",
        "retry_count": 0,
        "needs_human_input": False,
        "human_question": "",
        "human_answer": "",
        "messages": [],
        "trace_log": [],
    }
