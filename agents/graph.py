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
from tools.config import get_env

_MAX_RETRIES = None


def _get_max_retries():
    global _MAX_RETRIES
    if _MAX_RETRIES is None:
        _MAX_RETRIES = int(get_env("MAX_RETRIES", "2"))
    return _MAX_RETRIES


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


def build_graph(df=None) -> StateGraph:
    """Construct and compile the multi-agent analysis graph.

    Parameters
    ----------
    df : pd.DataFrame, optional
        The DataFrame to make available to the Coder node.
        Passed via closure to bypass LangGraph state serialization.

    Returns a compiled graph with MemorySaver checkpointer.
    """
    workflow = StateGraph(AgentState)

    # Wrap coder_node so the DataFrame is injected via closure.
    # coder_node accepts df as a keyword arg; the @log_node decorator
    # forwards **kwargs so it reaches the original function.
    def _coder_with_df(state: dict[str, Any]) -> dict[str, Any]:
        return coder_node(state, df=df)

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
