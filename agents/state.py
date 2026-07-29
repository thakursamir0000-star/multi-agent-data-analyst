"""
AgentState — shared TypedDict that flows through every node in the graph.

This is the single source of truth for the multi-agent pipeline.
Every node reads from and writes to this state object.
"""

from __future__ import annotations

from typing import Any, TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def _append_list(existing: list, new: list) -> list:
    """Reducer: concatenate lists instead of overwriting."""
    return existing + new


class AgentState(TypedDict):
    """Shared state for the multi-agent data analyst graph.

    Fields
    ------
    user_query : str
        The original natural-language question from the user.
    dataframe_profile : str
        Compact text summary of the uploaded dataset (from DataTool.profile_to_text).
    sub_tasks : list[str]
        Ordered list of concrete sub-tasks produced by the Planner.
    current_task_index : int
        Index into sub_tasks indicating which task the Coder should work on.
    code_outputs : list[dict]
        Accumulated results from the Coder's sandbox executions.
        Each dict: {task, code, stdout, plot_base64, error}.
    draft_insight : str
        The Analyst's first-pass narrative summary.
    verified_insight : str
        The Critic-approved final insight (empty until Critic passes).
    critic_feedback : str
        Specific feedback from the Critic when a retry is needed.
    retry_count : int
        Number of Critic → Coder retries (capped at MAX_RETRIES).
    needs_human_input : bool
        Flag set by the Critic to escalate to the HumanInput node.
    human_question : str
        Clarifying question surfaced to the user.
    human_answer : str
        The user's reply to the clarifying question.
    messages : list[BaseMessage]
        LangChain message history (uses add_messages reducer).
    trace_log : list[dict]
        Per-node observability records appended by the @log_node decorator.
    """

    user_query: str
    dataframe_profile: str

    # Plan
    sub_tasks: list[str]
    current_task_index: int

    # Execution
    code_outputs: Annotated[list[dict], _append_list]

    # Analysis
    draft_insight: str
    verified_insight: str
    critic_feedback: str

    # Control flow
    retry_count: int
    needs_human_input: bool
    human_question: str
    human_answer: str

    # Observability
    messages: Annotated[list[BaseMessage], add_messages]
    trace_log: Annotated[list[dict], _append_list]
