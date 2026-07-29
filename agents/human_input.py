"""
HumanInput Node — pauses the graph for human-in-the-loop interaction.

Uses LangGraph's interrupt() mechanism to surface a clarifying question
to the user via the Streamlit UI.  The graph resumes when the user
provides an answer.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from tools.observability import log_node


@log_node("HumanInput")
def human_input_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: pause execution and ask the user for clarification.

    Reads: human_question, needs_human_input
    Writes: human_answer, needs_human_input

    The interrupt() call pauses the graph. When the user resumes with
    their answer (via Command(resume=answer)), this node returns the
    answer to be routed back to the Planner for re-planning.
    """
    question = state.get("human_question", "Could you provide more details?")

    # This call pauses the graph and waits for user input
    human_answer = interrupt(question)

    return {
        "human_answer": human_answer,
        "needs_human_input": False,
        "retry_count": 0,  # Reset retries after human clarification
    }
