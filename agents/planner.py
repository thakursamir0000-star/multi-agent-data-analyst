"""
Planner Node — breaks a user query into concrete, pandas-executable sub-tasks.

Uses the dataset profile to ground the plan in actual column names and dtypes,
preventing the LLM from inventing columns or operations that don't apply.
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from tools.observability import log_node

load_dotenv()

_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")

PLANNER_SYSTEM_PROMPT = """\
You are the **Planner** agent in a multi-agent data analysis system.

Your job: given a user's natural-language question and a dataset profile,
produce an ordered list of concrete, pandas-executable sub-tasks that,
when completed sequentially, will fully answer the question.

## Rules
1. Each sub-task must reference ACTUAL column names from the dataset profile.
2. Each sub-task must be specific enough that a Coder agent can write
   a single pandas code snippet to accomplish it.
3. Do NOT produce vague tasks like "analyze the data" or "explore trends".
   Instead: "Group by 'category' column, sum 'revenue', sort descending, take top 5."
4. If a visualization is needed, include a separate sub-task for it
   (e.g., "Create a bar chart of top 5 categories by revenue").
5. Limit to 5 sub-tasks maximum. If the query is simple, 1-2 is fine.
6. Return ONLY a JSON array of strings. No preamble, no explanation.

## Example output
["Filter rows where 'region' == 'West'", "Group by 'product', sum 'revenue', sort descending", "Take top 10 results", "Create a horizontal bar chart of the top 10"]
"""


def _build_planner_prompt(user_query: str, profile_text: str) -> str:
    """Build the human message for the planner."""
    return (
        f"## Dataset Profile\n{profile_text}\n\n"
        f"## User Question\n{user_query}\n\n"
        f"Produce the sub-task list (JSON array of strings):"
    )


def _parse_sub_tasks(response_text: str) -> list[str]:
    """Extract a JSON list of strings from the LLM response.

    Handles common issues: markdown code fences, leading text, etc.
    """
    text = response_text.strip()

    # Strip markdown code fences
    if "```" in text:
        lines = text.split("```")
        for segment in lines:
            segment = segment.strip()
            if segment.startswith("json"):
                segment = segment[4:].strip()
            if segment.startswith("["):
                text = segment
                break

    # Find the JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    try:
        tasks = json.loads(text)
        if isinstance(tasks, list) and all(isinstance(t, str) for t in tasks):
            return tasks
    except json.JSONDecodeError:
        pass

    # Fallback: split by newlines if JSON parsing fails
    lines = [
        line.strip().lstrip("0123456789.-) ")
        for line in response_text.strip().split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]
    return [line for line in lines if len(line) > 10]


@log_node("Planner")
def planner_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: plan sub-tasks for the user query.

    Reads: user_query, dataframe_profile, human_answer (if re-planning)
    Writes: sub_tasks, current_task_index, messages
    """
    llm = ChatGroq(model=_MODEL, temperature=0.2)

    user_query = state["user_query"]
    profile_text = state.get("dataframe_profile", "")

    # If re-planning after human clarification, incorporate the answer
    human_answer = state.get("human_answer", "")
    if human_answer:
        user_query = (
            f"{user_query}\n\n"
            f"[Clarification from user]: {human_answer}"
        )

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=_build_planner_prompt(user_query, profile_text)),
    ]

    response = llm.invoke(messages)
    sub_tasks = _parse_sub_tasks(response.content)

    # Ensure at least one task
    if not sub_tasks:
        sub_tasks = [f"Analyze the dataset to answer: {user_query}"]

    return {
        "sub_tasks": sub_tasks,
        "current_task_index": 0,
        "messages": [
            HumanMessage(content=user_query),
            response,
        ],
    }
