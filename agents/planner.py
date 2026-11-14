"""
Planner Node — breaks a user query into concrete, pandas-executable sub-tasks.

Uses the dataset profile to ground the plan in actual column names and dtypes,
preventing the LLM from inventing columns or operations that don't apply.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from agents.llm import count_tokens, get_api_retries, get_model, strip_thinking
from agents.state import question_with_clarification
from tools.observability import TOKEN_USAGE_KEY, log_node

PLANNER_SYSTEM_PROMPT = """\
/no_think
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
6. Sub-tasks run independently: each one is a separate snippet on the FULL
   dataset, and nothing (variables, filtered tables) carries over to the next.
   So every sub-task that computes a number must itself state ALL the
   conditions the question sets: the time period, the group (e.g. a region,
   category or product) and any threshold. Never make "filter the rows" a
   sub-task of its own.
7. Prefer ONE sub-task that answers the question when that is possible; add
   more only for genuinely separate parts (e.g. a chart, a second period).
8. Output ONLY the JSON array. No thinking, no explanation, no markdown.

## Output format
["task 1", "task 2"]

Examples:
["Group by 'product', sum 'revenue', sort descending, take the top 5", "Create a horizontal bar chart of the top 5 products by summed 'revenue'"]

For "Which store had the highest average order value in March 2024?":
["Keep rows where 'order_date' is in March 2024, group by 'store', take the mean of 'order_value', sort descending and print the top store with its value"]
"""


def _build_planner_prompt(user_query: str, profile_text: str) -> str:
    """Build the human message for the planner."""
    return (
        f"## Dataset Profile\n{profile_text}\n\n"
        f"## User Question\n{user_query}\n\n"
        f"Produce the sub-task list (JSON array of strings):"
    )


# Matches "1. task", "2) task", "- task", "* task", "• task"
_LIST_ITEM_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+(.+?)\s*$")


def _parse_list_items(text: str) -> list[str]:
    """Fallback: extract sub-tasks from a numbered or bulleted list."""
    items = []
    for line in text.splitlines():
        match = _LIST_ITEM_RE.match(line)
        if match:
            items.append(match.group(1).strip())
    return [t for t in items if len(t) > 5]


def _parse_sub_tasks(response_text: str) -> list[str]:
    """Extract a JSON list of strings from the LLM response.

    Handles common issues: markdown code fences, leading text, etc.
    Falls back to parsing a numbered/bulleted list if no JSON array is found.
    """
    text = response_text.strip()

    # Strip markdown code fences
    if "```" in text:
        lines = text.split("\n")
        in_block = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_block:
                    break
                in_block = True
                continue
            if in_block and stripped.startswith("["):
                text = "\n".join(
                    ln for ln in lines[lines.index(line) :]
                )
                break

    # Find the JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    try:
        tasks = json.loads(text)
        if isinstance(tasks, list) and all(isinstance(t, str) for t in tasks):
            return [t.strip() for t in tasks if len(t.strip()) > 5]
    except json.JSONDecodeError:
        pass

    return _parse_list_items(response_text)


@log_node("Planner")
def planner_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: plan sub-tasks for the user query.

    Reads: user_query, dataframe_profile, human_answer (if re-planning)
    Writes: sub_tasks, current_task_index, messages
    """
    llm = ChatGroq(model=get_model(), temperature=0.2, max_retries=get_api_retries())

    # If re-planning after human clarification, the question includes the answer
    user_query = question_with_clarification(state)
    profile_text = state.get("dataframe_profile", "")

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=_build_planner_prompt(user_query, profile_text)),
    ]

    response = llm.invoke(messages)
    sub_tasks = _parse_sub_tasks(strip_thinking(response.content))

    # Ensure at least one task
    if not sub_tasks:
        sub_tasks = [f"Analyze the dataset to answer: {user_query}"]

    return {
        "sub_tasks": sub_tasks,
        "current_task_index": 0,
        "critic_feedback": "",  # a new plan starts without the old plan's feedback
        "messages": [
            HumanMessage(content=user_query),
            response,
        ],
        TOKEN_USAGE_KEY: count_tokens(response),
    }
