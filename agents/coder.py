"""
Coder Node — generates and executes pandas code for each sub-task.

Iterates through sub_tasks one at a time. For each task, asks the LLM
to write pandas/matplotlib code, then runs it in the sandbox. On execution
error, does one internal self-correction attempt before passing the error
upstream to the Critic.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from tools.observability import log_node
from tools.sandbox import run_code

load_dotenv()

_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")

CODER_SYSTEM_PROMPT = """\
You are the **Coder** agent in a multi-agent data analysis system.

Your job: write Python code using pandas and matplotlib to accomplish
a specific sub-task on a DataFrame called `df`.

## Rules
1. The variable `df` is already loaded in the execution namespace.
   Do NOT load any files. `df` is ready to use.
2. Use `pd` for pandas and `plt` for matplotlib (both pre-imported).
3. Use `np` for numpy (pre-imported).
4. `print()` all results so they appear in stdout.
5. For plots, always call `plt.title()`, `plt.xlabel()`, `plt.ylabel()`
   and `plt.tight_layout()` before finishing.
6. Do NOT use `plt.show()` — plots are captured automatically.
7. Return ONLY the Python code. No markdown fences, no explanations.
8. Handle potential issues: missing values (use .dropna() or .fillna()),
   type conversions, empty results.
9. Keep code concise — typically 5-20 lines.
"""


def _build_coder_prompt(
    task: str,
    profile_text: str,
    critic_feedback: str = "",
    previous_error: str = "",
) -> str:
    """Build the human message for the coder."""
    parts = [f"## Dataset Profile\n{profile_text}\n"]

    if critic_feedback:
        parts.append(f"## Critic Feedback (fix this)\n{critic_feedback}\n")

    if previous_error:
        parts.append(f"## Previous Execution Error\n{previous_error}\n")

    parts.append(f"## Task\n{task}\n")
    parts.append("Write the Python code (no markdown, no explanation):")
    return "\n".join(parts)


def _extract_code(response_text: str) -> str:
    """Extract pure Python code from LLM response, stripping markdown fences."""
    text = response_text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```python or ```)
        lines = lines[1:]
        # Remove last ``` if present
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    return text.strip()


@log_node("Coder")
def coder_node(state: dict[str, Any], *, df=None) -> dict[str, Any]:
    """LangGraph node: write and execute code for the current sub-task.

    Reads: sub_tasks, current_task_index, dataframe_profile, critic_feedback
    Writes: code_outputs, current_task_index, messages

    Parameters
    ----------
    df : pd.DataFrame, optional
        The DataFrame to execute code against. Passed via the graph
        wrapper closure — NOT through LangGraph state.
    """
    llm = ChatGroq(model=_MODEL, temperature=0.1)

    sub_tasks = state.get("sub_tasks", [])
    task_idx = state.get("current_task_index", 0)
    profile_text = state.get("dataframe_profile", "")
    critic_feedback = state.get("critic_feedback", "")

    if task_idx >= len(sub_tasks):
        return {"current_task_index": task_idx}

    task = sub_tasks[task_idx]

    # Build prompt
    prompt_text = _build_coder_prompt(
        task=task,
        profile_text=profile_text,
        critic_feedback=critic_feedback,
    )

    messages = [
        SystemMessage(content=CODER_SYSTEM_PROMPT),
        HumanMessage(content=prompt_text),
    ]

    response = llm.invoke(messages)
    code = _extract_code(response.content)

    # Execute in sandbox
    local_vars = {"df": df} if df is not None else {}
    exec_result = run_code(code, local_vars=local_vars)

    # One internal self-correction if there's an error
    if exec_result.get("error") and not critic_feedback:
        retry_prompt = _build_coder_prompt(
            task=task,
            profile_text=profile_text,
            previous_error=exec_result["error"],
        )
        retry_messages = [
            SystemMessage(content=CODER_SYSTEM_PROMPT),
            HumanMessage(content=retry_prompt),
        ]
        retry_response = llm.invoke(retry_messages)
        code = _extract_code(retry_response.content)
        exec_result = run_code(code, local_vars=local_vars)
        response = retry_response  # Use the corrected response for messages

    # Build output record
    output_record = {
        "task": task,
        "task_index": task_idx,
        "code": code,
        "stdout": exec_result.get("stdout", ""),
        "plot_base64": exec_result.get("plot_base64"),
        "error": exec_result.get("error"),
    }

    return {
        "code_outputs": [output_record],
        "current_task_index": task_idx + 1,
        "critic_feedback": "",  # Clear critic feedback after processing
        "messages": [response],
    }
