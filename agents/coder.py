"""
Coder Node — generates and executes pandas code for each sub-task.

Iterates through sub_tasks one at a time. For each task, asks the LLM
to write pandas/matplotlib code, then runs it in the sandbox. On execution
error, does one internal self-correction attempt before passing the error
upstream to the Critic.

When the graph is built with a dataset registry, the LLM may instead reply
with one JSON tool call (e.g. group_aggregate); it runs through the
deterministic core layer (core.tools) and returns an exact, structured
result. Anything else is treated as code and runs in the sandbox as before
(the run_pandas fallback).
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from agents.llm import core_tools_enabled, count_tokens, get_model, strip_thinking
from agents.state import NewAttempt
from core.datasets import DatasetRegistry
from core.errors import CoreError
from core.schemas import ChartResult
from core.tools import parse_tool_call, result_json, run_tool
from tools.observability import TOKEN_USAGE_KEY, log_node
from tools.sandbox import SecurityError, run_code

CODER_SYSTEM_PROMPT = """\
/no_think
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
10. Do not read or write files; plots are captured automatically.
"""

# Appended to CODER_SYSTEM_PROMPT when core tools are available.
# <TABLE> is replaced with the dataset's SQL table name.
CORE_TOOLS_PROMPT = """
## Deterministic tools — use them instead of code whenever one fits
This overrides rule 7: if a tool below can do the task, reply with exactly
ONE JSON object that calls it (no markdown, no explanation) instead of
code. Tools run exact, tested code on the dataset as uploaded and return
structured results, so their numbers are guaranteed correct:
{"tool": "<name>", "arguments": {...}}

Tools and their arguments (optional ones may be omitted):
- describe: {"columns": ["col", ...]} — count/mean/std/min/quartiles/max;
  default: all numeric columns
- group_aggregate: {"group_by": "col" or ["col", ...],
  "agg": "count|sum|mean|median|min|max", "value_column": "col",
  "sort": "desc|asc|none", "limit": 50} — value_column is required unless
  agg is "count"
- correlate: {"columns": ["col", ...], "method": "pearson|spearman"}
- detect_anomalies: {"column": "col", "method": "iqr|zscore", "threshold": 1.5}
- make_chart: {"chart_type": "bar|line|scatter|histogram", "x": "col",
  "y": "col", "agg": "count|sum|mean|median|min|max", "title": "...",
  "limit": 20} — bar: agg of y per x, largest first, top `limit` bars;
  line: agg of y per x in x order; scatter: y vs x; histogram: x only
- query_sql: {"sql": "SELECT ... FROM <TABLE> ..."} — one read-only DuckDB
  SELECT; the dataset is the table <TABLE>

Examples:
{"tool": "group_aggregate", "arguments": {"group_by": "category",
"agg": "sum", "value_column": "revenue", "limit": 5}}
{"tool": "make_chart", "arguments": {"chart_type": "bar", "x": "category",
"y": "revenue", "agg": "sum", "limit": 5}}

Write Python code only when no tool fits: derived columns, date parsing,
reshaping, multi-step logic or a custom plot.
"""

# Last line of the human message: with tools on, it must not ask for code only
_REPLY_CODE = "Write the Python code (no markdown, no explanation):"
_REPLY_TOOL_OR_CODE = (
    "Reply with ONE JSON tool call if a tool fits the task, otherwise the "
    "Python code (no markdown, no explanation):"
)


def _build_coder_prompt(
    task: str,
    profile_text: str,
    critic_feedback: str = "",
    previous_error: str = "",
    tools: bool = False,
) -> str:
    """Build the human message for the coder."""
    parts = [f"## Dataset Profile\n{profile_text}\n"]

    if critic_feedback:
        parts.append(f"## Critic Feedback (fix this)\n{critic_feedback}\n")

    if previous_error:
        parts.append(f"## Previous Execution Error\n{previous_error}\n")

    parts.append(f"## Task\n{task}\n")
    parts.append(_REPLY_TOOL_OR_CODE if tools else _REPLY_CODE)
    return "\n".join(parts)


def _extract_code(response_text: str) -> str:
    """Extract pure Python code from LLM response, stripping markdown fences,
    thinking tags, and any surrounding conversational text."""
    text = strip_thinking(response_text)

    # Find code block between ``` markers (handles surrounding text)
    if "```" in text:
        lines = text.split("\n")
        in_block = False
        code_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_block:
                    break  # End of code block
                in_block = True
                continue
            if in_block:
                code_lines.append(line)
        if code_lines:
            return "\n".join(code_lines).strip()

    return text.strip()


def _system_prompt(core: tuple[DatasetRegistry, str] | None) -> str:
    """The Coder system prompt, with the tool menu when core tools are on."""
    if core is None:
        return CODER_SYSTEM_PROMPT
    registry, dataset_id = core
    table = registry.get(dataset_id).table_name
    return CODER_SYSTEM_PROMPT + CORE_TOOLS_PROMPT.replace("<TABLE>", table)


def _tool_request(reply: str) -> dict[str, Any] | str | None:
    """Detect a JSON tool call in the LLM reply.

    Returns the parsed object, an error message for a malformed tool call,
    or None when the reply is not a tool call (i.e. it is code).
    """
    text = reply.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        return f"Invalid tool call JSON: {e}" if '"tool"' in text else None
    if isinstance(payload, dict) and "tool" in payload:
        return payload
    return None


def _error_result(message: str) -> dict[str, Any]:
    return {"stdout": "", "plot_base64": None, "error": message, "result": None}


def _run_core_tool(
    request: dict[str, Any] | str, core: tuple[DatasetRegistry, str]
) -> dict[str, Any]:
    """Run a tool call through the core layer. Errors become normal error
    results so they flow into self-correction, like sandbox errors."""
    if isinstance(request, str):
        return _error_result(request)
    registry, dataset_id = core
    try:
        call = parse_tool_call(request)
        result = run_tool(registry, dataset_id, call.tool, call.arguments)
    except CoreError as e:
        return _error_result(f"{type(e).__name__}: {e}")
    except Exception as e:  # a tool bug must not crash the graph
        return _error_result(f"Tool {request.get('tool')!r} failed: {type(e).__name__}: {e}")
    return {
        "stdout": f"Result of {call.tool}:\n{result_json(result)}\n",
        "plot_base64": result.image_base64 if isinstance(result, ChartResult) else None,
        "error": None,
        "result": None,
    }


def _execute(
    reply: str, local_vars: dict[str, Any], core: tuple[DatasetRegistry, str] | None
) -> tuple[dict[str, Any], str | None]:
    """Run the LLM reply: a tool call through the core layer, anything else
    in the sandbox. Returns (execution result, tool name or None)."""
    if core is not None:
        request = _tool_request(reply)
        if request is not None:
            tool = request.get("tool") if isinstance(request, dict) else None
            return _run_core_tool(request, core), tool if isinstance(tool, str) else None
    return _run_sandboxed(reply, local_vars), None


def _run_sandboxed(code: str, local_vars: dict[str, Any]) -> dict[str, Any]:
    """Run code in the sandbox. A SecurityError becomes a normal error
    result so it flows into self-correction instead of crashing the graph."""
    try:
        return run_code(code, local_vars=local_vars)
    except SecurityError as e:
        return {
            "stdout": "",
            "plot_base64": None,
            "error": f"SecurityError: {e}",
            "result": None,
        }


@log_node("Coder")
def coder_node(
    state: dict[str, Any],
    *,
    df=None,
    registry: DatasetRegistry | None = None,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    """LangGraph node: write and execute code for the current sub-task.

    Reads: sub_tasks, current_task_index, dataframe_profile, critic_feedback
    Writes: code_outputs, current_task_index, messages

    Parameters
    ----------
    df : pd.DataFrame, optional
        The DataFrame to execute code against. Passed via the graph
        wrapper closure — NOT through LangGraph state.
    registry, dataset_id : optional
        The dataset registry and the ID of the dataset being analysed.
        When both are given (and ENABLE_CORE_TOOLS is not false), the LLM
        may answer with a deterministic core tool call instead of code.
    """
    llm = ChatGroq(model=get_model(), temperature=0.1)

    sub_tasks = state.get("sub_tasks", [])
    task_idx = state.get("current_task_index", 0)
    profile_text = state.get("dataframe_profile", "")
    critic_feedback = state.get("critic_feedback", "")

    if task_idx >= len(sub_tasks):
        return {"current_task_index": task_idx}

    task = sub_tasks[task_idx]
    core = (
        (registry, dataset_id)
        if registry is not None and dataset_id is not None and core_tools_enabled()
        else None
    )
    system_prompt = _system_prompt(core)

    # Build prompt
    prompt_text = _build_coder_prompt(
        task=task,
        profile_text=profile_text,
        critic_feedback=critic_feedback,
        tools=core is not None,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt_text),
    ]

    response = llm.invoke(messages)
    llm_responses = [response]
    code = _extract_code(response.content)

    # Execute: a core tool call, or code in the sandbox
    local_vars = {"df": df} if df is not None else {}
    exec_result, tool = _execute(code, local_vars, core)

    # One internal self-correction if there's an error
    if exec_result.get("error") and not critic_feedback:
        retry_prompt = _build_coder_prompt(
            task=task,
            profile_text=profile_text,
            previous_error=exec_result["error"],
            tools=core is not None,
        )
        retry_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=retry_prompt),
        ]
        retry_response = llm.invoke(retry_messages)
        llm_responses.append(retry_response)
        code = _extract_code(retry_response.content)
        exec_result, tool = _execute(code, local_vars, core)
        response = retry_response  # Use the corrected response for messages

    # Build output record
    output_record = {
        "task": task,
        "task_index": task_idx,
        "code": code,
        "tool": tool,  # core tool name, or None for sandbox code
        "stdout": exec_result.get("stdout", ""),
        "plot_base64": exec_result.get("plot_base64"),
        "error": exec_result.get("error"),
    }

    return {
        # Task 0 starts an attempt (new plan or Critic retry): replace the
        # previous attempt's outputs instead of appending to them
        "code_outputs": NewAttempt([output_record]) if task_idx == 0 else [output_record],
        "current_task_index": task_idx + 1,
        "critic_feedback": "",  # Clear critic feedback after processing
        "messages": [response],
        # Both the initial attempt and any self-correction retry
        TOKEN_USAGE_KEY: count_tokens(*llm_responses),
    }
