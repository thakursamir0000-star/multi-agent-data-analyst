"""
Critic Node — verifies the Analyst's insight against actual code outputs.

Cross-checks every numerical claim, flags hallucinations, and decides:
  PASS  → verified_insight is set, graph ends.
  FAIL  → retry (route back to Coder) or escalate (HumanInput node).
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from agents.llm import count_tokens, get_max_retries, get_model, strip_thinking
from tools.observability import TOKEN_USAGE_KEY, log_node

CRITIC_SYSTEM_PROMPT = """\
/no_think
You are the **Critic** agent in a multi-agent data analysis system.

Your job: cross-check the Analyst's insight against the actual code
execution outputs. You are a fact-checker, not a writer.

## Rules
1. Compare every numerical claim in the insight to the actual stdout
   outputs from code execution.
2. Flag any number, percentage, ranking, or trend that does NOT appear
   in the code outputs — that's a hallucination.
3. Check logical consistency: does the insight's conclusion follow from
   the data shown?
4. If a code execution had an error but the insight ignores it, flag that.

## Output Format
Respond with a JSON object (no markdown fences):
{
  "verdict": "PASS" | "FAIL",
  "issues": ["list of specific issues found"],
  "suggestion": "If FAIL, specific guidance for the Coder to fix things",
  "is_ambiguous": true/false  // true if the issue needs human clarification
}

If everything checks out, return:
{"verdict": "PASS", "issues": [], "suggestion": "", "is_ambiguous": false}
"""


def _format_verification_context(
    draft_insight: str, code_outputs: list[dict]
) -> str:
    """Build the verification context for the Critic."""
    parts = ["## Draft Insight\n", draft_insight, "\n\n## Actual Code Outputs\n"]

    for i, output in enumerate(code_outputs):
        parts.append(f"### Task {i + 1}: {output.get('task', 'N/A')}")
        if output.get("stdout"):
            parts.append(f"stdout:\n```\n{output['stdout']}\n```")
        if output.get("error"):
            parts.append(f"error: {output['error']}")
        parts.append("")

    return "\n".join(parts)


def _fail_result(issue: str) -> dict[str, Any]:
    """A FAIL verdict for a Critic response that could not be trusted."""
    return {
        "verdict": "FAIL",
        "issues": [issue],
        "suggestion": "",
        "is_ambiguous": False,
    }


def _normalize_critic_result(parsed: Any) -> dict[str, Any]:
    """Validate a parsed Critic response. Fails closed: anything other than
    an explicit "PASS" or "FAIL" verdict is treated as FAIL."""
    if not isinstance(parsed, dict):
        return _fail_result(
            "Verification failed: the Critic's response was not a JSON object."
        )

    verdict = parsed.get("verdict")
    if not isinstance(verdict, str) or verdict.strip().upper() not in ("PASS", "FAIL"):
        return _fail_result(
            f"Verification failed: the Critic returned a missing or invalid "
            f"verdict ({verdict!r})."
        )

    issues = parsed.get("issues", [])
    if isinstance(issues, str):
        issues = [issues]
    elif not isinstance(issues, list):
        issues = []

    suggestion = parsed.get("suggestion", "")
    return {
        "verdict": verdict.strip().upper(),
        "issues": [str(issue) for issue in issues],
        "suggestion": suggestion if isinstance(suggestion, str) else "",
        "is_ambiguous": parsed.get("is_ambiguous") is True,
    }


def _parse_critic_response(response_text: str) -> dict[str, Any]:
    """Parse and validate the Critic's JSON response.

    Fails closed: malformed or empty output, a missing verdict, or an
    invalid verdict all return a FAIL result.
    """
    text = response_text.strip()

    # Strip markdown fences
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
            if in_block and stripped.startswith("{"):
                text = "\n".join(
                    ln for ln in lines[lines.index(line) :]
                )
                break

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return _normalize_critic_result(json.loads(text[start : end + 1]))
        except json.JSONDecodeError:
            pass

    return _fail_result(
        "Verification failed: the Critic's response could not be parsed, "
        "so the answer was not verified."
    )


@log_node("Critic")
def critic_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: verify the Analyst's draft insight.

    Reads: draft_insight, code_outputs, retry_count
    Writes: verified_insight OR (critic_feedback + retry_count) OR
            (needs_human_input + human_question)
    """
    llm = ChatGroq(model=get_model(), temperature=0.1)

    draft_insight = state.get("draft_insight", "")
    code_outputs = state.get("code_outputs", [])
    retry_count = state.get("retry_count", 0)

    context = _format_verification_context(draft_insight, code_outputs)

    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=context),
    ]

    response = llm.invoke(messages)
    tokens = count_tokens(response)
    result = _parse_critic_response(strip_thinking(response.content))

    verdict = result["verdict"]  # always "PASS" or "FAIL" after parsing
    issues = result["issues"]
    suggestion = result["suggestion"]
    is_ambiguous = result["is_ambiguous"]

    if verdict == "PASS":
        return {
            "verified_insight": draft_insight,
            "messages": [response],
            TOKEN_USAGE_KEY: tokens,
        }

    # FAIL path
    if is_ambiguous or retry_count >= get_max_retries():
        # Escalate to human
        question = (
            "The analysis encountered issues that need your input:\n"
            "- " + "\n- ".join(issues) + "\n\n"
            "Could you clarify or refine your question?"
        )
        return {
            "needs_human_input": True,
            "human_question": question,
            "messages": [response],
            TOKEN_USAGE_KEY: tokens,
        }

    # Retry: send feedback to Coder
    return {
        "critic_feedback": suggestion or "; ".join(issues),
        "retry_count": retry_count + 1,
        "current_task_index": 0,  # Re-run from first task with feedback
        "messages": [response],
        TOKEN_USAGE_KEY: tokens,
    }
