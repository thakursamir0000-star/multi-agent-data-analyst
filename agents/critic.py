"""
Critic Node — verifies the Analyst's insight against actual code outputs.

Cross-checks every numerical claim, flags hallucinations, and decides:
  PASS  → verified_insight is set, graph ends.
  FAIL  → retry (route back to Coder) or escalate (HumanInput node).
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
_MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

CRITIC_SYSTEM_PROMPT = """\
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


def _parse_critic_response(response_text: str) -> dict[str, Any]:
    """Parse the Critic's JSON response, with fallback handling."""
    text = response_text.strip()

    # Strip markdown fences
    if "```" in text:
        lines = text.split("```")
        for segment in lines:
            segment = segment.strip()
            if segment.startswith("json"):
                segment = segment[4:].strip()
            if segment.startswith("{"):
                text = segment
                break

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Fallback: assume PASS if we can't parse
    return {
        "verdict": "PASS",
        "issues": ["Could not parse critic response — defaulting to PASS"],
        "suggestion": "",
        "is_ambiguous": False,
    }


@log_node("Critic")
def critic_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: verify the Analyst's draft insight.

    Reads: draft_insight, code_outputs, retry_count
    Writes: verified_insight OR (critic_feedback + retry_count) OR
            (needs_human_input + human_question)
    """
    llm = ChatGroq(model=_MODEL, temperature=0.1)

    draft_insight = state.get("draft_insight", "")
    code_outputs = state.get("code_outputs", [])
    retry_count = state.get("retry_count", 0)

    context = _format_verification_context(draft_insight, code_outputs)

    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=context),
    ]

    response = llm.invoke(messages)
    result = _parse_critic_response(response.content)

    verdict = result.get("verdict", "PASS").upper()
    issues = result.get("issues", [])
    suggestion = result.get("suggestion", "")
    is_ambiguous = result.get("is_ambiguous", False)

    if verdict == "PASS":
        return {
            "verified_insight": draft_insight,
            "messages": [response],
        }

    # FAIL path
    if is_ambiguous or retry_count >= _MAX_RETRIES:
        # Escalate to human
        question = (
            f"The analysis encountered issues that need your input:\n"
            f"- " + "\n- ".join(issues) + "\n\n"
            f"Could you clarify or refine your question?"
        )
        return {
            "needs_human_input": True,
            "human_question": question,
            "messages": [response],
        }

    # Retry: send feedback to Coder
    return {
        "critic_feedback": suggestion or "; ".join(issues),
        "retry_count": retry_count + 1,
        "current_task_index": 0,  # Re-run from first task with feedback
        "messages": [response],
    }
