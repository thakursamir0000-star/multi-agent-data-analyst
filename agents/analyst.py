"""
Analyst Node — synthesizes code outputs into plain-English insights.

Takes the accumulated code_outputs (stdout, errors, plots) and produces
a cohesive, data-backed narrative that a non-technical user can understand.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from tools.observability import log_node

load_dotenv()

_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

ANALYST_SYSTEM_PROMPT = """\
You are the **Analyst** agent in a multi-agent data analysis system.

Your job: given a set of code execution results (stdout outputs, any errors),
write a clear, concise, data-backed insight paragraph that answers the
user's original question.

## Rules
1. Cite specific numbers from the code outputs. Never invent or estimate
   numbers that aren't in the outputs.
2. If a code execution had an error, mention that the specific sub-task
   could not be completed and note the limitation.
3. Write for a non-technical audience. Avoid jargon.
4. Structure your response with a brief summary first, then supporting details.
5. If plots were generated, reference them (e.g., "as shown in the chart").
6. Keep the response to 2-4 paragraphs maximum.
7. If the data reveals something unexpected or noteworthy, call it out.
"""


def _format_code_outputs(code_outputs: list[dict]) -> str:
    """Format code outputs into a readable string for the LLM."""
    parts = []
    for i, output in enumerate(code_outputs):
        parts.append(f"### Sub-task {i + 1}: {output.get('task', 'N/A')}")
        parts.append(f"**Code:**\n```python\n{output.get('code', '')}\n```")

        if output.get("error"):
            parts.append(f"**Error:** {output['error']}")
        elif output.get("stdout"):
            parts.append(f"**Output:**\n```\n{output['stdout']}\n```")
        else:
            parts.append("**Output:** (no printed output)")

        if output.get("plot_base64"):
            parts.append("**Plot:** A chart was generated for this task.")

        parts.append("")

    return "\n".join(parts)


@log_node("Analyst")
def analyst_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: synthesize code outputs into an insight narrative.

    Reads: user_query, code_outputs
    Writes: draft_insight, messages
    """
    llm = ChatGroq(model=_MODEL, temperature=0.3)

    user_query = state.get("user_query", "")
    code_outputs = state.get("code_outputs", [])

    formatted_outputs = _format_code_outputs(code_outputs)

    prompt = (
        f"## Original Question\n{user_query}\n\n"
        f"## Code Execution Results\n{formatted_outputs}\n\n"
        f"Write your insight based on these results:"
    )

    messages = [
        SystemMessage(content=ANALYST_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)

    return {
        "draft_insight": response.content,
        "messages": [response],
    }
