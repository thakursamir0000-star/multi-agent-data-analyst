"""
LLM helpers — shared by every agent node.

Config values are read lazily on first use (not at import time) so that
Streamlit Cloud secrets are available when they are read.

Each node still constructs its own ``ChatGroq`` so tests can keep
patching ``agents.<node>.ChatGroq``.
"""

from __future__ import annotations

import re
from typing import Any

from tools.config import get_env

DEFAULT_MODEL = "openai/gpt-oss-120b"

_MODEL: str | None = None
_MAX_RETRIES: int | None = None
_CORE_TOOLS: bool | None = None

# A closed <think>...</think> or <thinking>...</thinking> block
_THINK_BLOCK_RE = re.compile(r"<(think|thinking)>.*?</\1>", flags=re.DOTALL)
# An opening tag left unclosed after closed blocks are removed
_THINK_OPEN_RE = re.compile(r"<think(?:ing)?>")


def get_model() -> str:
    """Return the Groq model name (GROQ_MODEL), read once on first use."""
    global _MODEL
    if _MODEL is None:
        _MODEL = get_env("GROQ_MODEL", DEFAULT_MODEL)
    return _MODEL


def get_max_retries() -> int:
    """Return the max Critic → Coder retries (MAX_RETRIES), read once on first use."""
    global _MAX_RETRIES
    if _MAX_RETRIES is None:
        _MAX_RETRIES = int(get_env("MAX_RETRIES", "2"))
    return _MAX_RETRIES


def core_tools_enabled() -> bool:
    """Whether the Coder may use deterministic core tools (ENABLE_CORE_TOOLS,
    default true), read once on first use. Set it to false to restore the
    code-only Coder."""
    global _CORE_TOOLS
    if _CORE_TOOLS is None:
        value = get_env("ENABLE_CORE_TOOLS", "true").strip().lower()
        _CORE_TOOLS = value not in ("0", "false", "no", "off")
    return _CORE_TOOLS


def strip_thinking(text: str) -> str:
    """Remove <think>/<thinking> reasoning blocks from LLM output.

    Closed blocks are removed. An unclosed opening tag truncates the text
    from that tag to the end.
    """
    text = _THINK_BLOCK_RE.sub("", text).strip()
    match = _THINK_OPEN_RE.search(text)
    if match:
        text = text[: match.start()].strip()
    return text


def count_tokens(*responses: Any) -> dict[str, int]:
    """Sum token usage across LLM responses via their ``usage_metadata``.

    Returns ``{"input_tokens", "output_tokens", "total_tokens"}``. Responses
    without usage metadata (e.g. test mocks) contribute zero.
    """
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for response in responses:
        usage = getattr(response, "usage_metadata", None)
        if not isinstance(usage, dict):
            continue
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] += value
    return totals
