"""
LLM helpers — shared by every agent node.

Config values are read lazily on first use (not at import time) so that
Streamlit Cloud secrets are available when they are read.

Two providers, chosen with LLM_PROVIDER:
- "groq" (default): ChatGroq with GROQ_API_KEY / GROQ_MODEL;
- "gemini": ChatGoogleGenerativeAI with GEMINI_API_KEY / GEMINI_MODEL.

Each node passes its own ``ChatGroq`` class to ``chat_model`` so tests can
keep patching ``agents.<node>.ChatGroq``.
"""

from __future__ import annotations

import re
import sys
import threading
import time
from collections import deque
from typing import Any

from tools.config import get_env

DEFAULT_MODEL = "openai/gpt-oss-120b"  # Groq
DEFAULT_GEMINI_MODEL = "gemini-4-hydrogen"  # free tier; fast enough for the benchmark
DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-120b"
DEFAULT_OPENROUTER_PROVIDER = "primeintellect"  # OpenRouter's slug for Prime Intellect
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PROVIDERS = ("groq", "gemini", "openrouter")
_KEY_NAMES = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}

_PROVIDER: str | None = None
_MODEL: str | None = None
_MAX_RETRIES: int | None = None
_API_RETRIES: int | None = None
_CORE_TOOLS: bool | None = None
_REQUESTS_PER_MINUTE: int | None = None

# A closed <think>...</think> or <thinking>...</thinking> block
_THINK_BLOCK_RE = re.compile(r"<(think|thinking)>.*?</\1>", flags=re.DOTALL)
# An opening tag left unclosed after closed blocks are removed
_THINK_OPEN_RE = re.compile(r"<think(?:ing)?>")


def get_provider() -> str:
    """Return the LLM provider (LLM_PROVIDER: groq, gemini or openrouter), read once on first use."""
    global _PROVIDER
    if _PROVIDER is None:
        provider = get_env("LLM_PROVIDER", "groq").strip().lower() or "groq"
        if provider not in PROVIDERS:
            raise ValueError(f"LLM_PROVIDER must be one of {', '.join(PROVIDERS)}, not {provider!r}")
        _PROVIDER = provider
    return _PROVIDER


def api_key_name() -> str:
    """The environment variable that holds the active provider's API key."""
    return _KEY_NAMES[get_provider()]


def get_model() -> str:
    """Return the model name for the active provider (GROQ_MODEL, GEMINI_MODEL or
    OPENROUTER_MODEL), read once."""
    global _MODEL
    if _MODEL is None:
        if get_provider() == "gemini":
            _MODEL = get_env("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        elif get_provider() == "openrouter":
            _MODEL = get_env("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
        else:
            _MODEL = get_env("GROQ_MODEL", DEFAULT_MODEL)
    return _MODEL


def openrouter_routing() -> dict[str, Any]:
    """OpenRouter provider routing: which provider serves the model, and whether
    OpenRouter may fall back to another one.

    OPENROUTER_PROVIDER (default primeintellect; comma-separated for an order)
    and OPENROUTER_ALLOW_FALLBACKS (default false). Fallbacks stay off for the
    benchmark: other providers serve the same model at other precisions, so
    one run must not mix them.
    """
    order = [p.strip() for p in get_env("OPENROUTER_PROVIDER", DEFAULT_OPENROUTER_PROVIDER).split(",") if p.strip()]
    allow = get_env("OPENROUTER_ALLOW_FALLBACKS", "false").strip().lower() in ("1", "true", "yes", "on")
    routing: dict[str, Any] = {"allow_fallbacks": allow}
    if order:
        routing["order"] = order
    return routing


def get_max_retries() -> int:
    """Return the max Critic → Coder retries (MAX_RETRIES), read once on first use."""
    global _MAX_RETRIES
    if _MAX_RETRIES is None:
        _MAX_RETRIES = int(get_env("MAX_RETRIES", "2"))
    return _MAX_RETRIES


def get_api_retries() -> int:
    """Attempts for one LLM call after a rate limit (LLM_API_RETRIES, or the
    older GROQ_API_RETRIES; default 6), read once on first use.

    The Groq client waits the time Groq suggests before each retry. Its default
    of 2 wasn't enough on the free tier: one question uses ~8-11k tokens and the
    limit is 8k tokens per minute, so a busy minute failed whole questions.
    Gemini calls are retried by RetryingChatModel instead (see there).
    """
    global _API_RETRIES
    if _API_RETRIES is None:
        _API_RETRIES = int(get_env("LLM_API_RETRIES", get_env("GROQ_API_RETRIES", "6")))
    return _API_RETRIES


def get_requests_per_minute() -> int:
    """Client-side cap on LLM requests per minute (LLM_REQUESTS_PER_MINUTE, 0 = off).

    Free tiers limit requests per minute per project; pacing requests below
    that avoids 429s instead of recovering from them.
    """
    global _REQUESTS_PER_MINUTE
    if _REQUESTS_PER_MINUTE is None:
        _REQUESTS_PER_MINUTE = max(0, int(get_env("LLM_REQUESTS_PER_MINUTE", "0") or 0))
    return _REQUESTS_PER_MINUTE


def core_tools_enabled() -> bool:
    """Whether the Coder may use deterministic core tools (ENABLE_CORE_TOOLS,
    default true), read once on first use. Set it to false to restore the
    code-only Coder."""
    global _CORE_TOOLS
    if _CORE_TOOLS is None:
        value = get_env("ENABLE_CORE_TOOLS", "true").strip().lower()
        _CORE_TOOLS = value not in ("0", "false", "no", "off")
    return _CORE_TOOLS


# ── rate limits ─────────────────────────────────────────────────────

_RETRY_AFTER = (
    re.compile(r"try again in (?:(\d+)m)?([\d.]+)s"),  # Groq: "Please try again in 1m43.2s"
    re.compile(r"retry in ([\d.]+)s"),  # Gemini: "Please retry in 33.6s."
    re.compile(r"retryDelay'?\"?:\s*'?\"?([\d.]+)s"),  # Gemini RetryInfo: 'retryDelay': '33s'
)
_OVERLOADED = ("UNAVAILABLE", "overloaded", "DEADLINE_EXCEEDED", "INTERNAL", "503", "504")
# OpenRouter (openai SDK) errors that are worth a retry: upstream provider
# failures, server errors and timeouts. Used for OpenRouter only, so the Groq
# and Gemini paths retry exactly as before.
_OPENROUTER_TRANSIENT = _OVERLOADED + (
    "Error code: 500", "Error code: 502", "Provider returned error", "APITimeoutError",
    "APIConnectionError", "Request timed out", "returned no answer",
)


def rate_limit_kind(error: str | None) -> str | None:
    """"daily" / "minute" for a Groq or Gemini rate-limit error, else None."""
    if not error or not ("429" in error or "rate limit" in error.lower() or "RESOURCE_EXHAUSTED" in error):
        return None
    # Groq says "tokens per day (TPD)"; Gemini's quota IDs say "...PerDayPerProject..."
    return "daily" if re.search(r"per ?day", error, flags=re.IGNORECASE) else "minute"


def retry_after_seconds(error: str, default: float = 30.0) -> float:
    """The wait the provider suggests in a rate-limit error, in seconds."""
    groq = _RETRY_AFTER[0].search(error)
    if groq:
        return int(groq.group(1) or 0) * 60 + float(groq.group(2))
    for pattern in _RETRY_AFTER[1:]:
        match = pattern.search(error)
        if match:
            return float(match.group(1))
    return default


_LOCK = threading.Lock()
_CALL_TIMES: deque[float] = deque()
_STATS = {"requests": 0, "wait_s": 0.0, "billed_usd": 0.0}
_SERVED_BY: dict[str, int] = {}  # provider name -> answered calls (OpenRouter reports it)


def request_stats() -> dict[str, Any]:
    """LLM requests sent (including retried ones), seconds spent waiting, what
    OpenRouter billed, and which upstream providers answered, so far."""
    with _LOCK:
        return {**_STATS, "served_by": dict(_SERVED_BY)}


def _record_answer(provider: str | None, cost: float | None) -> None:
    with _LOCK:
        key = provider or "unknown"
        _SERVED_BY[key] = _SERVED_BY.get(key, 0) + 1
        if cost:
            _STATS["billed_usd"] += float(cost)


def _pace(rpm: int, sleep=time.sleep, now=time.monotonic) -> None:
    """Block until a request fits in the last minute's budget, then record it."""
    while True:
        with _LOCK:
            t = now()
            while _CALL_TIMES and t - _CALL_TIMES[0] >= 60:
                _CALL_TIMES.popleft()
            if not rpm or len(_CALL_TIMES) < rpm:
                _CALL_TIMES.append(t)
                _STATS["requests"] += 1
                return
            wait = 60 - (t - _CALL_TIMES[0]) + 0.05
            _STATS["wait_s"] += wait
        sleep(wait)


class RetryingChatModel:
    """A chat model whose ``invoke`` paces and retries requests within bounds.

    - Requests are paced to LLM_REQUESTS_PER_MINUTE (when set).
    - A per-minute rate limit waits the delay the provider suggests, and at
      least an exponential backoff (4, 8, 16, ... s); the Gemini SDK's own
      retries ignore the suggested delay, so they are turned off.
    - A temporary server error (overloaded, unavailable) backs off exponentially.
    - A daily quota error is raised at once: retrying can't succeed until it resets.
    Gives up after ``attempts`` tries.
    """

    def __init__(self, model: Any, attempts: int, *, rpm: int = 0, max_wait: float = 90.0,
                 sleep=time.sleep, now=time.monotonic, transient: tuple[str, ...] = _OVERLOADED):
        self.model = model
        self.attempts = max(1, attempts)
        self.rpm = rpm
        self.max_wait = max_wait
        self.transient = transient
        self._sleep = sleep
        self._now = now

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        for attempt in range(1, self.attempts + 1):
            _pace(self.rpm, self._sleep, self._now)
            try:
                return self.model.invoke(messages, **kwargs)
            except Exception as e:  # inspected below; anything else is re-raised
                text = f"{type(e).__name__}: {e}"
                kind = rate_limit_kind(text)
                overloaded = kind is None and any(marker in text for marker in self.transient)
                if kind == "daily" or not (kind == "minute" or overloaded) or attempt == self.attempts:
                    raise
                backoff = min(2.0 ** (attempt + 1), self.max_wait)
                wait = min(max(retry_after_seconds(text, 0.0) + 1, backoff), self.max_wait)
                what = "rate limit" if kind else "server busy"
                print(f"   {what}; retrying in {wait:.0f}s (attempt {attempt + 1}/{self.attempts})",
                      file=sys.stderr)
                with _LOCK:
                    _STATS["wait_s"] += wait
                self._sleep(wait)
        raise AssertionError("unreachable")  # pragma: no cover


_ROLES = {"system": "system", "human": "user", "ai": "assistant"}


class OpenRouterChat:
    """A minimal chat model over OpenRouter's OpenAI-compatible API (openai SDK).

    ``invoke(messages)`` takes LangChain messages and returns an ``AIMessage``,
    like the LangChain clients, so the agents don't change. Every request:
    - pins the upstream provider (``routing``, see ``openrouter_routing``);
    - asks OpenRouter for the billed cost (``usage.include``).
    The reply records which provider actually answered and what was billed
    (``response_metadata``); reasoning tokens are counted in the output tokens,
    since they are billed as output. The SDK's own retries are off:
    RetryingChatModel retries instead.
    """

    def __init__(self, model: str, temperature: float, api_key: str, routing: dict[str, Any],
                 *, base_url: str = OPENROUTER_BASE_URL, timeout: float = 180.0, client: Any = None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=timeout)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.routing = routing

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        from langchain_core.messages import AIMessage

        payload = [{"role": _ROLES.get(getattr(m, "type", "human"), "user"), "content": response_text(m)}
                   for m in messages]
        response = self.client.chat.completions.create(
            model=self.model, messages=payload, temperature=self.temperature,
            extra_body={"provider": self.routing, "usage": {"include": True}},
        )
        extra = getattr(response, "model_extra", None) or {}
        if not getattr(response, "choices", None):
            raise RuntimeError(f"OpenRouter returned no answer: {extra.get('error') or response}")
        choice = response.choices[0]
        usage = response.usage
        usage_extra = getattr(usage, "model_extra", None) or {}
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = int(getattr(details, "reasoning_tokens", 0) or 0)
        provider = getattr(response, "provider", None) or extra.get("provider")
        cost = getattr(usage, "cost", None) or usage_extra.get("cost")
        _record_answer(provider, cost)
        input_tokens, output_tokens = int(usage.prompt_tokens or 0), int(usage.completion_tokens or 0)
        return AIMessage(
            content=choice.message.content or "",
            usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                            "output_token_details": {"reasoning": reasoning}},
            response_metadata={"provider": provider, "model": getattr(response, "model", None),
                               "id": getattr(response, "id", None), "billed_usd": cost,
                               "finish_reason": choice.finish_reason},
        )


def chat_model(groq_class: Any, temperature: float) -> RetryingChatModel:
    """The chat model for one agent call, for the active provider.

    ``groq_class`` is the node's own ``ChatGroq`` import (patched in tests).
    Groq retries rate limits itself (honouring Groq's suggested wait), so the
    wrapper doesn't retry it again; Gemini's SDK retries are turned off
    (``max_retries=1``) and the wrapper retries instead; so are OpenRouter's.
    """
    rpm = get_requests_per_minute()
    if get_provider() == "openrouter":
        model = OpenRouterChat(get_model(), temperature, get_env("OPENROUTER_API_KEY"), openrouter_routing())
        return RetryingChatModel(model, get_api_retries(), rpm=rpm, transient=_OPENROUTER_TRANSIENT)
    if get_provider() == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(
            model=get_model(), temperature=temperature, max_retries=1,
            google_api_key=get_env("GEMINI_API_KEY"),
        )
        return RetryingChatModel(model, get_api_retries(), rpm=rpm)
    model = groq_class(model=get_model(), temperature=temperature, max_retries=get_api_retries())
    return RetryingChatModel(model, 1, rpm=rpm)


# ── responses ───────────────────────────────────────────────────────

def response_text(response: Any) -> str:
    """The text of a chat response.

    Groq returns a string; Gemini may return a list of content blocks
    (text, plus reasoning or signatures), of which only the text is kept.
    """
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")


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
