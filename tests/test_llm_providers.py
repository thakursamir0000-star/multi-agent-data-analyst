"""
Tests for the provider layer in agents/llm.py: choosing Groq or Gemini,
pacing and retrying requests within bounds, and reading Gemini responses.
No network: the models are mocks.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

import agents.llm as llm
from agents.llm import RetryingChatModel, rate_limit_kind, response_text, retry_after_seconds

# Shapes of real provider errors (ids shortened)
GEMINI_MINUTE = (
    "ChatGoogleGenerativeAIError: Error calling model 'gemini-3.8-flash' (RESOURCE_EXHAUSTED): 429 "
    "RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota. Please retry in "
    "21.4s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', "
    "'violations': [{'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaValue': '10'}]}, "
    "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '21s'}]}}"
)
GEMINI_DAILY = GEMINI_MINUTE.replace("PerMinutePerProject", "PerDayPerProject").replace("'10'", "'250'")
GEMINI_BUSY = "ServerError: 503 UNAVAILABLE. {'error': {'message': 'The model is overloaded. Please try again later.'}}"


@pytest.fixture(autouse=True)
def reset_llm_cache():
    names = ("_PROVIDER", "_MODEL", "_API_RETRIES", "_REQUESTS_PER_MINUTE")
    for name in names:
        setattr(llm, name, None)
    llm._CALL_TIMES.clear()
    yield
    for name in names:
        setattr(llm, name, None)
    llm._CALL_TIMES.clear()


def _env(**values):
    return patch("agents.llm.get_env", side_effect=lambda key, default="": values.get(key, default))


# ── choosing the provider ───────────────────────────────────────────

def test_groq_is_the_default_provider():
    with _env():
        assert llm.get_provider() == "groq" and llm.api_key_name() == "GROQ_API_KEY"
        assert llm.get_model() == llm.DEFAULT_MODEL


def test_gemini_provider_reads_its_own_key_and_model():
    with _env(LLM_PROVIDER="Gemini", GEMINI_MODEL="gemini-x"):
        assert llm.get_provider() == "gemini" and llm.api_key_name() == "GEMINI_API_KEY"
        assert llm.get_model() == "gemini-x"
    llm._MODEL = None
    with _env(LLM_PROVIDER="gemini"):
        assert llm.get_model() == llm.DEFAULT_GEMINI_MODEL


def test_unknown_provider_is_rejected():
    with _env(LLM_PROVIDER="openai"), pytest.raises(ValueError, match="groq, gemini"):
        llm.get_provider()


def test_retries_setting_has_a_provider_neutral_name():
    with _env(LLM_API_RETRIES="4", GROQ_API_RETRIES="9"):
        assert llm.get_api_retries() == 4
    llm._API_RETRIES = None
    with _env(GROQ_API_RETRIES="9"):
        assert llm.get_api_retries() == 9


def test_chat_model_builds_groq_with_its_own_retries():
    groq_cls = MagicMock()
    with _env():
        wrapped = llm.chat_model(groq_cls, temperature=0.2)
    assert groq_cls.call_args.kwargs == {"model": llm.DEFAULT_MODEL, "temperature": 0.2, "max_retries": 6}
    assert wrapped.attempts == 1  # the Groq client already retries, honouring Groq's suggested wait


def test_chat_model_builds_gemini_with_sdk_retries_off():
    groq_cls = MagicMock()
    with _env(LLM_PROVIDER="gemini", GEMINI_API_KEY="test-key-not-real", LLM_REQUESTS_PER_MINUTE="8"), \
            patch("langchain_google_genai.ChatGoogleGenerativeAI") as gemini_cls:
        wrapped = llm.chat_model(groq_cls, temperature=0.1)
    groq_cls.assert_not_called()
    kwargs = gemini_cls.call_args.kwargs
    assert kwargs["model"] == llm.DEFAULT_GEMINI_MODEL and kwargs["temperature"] == 0.1
    assert kwargs["max_retries"] == 1 and kwargs["google_api_key"] == "test-key-not-real"
    assert (wrapped.attempts, wrapped.rpm) == (6, 8)


# ── reading rate-limit errors ───────────────────────────────────────

def test_gemini_rate_limit_errors_are_classified():
    assert rate_limit_kind(GEMINI_MINUTE) == "minute"
    assert rate_limit_kind(GEMINI_DAILY) == "daily"
    assert rate_limit_kind(GEMINI_BUSY) is None
    assert retry_after_seconds(GEMINI_MINUTE) == pytest.approx(21.4)
    assert retry_after_seconds("RetryInfo {'retryDelay': '17s'}") == 17.0


# ── retrying within bounds ──────────────────────────────────────────

def _model(*outcomes):
    model = MagicMock()
    model.invoke.side_effect = list(outcomes)
    return model


def test_a_busy_minute_waits_the_suggested_delay_then_succeeds():
    waits = []
    model = _model(RuntimeError(GEMINI_MINUTE), AIMessage(content="ok"))
    reply = RetryingChatModel(model, 6, sleep=waits.append).invoke(["m"])
    assert reply.content == "ok" and model.invoke.call_count == 2
    assert waits == [pytest.approx(22.4)]  # Gemini's 21.4 s + 1, above the 4 s backoff


def test_overloads_back_off_exponentially_and_give_up_after_the_bound():
    waits = []
    model = _model(*[RuntimeError(GEMINI_BUSY)] * 4)
    with pytest.raises(RuntimeError, match="UNAVAILABLE"):
        RetryingChatModel(model, 4, sleep=waits.append).invoke(["m"])
    assert model.invoke.call_count == 4 and waits == [4.0, 8.0, 16.0]


def test_a_daily_quota_is_not_retried():
    model = _model(RuntimeError(GEMINI_DAILY))
    with pytest.raises(RuntimeError, match="PerDay"):
        RetryingChatModel(model, 6, sleep=lambda s: pytest.fail("no waiting on a daily quota")).invoke(["m"])
    assert model.invoke.call_count == 1


def test_other_errors_are_not_retried():
    model = _model(ValueError("bad request"))
    with pytest.raises(ValueError):
        RetryingChatModel(model, 6, sleep=lambda s: pytest.fail("no retry")).invoke(["m"])


def test_requests_are_paced_and_counted():
    clock = {"t": 0.0}
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        clock["t"] += seconds

    before = llm.request_stats()["requests"]
    wrapped = RetryingChatModel(_model(*["r"] * 3), 1, rpm=2, sleep=sleep, now=lambda: clock["t"])
    for _ in range(3):
        wrapped.invoke(["m"])
    assert llm.request_stats()["requests"] - before == 3
    assert waits == [pytest.approx(60.05)]  # the third request waited for the minute to roll over


# ── reading responses ───────────────────────────────────────────────

def test_response_text_handles_strings_and_content_blocks():
    assert response_text(AIMessage(content="plain")) == "plain"
    blocks = [{"type": "thinking", "thinking": "hidden"}, {"type": "text", "text": "Answer "},
              {"type": "text", "text": "here", "extras": {"signature": "abc"}}]
    assert response_text(AIMessage(content=blocks)) == "Answer here"
    assert response_text(MagicMock(content="from a mock")) == "from a mock"
