"""
Tests for agents/llm.py — shared LLM helpers.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

import agents.llm as llm
from agents.llm import count_tokens, strip_thinking


# ── strip_thinking ──────────────────────────────────────────────────

class TestStripThinking:
    def test_no_tags_unchanged(self):
        assert strip_thinking("  Revenue is 100  ") == "Revenue is 100"

    def test_closed_thinking_block_removed(self):
        text = "<thinking>let me reason</thinking>Revenue is 100"
        assert strip_thinking(text) == "Revenue is 100"

    def test_closed_think_block_removed(self):
        text = "<think>let me reason</think>\nRevenue is 100"
        assert strip_thinking(text) == "Revenue is 100"

    def test_multiline_block_removed(self):
        text = "<think>\nline 1\nline 2\n</think>\n[\"Task 1\"]"
        assert strip_thinking(text) == '["Task 1"]'

    def test_multiple_blocks_removed(self):
        text = "<think>a</think>Answer<thinking>b</thinking> here"
        assert strip_thinking(text) == "Answer here"

    def test_unclosed_thinking_truncates(self):
        text = "Revenue is 100 <thinking>still reasoning"
        assert strip_thinking(text) == "Revenue is 100"

    def test_unclosed_think_truncates(self):
        text = "Revenue is 100 <think>still reasoning"
        assert strip_thinking(text) == "Revenue is 100"

    def test_only_unclosed_tag_returns_empty(self):
        assert strip_thinking("<think>never finished") == ""


# ── Lazy config readers ────────────────────────────────────────────

@pytest.fixture
def reset_llm_cache():
    """Clear the module-level caches before and after each test."""
    llm._MODEL = None
    llm._MAX_RETRIES = None
    llm._CORE_TOOLS = None
    yield
    llm._MODEL = None
    llm._MAX_RETRIES = None
    llm._CORE_TOOLS = None


class TestConfigReaders:
    def test_get_model_reads_env_once(self, reset_llm_cache):
        with patch("agents.llm.get_env", return_value="some/model") as mock_env:
            assert llm.get_model() == "some/model"
            assert llm.get_model() == "some/model"
        mock_env.assert_called_once_with("GROQ_MODEL", llm.DEFAULT_MODEL)

    def test_get_model_default(self, reset_llm_cache):
        with patch("agents.llm.get_env", side_effect=lambda key, default="": default):
            assert llm.get_model() == "openai/gpt-oss-120b"

    def test_get_max_retries_is_int(self, reset_llm_cache):
        with patch("agents.llm.get_env", return_value="3"):
            assert llm.get_max_retries() == 3

    def test_core_tools_enabled_by_default(self, reset_llm_cache):
        with patch("agents.llm.get_env", side_effect=lambda key, default="": default) as mock_env:
            assert llm.core_tools_enabled() is True
            assert llm.core_tools_enabled() is True
        mock_env.assert_called_once_with("ENABLE_CORE_TOOLS", "true")

    @pytest.mark.parametrize("value, expected", [
        ("false", False), ("0", False), (" OFF ", False), ("no", False),
        ("true", True), ("1", True), ("yes", True),
    ])
    def test_core_tools_flag_values(self, reset_llm_cache, value, expected):
        with patch("agents.llm.get_env", return_value=value):
            assert llm.core_tools_enabled() is expected


# ── count_tokens ────────────────────────────────────────────────────

def _ai_message(input_tokens: int, output_tokens: int) -> AIMessage:
    return AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


class TestCountTokens:
    def test_single_response(self):
        assert count_tokens(_ai_message(120, 30)) == {
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
        }

    def test_sums_multiple_responses(self):
        result = count_tokens(_ai_message(100, 20), _ai_message(110, 25))
        assert result == {
            "input_tokens": 210,
            "output_tokens": 45,
            "total_tokens": 255,
        }

    def test_missing_usage_counts_zero(self):
        result = count_tokens(AIMessage(content="no usage"), MagicMock())
        assert result == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def test_no_responses(self):
        assert count_tokens()["total_tokens"] == 0
