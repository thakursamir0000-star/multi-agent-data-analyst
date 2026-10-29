"""
Tests for agents/critic.py — Critic node.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.critic import _parse_critic_response, critic_node


# ── Parser Tests ────────────────────────────────────────────────────

class TestParseCriticResponse:
    def test_valid_pass_response(self):
        text = '{"verdict": "PASS", "issues": [], "suggestion": "", "is_ambiguous": false}'
        result = _parse_critic_response(text)
        assert result["verdict"] == "PASS"
        assert result["issues"] == []

    def test_valid_fail_response(self):
        text = json.dumps({
            "verdict": "FAIL",
            "issues": ["Revenue number is wrong"],
            "suggestion": "Re-calculate revenue",
            "is_ambiguous": False,
        })
        result = _parse_critic_response(text)
        assert result["verdict"] == "FAIL"
        assert len(result["issues"]) == 1

    def test_json_with_markdown(self):
        text = '```json\n{"verdict": "PASS", "issues": [], "suggestion": "", "is_ambiguous": false}\n```'
        result = _parse_critic_response(text)
        assert result["verdict"] == "PASS"

    def test_unparseable_fails_closed(self):
        text = "This is not JSON at all"
        result = _parse_critic_response(text)
        assert result["verdict"] == "FAIL"
        assert result["is_ambiguous"] is False
        assert "could not be parsed" in result["issues"][0]

    @pytest.mark.parametrize(
        "text",
        [
            "",  # empty response
            '{"verdict": "PASS", "issues": [',  # truncated JSON
            '{"issues": [], "suggestion": ""}',  # missing verdict
            '{"verdict": "MAYBE", "issues": []}',  # invalid verdict
            '{"verdict": true, "issues": []}',  # non-string verdict
            '{"verdict": null}',  # null verdict
            '["PASS"]',  # not a JSON object
        ],
        ids=["empty", "truncated", "missing", "invalid", "non-string", "null", "array"],
    )
    def test_untrusted_output_fails_closed(self, text):
        assert _parse_critic_response(text)["verdict"] == "FAIL"

    def test_verdict_case_and_whitespace_tolerated(self):
        result = _parse_critic_response('{"verdict": " pass ", "issues": []}')
        assert result["verdict"] == "PASS"

    def test_string_issues_and_ambiguity_normalized(self):
        text = '{"verdict": "FAIL", "issues": "Wrong total", "is_ambiguous": "false"}'
        result = _parse_critic_response(text)
        assert result["issues"] == ["Wrong total"]
        assert result["is_ambiguous"] is False  # only JSON true counts


# ── Node Tests ──────────────────────────────────────────────────────

class TestCriticNode:
    @patch("agents.critic.ChatGroq")
    def test_critic_pass(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "verdict": "PASS",
            "issues": [],
            "suggestion": "",
            "is_ambiguous": False,
        })
        mock_llm.invoke.return_value = mock_response
        mock_groq_cls.return_value = mock_llm

        state = {
            "draft_insight": "Revenue is $100K",
            "code_outputs": [{"task": "Sum revenue", "stdout": "100000", "error": None}],
            "retry_count": 0,
        }

        result = critic_node(state)
        assert result.get("verified_insight") == "Revenue is $100K"

    @patch("agents.critic.ChatGroq")
    def test_critic_fail_with_retry(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "verdict": "FAIL",
            "issues": ["Revenue should be $200K not $100K"],
            "suggestion": "Recalculate total revenue",
            "is_ambiguous": False,
        })
        mock_llm.invoke.return_value = mock_response
        mock_groq_cls.return_value = mock_llm

        state = {
            "draft_insight": "Revenue is $100K",
            "code_outputs": [{"task": "Sum revenue", "stdout": "200000", "error": None}],
            "retry_count": 0,
        }

        result = critic_node(state)
        assert "verified_insight" not in result or not result.get("verified_insight")
        assert result.get("critic_feedback")
        assert result["retry_count"] == 1

    @patch("agents.critic.ChatGroq")
    def test_critic_escalate_on_max_retries(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "verdict": "FAIL",
            "issues": ["Still wrong"],
            "suggestion": "Need clarification",
            "is_ambiguous": False,
        })
        mock_llm.invoke.return_value = mock_response
        mock_groq_cls.return_value = mock_llm

        state = {
            "draft_insight": "Revenue is $100K",
            "code_outputs": [{"task": "Sum revenue", "stdout": "200000", "error": None}],
            "retry_count": 2,  # Already at max
        }

        result = critic_node(state)
        assert result.get("needs_human_input") is True
        assert result.get("human_question")

    @patch("agents.critic.ChatGroq")
    def test_critic_escalate_on_ambiguous(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "verdict": "FAIL",
            "issues": ["Query is too vague"],
            "suggestion": "Ask user what time period",
            "is_ambiguous": True,
        })
        mock_llm.invoke.return_value = mock_response
        mock_groq_cls.return_value = mock_llm

        state = {
            "draft_insight": "Trends are mixed",
            "code_outputs": [],
            "retry_count": 0,
        }

        result = critic_node(state)
        assert result.get("needs_human_input") is True

    @patch("agents.critic.ChatGroq")
    def test_malformed_output_takes_retry_path(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Looks good to me!"  # no JSON at all
        mock_llm.invoke.return_value = mock_response
        mock_groq_cls.return_value = mock_llm

        state = {
            "draft_insight": "Revenue is $100K",
            "code_outputs": [{"task": "Sum revenue", "stdout": "100000", "error": None}],
            "retry_count": 0,
        }

        result = critic_node(state)
        assert not result.get("verified_insight")
        assert result.get("critic_feedback")
        assert result["retry_count"] == 1
        assert result["current_task_index"] == 0

    @patch("agents.critic.ChatGroq")
    def test_missing_verdict_escalates_after_max_retries(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"issues": [], "suggestion": ""}'
        mock_llm.invoke.return_value = mock_response
        mock_groq_cls.return_value = mock_llm

        state = {
            "draft_insight": "Revenue is $100K",
            "code_outputs": [{"task": "Sum revenue", "stdout": "100000", "error": None}],
            "retry_count": 2,  # Already at max
        }

        result = critic_node(state)
        assert not result.get("verified_insight")
        assert result.get("needs_human_input") is True
        assert "invalid verdict" in result["human_question"]
