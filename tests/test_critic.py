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

    def test_unparseable_fallback(self):
        text = "This is not JSON at all"
        result = _parse_critic_response(text)
        assert result["verdict"] == "PASS"  # Safe fallback


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
