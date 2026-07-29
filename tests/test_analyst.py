"""
Tests for agents/analyst.py — Analyst node.
"""

from unittest.mock import MagicMock, patch

import pytest

from agents.analyst import analyst_node


class TestAnalystNode:
    @patch("agents.analyst.ChatGroq")
    def test_analyst_produces_insight(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            "The top product by revenue is Widget Pro at $125,000. "
            "The total dataset revenue is $450,000."
        )
        mock_llm.invoke.return_value = mock_response
        mock_groq_cls.return_value = mock_llm

        state = {
            "user_query": "What are the top products by revenue?",
            "code_outputs": [
                {
                    "task": "Group by product, sum revenue",
                    "code": "print(df.groupby('product')['revenue'].sum())",
                    "stdout": "Widget Pro    125000\nGadget X      95000\nWidget A      80000",
                    "plot_base64": None,
                    "error": None,
                }
            ],
        }

        result = analyst_node(state)
        assert "draft_insight" in result
        assert len(result["draft_insight"]) > 0
        assert "messages" in result

    @patch("agents.analyst.ChatGroq")
    def test_analyst_handles_empty_outputs(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "No data was available for analysis."
        mock_llm.invoke.return_value = mock_response
        mock_groq_cls.return_value = mock_llm

        state = {
            "user_query": "Analyze trends",
            "code_outputs": [],
        }

        result = analyst_node(state)
        assert "draft_insight" in result

    @patch("agents.analyst.ChatGroq")
    def test_analyst_handles_error_outputs(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "The analysis could not be completed due to an error."
        mock_llm.invoke.return_value = mock_response
        mock_groq_cls.return_value = mock_llm

        state = {
            "user_query": "Calculate correlation",
            "code_outputs": [
                {
                    "task": "Calculate correlation",
                    "code": "print(df.corr())",
                    "stdout": "",
                    "plot_base64": None,
                    "error": "KeyError: 'nonexistent'",
                }
            ],
        }

        result = analyst_node(state)
        assert "draft_insight" in result
