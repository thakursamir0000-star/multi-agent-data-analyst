"""
Tests for agents/planner.py — Planner node.

Uses mocked LLM to test parsing and state updates without API calls.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.planner import _parse_sub_tasks, planner_node


# ── Parser Tests ────────────────────────────────────────────────────

class TestParseSubTasks:
    def test_valid_json_array(self):
        text = '["Task 1", "Task 2", "Task 3"]'
        result = _parse_sub_tasks(text)
        assert result == ["Task 1", "Task 2", "Task 3"]

    def test_json_with_markdown_fences(self):
        text = '```json\n["Task 1", "Task 2"]\n```'
        result = _parse_sub_tasks(text)
        assert result == ["Task 1", "Task 2"]

    def test_json_with_preamble(self):
        text = 'Here are the tasks:\n["Task 1", "Task 2"]'
        result = _parse_sub_tasks(text)
        assert result == ["Task 1", "Task 2"]

    def test_numbered_list_fallback(self):
        text = "1. Group by category and sum revenue\n2. Sort descending and take top results\n3. Create a bar chart visualization"
        result = _parse_sub_tasks(text)
        assert len(result) == 3
        assert "Group by category" in result[0]

    def test_empty_response(self):
        result = _parse_sub_tasks("")
        assert result == []

    def test_single_task(self):
        text = '["Calculate the average revenue per region"]'
        result = _parse_sub_tasks(text)
        assert len(result) == 1


# ── Node Tests (Mocked LLM) ────────────────────────────────────────

class TestPlannerNode:
    @patch("agents.planner.ChatGroq")
    def test_planner_produces_sub_tasks(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps([
            "Group by 'product', sum 'revenue'",
            "Sort descending, take top 5",
            "Create a bar chart",
        ])
        mock_llm.invoke.return_value = mock_response
        mock_groq_cls.return_value = mock_llm

        state = {
            "user_query": "What are the top 5 products by revenue?",
            "dataframe_profile": "Dataset: 500 rows x 10 columns\n  - product (object)\n  - revenue (float64)",
            "human_answer": "",
        }

        result = planner_node(state)
        assert "sub_tasks" in result
        assert len(result["sub_tasks"]) == 3
        assert result["current_task_index"] == 0

    @patch("agents.planner.ChatGroq")
    def test_planner_handles_human_answer(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '["Refined task based on clarification"]'
        mock_llm.invoke.return_value = mock_response
        mock_groq_cls.return_value = mock_llm

        state = {
            "user_query": "Analyze trends",
            "dataframe_profile": "Dataset: 100 rows x 5 columns",
            "human_answer": "I meant monthly revenue trends",
        }

        result = planner_node(state)
        # Verify the LLM was called with the clarification
        call_args = mock_llm.invoke.call_args[0][0]
        human_msg = call_args[-1].content
        assert "monthly revenue trends" in human_msg

    @patch("agents.planner.ChatGroq")
    def test_planner_fallback_on_empty_response(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = ""
        mock_llm.invoke.return_value = mock_response
        mock_groq_cls.return_value = mock_llm

        state = {
            "user_query": "What is the average?",
            "dataframe_profile": "",
            "human_answer": "",
        }

        result = planner_node(state)
        assert len(result["sub_tasks"]) >= 1  # Fallback task
