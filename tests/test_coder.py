"""
Tests for agents/coder.py — Coder node.

Uses mocked LLM and real sandbox to test code generation and execution.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agents.coder import _extract_code, coder_node


# ── Code Extraction Tests ──────────────────────────────────────────

class TestExtractCode:
    def test_plain_code(self):
        text = "print(df.head())"
        assert _extract_code(text) == "print(df.head())"

    def test_markdown_python_fence(self):
        text = "```python\nprint(df.head())\n```"
        assert _extract_code(text) == "print(df.head())"

    def test_markdown_fence_no_language(self):
        text = "```\nprint(df.head())\n```"
        assert _extract_code(text) == "print(df.head())"

    def test_whitespace_handling(self):
        text = "  \n  print(df.head())  \n  "
        assert "print(df.head())" in _extract_code(text)


# ── Node Tests (Mocked LLM) ────────────────────────────────────────

class TestCoderNode:
    @patch("agents.coder.ChatGroq")
    def test_coder_generates_and_executes_code(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = 'print(df["a"].sum())'
        mock_llm.invoke.return_value = mock_response
        mock_groq_cls.return_value = mock_llm

        df = pd.DataFrame({"a": [10, 20, 30], "b": [1, 2, 3]})

        state = {
            "sub_tasks": ["Sum column a"],
            "current_task_index": 0,
            "dataframe_profile": "a (int64), b (int64)",
            "critic_feedback": "",
            "_df": df,
        }

        result = coder_node(state)
        assert "code_outputs" in result
        assert len(result["code_outputs"]) == 1
        assert "60" in result["code_outputs"][0]["stdout"]
        assert result["current_task_index"] == 1

    @patch("agents.coder.ChatGroq")
    def test_coder_advances_task_index(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = 'print("done")'
        mock_llm.invoke.return_value = mock_response
        mock_groq_cls.return_value = mock_llm

        state = {
            "sub_tasks": ["Task 1", "Task 2"],
            "current_task_index": 0,
            "dataframe_profile": "",
            "critic_feedback": "",
            "_df": pd.DataFrame({"x": [1]}),
        }

        result = coder_node(state)
        assert result["current_task_index"] == 1

    @patch("agents.coder.ChatGroq")
    def test_coder_handles_all_tasks_done(self, mock_groq_cls):
        state = {
            "sub_tasks": ["Task 1"],
            "current_task_index": 1,  # Already past the last task
            "dataframe_profile": "",
            "critic_feedback": "",
            "_df": pd.DataFrame(),
        }

        result = coder_node(state)
        assert result["current_task_index"] == 1  # Unchanged

    @patch("agents.coder.ChatGroq")
    def test_coder_captures_errors(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        # Code that will raise a runtime error (undefined column)
        mock_response.content = 'print(df["nonexistent"].sum())'
        mock_llm.invoke.return_value = mock_response
        mock_groq_cls.return_value = mock_llm

        df = pd.DataFrame({"a": [1, 2, 3]})

        state = {
            "sub_tasks": ["Sum nonexistent column"],
            "current_task_index": 0,
            "dataframe_profile": "a (int64)",
            "critic_feedback": "",
            "_df": df,
        }

        result = coder_node(state)
        # Should have attempted self-correction (2 LLM calls)
        assert mock_llm.invoke.call_count == 2
