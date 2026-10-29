"""
Tests for agents/analyst.py — Analyst node.
"""

from unittest.mock import MagicMock, patch


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


class TestAnalystToolOutputs:
    @patch("agents.analyst.ChatGroq")
    def test_tool_call_labelled_as_tool_not_code(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="A leads with 40.")
        mock_groq_cls.return_value = mock_llm
        call = '{"tool": "group_aggregate", "arguments": {"group_by": "category"}}'

        analyst_node({
            "user_query": "Revenue by category?",
            "code_outputs": [
                {
                    "task": "Sum revenue by category",
                    "code": call,
                    "tool": "group_aggregate",
                    "stdout": 'Result of group_aggregate:\n{"rows": []}',
                    "plot_base64": None,
                    "error": None,
                },
                {
                    "task": "Print total",
                    "code": "print(df.revenue.sum())",
                    "tool": None,
                    "stdout": "60",
                    "plot_base64": None,
                    "error": None,
                },
            ],
        })

        prompt = mock_llm.invoke.call_args[0][0][-1].content
        assert f"**Tool call (group_aggregate):**\n```json\n{call}\n```" in prompt
        assert "**Code:**\n```python\nprint(df.revenue.sum())\n```" in prompt
