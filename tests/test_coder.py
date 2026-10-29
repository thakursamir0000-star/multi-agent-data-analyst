"""
Tests for agents/coder.py — Coder node.

Uses mocked LLM and real sandbox to test code generation and execution.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

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
        }

        result = coder_node(state, df=df)
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
        }

        result = coder_node(state, df=pd.DataFrame({"x": [1]}))
        assert result["current_task_index"] == 1

    @patch("agents.coder.ChatGroq")
    def test_coder_handles_all_tasks_done(self, mock_groq_cls):
        state = {
            "sub_tasks": ["Task 1"],
            "current_task_index": 1,  # Already past the last task
            "dataframe_profile": "",
            "critic_feedback": "",
        }

        result = coder_node(state, df=pd.DataFrame())
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
        }

        result = coder_node(state, df=df)
        # Should have attempted self-correction (2 LLM calls)
        assert mock_llm.invoke.call_count == 2
        # The error must be the missing column, not a missing `df`
        assert "KeyError" in result["code_outputs"][0]["error"]


# ── Sandbox SecurityError Handling ─────────────────────────────────

def _mock_llm_returning(*contents):
    """A mock LLM whose successive invoke() calls return these contents."""
    mock_llm = MagicMock()
    responses = []
    for content in contents:
        response = MagicMock()
        response.content = content
        responses.append(response)
    mock_llm.invoke.side_effect = responses
    return mock_llm


class TestCoderSecurityError:
    @patch("agents.coder.ChatGroq")
    def test_security_error_triggers_self_correction(self, mock_groq_cls):
        mock_llm = _mock_llm_returning(
            "df.to_csv('out.csv')",  # blocked by the sandbox
            'print(df["a"].sum())',  # corrected
        )
        mock_groq_cls.return_value = mock_llm
        state = {
            "sub_tasks": ["Sum column a"],
            "current_task_index": 0,
            "dataframe_profile": "a (int64)",
            "critic_feedback": "",
        }

        result = coder_node(state, df=pd.DataFrame({"a": [10, 20, 30]}))

        assert mock_llm.invoke.call_count == 2
        # The retry prompt tells the LLM what was blocked
        retry_prompt = mock_llm.invoke.call_args_list[1][0][0][-1].content
        assert "SecurityError" in retry_prompt
        output = result["code_outputs"][0]
        assert output["error"] is None
        assert "60" in output["stdout"]

    @patch("agents.coder.ChatGroq")
    def test_repeated_security_error_becomes_error_record(self, mock_groq_cls):
        mock_llm = _mock_llm_returning("import os", "import os")
        mock_groq_cls.return_value = mock_llm
        state = {
            "sub_tasks": ["Get the cwd"],
            "current_task_index": 0,
            "dataframe_profile": "",
            "critic_feedback": "",
        }

        result = coder_node(state, df=pd.DataFrame({"a": [1]}))

        output = result["code_outputs"][0]
        assert output["error"].startswith("SecurityError:")
        assert output["stdout"] == ""
        assert result["current_task_index"] == 1

    @patch("agents.coder.ChatGroq")
    def test_security_error_with_critic_feedback_does_not_raise(self, mock_groq_cls):
        # With critic feedback there is no internal retry; the error record
        # must still be returned rather than raised.
        mock_llm = _mock_llm_returning("np.load('x.npy')")
        mock_groq_cls.return_value = mock_llm
        state = {
            "sub_tasks": ["Load data"],
            "current_task_index": 0,
            "dataframe_profile": "",
            "critic_feedback": "Recalculate",
        }

        result = coder_node(state, df=pd.DataFrame({"a": [1]}))

        assert mock_llm.invoke.call_count == 1
        assert result["code_outputs"][0]["error"].startswith("SecurityError:")


# ── Token Accounting ───────────────────────────────────────────────

class TestCoderTokenUsage:
    @patch("agents.coder.ChatGroq")
    def test_tokens_include_initial_attempt_and_retry(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(
                content='print(df["missing"].sum())',  # fails -> retry
                usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            ),
            AIMessage(
                content='print(df["a"].sum())',
                usage_metadata={"input_tokens": 150, "output_tokens": 15, "total_tokens": 165},
            ),
        ]
        mock_groq_cls.return_value = mock_llm
        state = {
            "sub_tasks": ["Sum column a"],
            "current_task_index": 0,
            "dataframe_profile": "a (int64)",
            "critic_feedback": "",
        }

        result = coder_node(state, df=pd.DataFrame({"a": [1, 2]}))

        entry = result["trace_log"][0]
        assert entry["input_tokens"] == 250
        assert entry["output_tokens"] == 35
        assert entry["tokens_used"] == 285
        assert "_tokens_used" not in result


# ── Attempt Boundaries ─────────────────────────────────────────────

class TestCoderAttemptBoundary:
    @patch("agents.coder.ChatGroq")
    def test_first_task_starts_new_attempt(self, mock_groq_cls):
        from agents.state import NewAttempt

        mock_groq_cls.return_value = _mock_llm_returning('print("a")', 'print("b")')
        state = {
            "sub_tasks": ["Task 1", "Task 2"],
            "current_task_index": 0,
            "dataframe_profile": "",
            "critic_feedback": "",
        }
        df = pd.DataFrame({"x": [1]})

        first = coder_node(state, df=df)["code_outputs"]
        second = coder_node({**state, "current_task_index": 1}, df=df)["code_outputs"]

        assert isinstance(first, NewAttempt) and first[0]["task_index"] == 0
        assert not isinstance(second, NewAttempt) and second[0]["task_index"] == 1
        # Still a plain list of records for readers of node updates (the UI)
        assert [co["task"] for co in first + second] == ["Task 1", "Task 2"]


# ── Deterministic Core Tools ───────────────────────────────────────

import json  # noqa: E402

import agents.llm  # noqa: E402
from agents.coder import CODER_SYSTEM_PROMPT  # noqa: E402
from core.datasets import DatasetRegistry  # noqa: E402


@pytest.fixture
def reset_core_tools_flag():
    agents.llm._CORE_TOOLS = None
    yield
    agents.llm._CORE_TOOLS = None


@pytest.fixture
def sales_registry():
    registry = DatasetRegistry()
    info = registry.register_dataframe(
        pd.DataFrame({
            "category": ["A", "B", "A", "C", "B"],
            "revenue": [10.0, 20.0, 30.0, 5.0, 1.0],
            "units": [1, 2, 3, 4, 5],
        }),
        "sales",
    )
    return registry, info.dataset_id


def _state(task="Sum revenue by category", critic_feedback=""):
    return {
        "sub_tasks": [task],
        "current_task_index": 0,
        "dataframe_profile": "category (object), revenue (float64), units (int64)",
        "critic_feedback": critic_feedback,
    }


def _tool_json(tool, **arguments):
    return json.dumps({"tool": tool, "arguments": arguments})


def _system_prompt_sent(mock_llm, call=0):
    return mock_llm.invoke.call_args_list[call][0][0][0].content


def _human_prompt_sent(mock_llm, call=0):
    return mock_llm.invoke.call_args_list[call][0][0][-1].content


@pytest.mark.usefixtures("reset_core_tools_flag")
class TestCoderCoreTools:
    @patch("agents.coder.ChatGroq")
    def test_prompt_lists_tools_only_with_registry(self, mock_groq_cls, sales_registry):
        registry, dataset_id = sales_registry
        mock_llm = _mock_llm_returning('print("x")', 'print("x")')
        mock_groq_cls.return_value = mock_llm

        coder_node(_state(), df=registry.get_dataframe(dataset_id))
        assert _system_prompt_sent(mock_llm, 0) == CODER_SYSTEM_PROMPT
        assert _human_prompt_sent(mock_llm, 0).endswith(
            "\nWrite the Python code (no markdown, no explanation):"
        )

        coder_node(_state(), df=registry.get_dataframe(dataset_id),
                   registry=registry, dataset_id=dataset_id)
        with_tools = _system_prompt_sent(mock_llm, 1)
        assert with_tools.startswith(CODER_SYSTEM_PROMPT)
        assert "group_aggregate" in with_tools
        assert "the dataset is the table sales" in with_tools
        assert "<TABLE>" not in with_tools
        # The closing instruction must not ask for code only
        assert _human_prompt_sent(mock_llm, 1).endswith(
            "\nReply with ONE JSON tool call if a tool fits the task, otherwise the "
            "Python code (no markdown, no explanation):"
        )

    @patch("agents.coder.run_code")
    @patch("agents.coder.ChatGroq")
    def test_tool_call_runs_in_core_not_sandbox(self, mock_groq_cls, mock_run_code, sales_registry):
        registry, dataset_id = sales_registry
        mock_groq_cls.return_value = _mock_llm_returning(
            _tool_json("group_aggregate", group_by="category", agg="sum", value_column="revenue")
        )

        result = coder_node(_state(), registry=registry, dataset_id=dataset_id)

        mock_run_code.assert_not_called()
        output = result["code_outputs"][0]
        assert output["tool"] == "group_aggregate"
        assert output["error"] is None
        assert output["stdout"].startswith("Result of group_aggregate:")
        payload = json.loads(output["stdout"].split("\n", 1)[1])
        assert payload["rows"] == [
            {"category": "A", "sum_revenue": 40.0},
            {"category": "B", "sum_revenue": 21.0},
            {"category": "C", "sum_revenue": 5.0},
        ]
        assert result["current_task_index"] == 1

    @patch("agents.coder.ChatGroq")
    def test_fenced_json_tool_call(self, mock_groq_cls, sales_registry):
        registry, dataset_id = sales_registry
        reply = "```json\n" + _tool_json("describe", columns=["units"]) + "\n```"
        mock_groq_cls.return_value = _mock_llm_returning(reply)

        output = coder_node(_state(), registry=registry, dataset_id=dataset_id)["code_outputs"][0]

        assert output["tool"] == "describe"
        assert '"mean":3.0' in output["stdout"]

    @patch("agents.coder.ChatGroq")
    def test_chart_tool_returns_plot(self, mock_groq_cls, sales_registry):
        registry, dataset_id = sales_registry
        mock_groq_cls.return_value = _mock_llm_returning(
            _tool_json("make_chart", chart_type="bar", x="category", y="revenue")
        )

        output = coder_node(_state(), registry=registry, dataset_id=dataset_id)["code_outputs"][0]

        assert output["plot_base64"].startswith("iVBORw0KGgo")  # PNG
        assert "image_base64" not in output["stdout"]
        assert '"sum_revenue":40.0' in output["stdout"]

    @patch("agents.coder.ChatGroq")
    def test_sql_tool(self, mock_groq_cls, sales_registry):
        registry, dataset_id = sales_registry
        mock_groq_cls.return_value = _mock_llm_returning(
            _tool_json("query_sql", sql="SELECT SUM(units) AS total FROM sales WHERE revenue > 5")
        )

        output = coder_node(_state(), registry=registry, dataset_id=dataset_id)["code_outputs"][0]

        assert output["error"] is None
        assert '"rows":[[6]]' in output["stdout"]

    @patch("agents.coder.ChatGroq")
    def test_code_still_runs_in_sandbox(self, mock_groq_cls, sales_registry):
        registry, dataset_id = sales_registry
        mock_groq_cls.return_value = _mock_llm_returning('print(df["units"].sum())')

        output = coder_node(_state(), df=registry.get_dataframe(dataset_id),
                            registry=registry, dataset_id=dataset_id)["code_outputs"][0]

        assert output["tool"] is None
        assert output["error"] is None
        assert "15" in output["stdout"]

    @patch("agents.coder.ChatGroq")
    def test_tool_error_self_corrects_to_code(self, mock_groq_cls, sales_registry):
        registry, dataset_id = sales_registry
        mock_llm = _mock_llm_returning(
            _tool_json("group_aggregate", group_by="region", agg="count"),  # no such column
            'print(df.groupby("category")["revenue"].sum()["A"])',
        )
        mock_groq_cls.return_value = mock_llm

        result = coder_node(_state(), df=registry.get_dataframe(dataset_id),
                            registry=registry, dataset_id=dataset_id)

        assert mock_llm.invoke.call_count == 2
        retry_prompt = _human_prompt_sent(mock_llm, 1)
        assert "ColumnError: Unknown column(s): region" in retry_prompt
        assert retry_prompt.endswith("otherwise the Python code (no markdown, no explanation):")
        assert "group_aggregate" in _system_prompt_sent(mock_llm, 1)  # tools still offered
        output = result["code_outputs"][0]
        assert output["tool"] is None and output["error"] is None
        assert "40.0" in output["stdout"]

    @pytest.mark.parametrize(
        "reply, message",
        [
            ('{"tool": "group_aggregate", "arguments": {"agg": "sum",}}', "Invalid tool call JSON"),
            (_tool_json("run_pandas", code="print(1)"), "Unknown tool: 'run_pandas'"),
            ('{"tool": "describe", "columns": ["units"]}', "Invalid tool call"),
            (_tool_json("describe", path="../../secret.txt"), "Extra inputs are not permitted"),
            (_tool_json("query_sql", sql="DROP TABLE sales"), "QueryError"),
        ],
        ids=["malformed json", "unknown tool", "arguments at top level", "unknown argument", "unsafe sql"],
    )
    @patch("agents.coder.ChatGroq")
    def test_invalid_tool_calls_become_error_records(self, mock_groq_cls, sales_registry, reply, message):
        registry, dataset_id = sales_registry
        mock_groq_cls.return_value = _mock_llm_returning(reply, reply)

        output = coder_node(_state(), registry=registry, dataset_id=dataset_id)["code_outputs"][0]

        assert message in output["error"]
        assert output["stdout"] == ""

    @patch("agents.coder.run_tool", side_effect=RuntimeError("boom"))
    @patch("agents.coder.ChatGroq")
    def test_unexpected_tool_failure_does_not_crash(self, mock_groq_cls, _mock_run_tool, sales_registry):
        registry, dataset_id = sales_registry
        reply = _tool_json("describe")
        mock_groq_cls.return_value = _mock_llm_returning(reply, reply)

        output = coder_node(_state(), registry=registry, dataset_id=dataset_id)["code_outputs"][0]

        assert output["error"] == "Tool 'describe' failed: RuntimeError: boom"

    @patch("agents.coder.ChatGroq")
    def test_tools_use_registered_data_not_sandbox_df(self, mock_groq_cls, sales_registry):
        registry, dataset_id = sales_registry
        mock_groq_cls.return_value = _mock_llm_returning(_tool_json("describe", columns=["units"]))
        changed = pd.DataFrame({"units": [100, 200]})  # e.g. modified by earlier sandbox code

        output = coder_node(_state(), df=changed, registry=registry,
                            dataset_id=dataset_id)["code_outputs"][0]

        assert '"mean":3.0' in output["stdout"]

    @pytest.mark.parametrize("flag", ["false", "0", "off", "No"])
    @patch("agents.coder.ChatGroq")
    def test_kill_switch_restores_code_only_coder(self, mock_groq_cls, sales_registry, flag):
        registry, dataset_id = sales_registry
        mock_llm = _mock_llm_returning(_tool_json("describe"))
        mock_groq_cls.return_value = mock_llm

        with patch("agents.llm.get_env", return_value=flag):
            result = coder_node(_state(), df=registry.get_dataframe(dataset_id),
                                registry=registry, dataset_id=dataset_id)

        assert _system_prompt_sent(mock_llm) == CODER_SYSTEM_PROMPT
        assert _human_prompt_sent(mock_llm).endswith(
            "\nWrite the Python code (no markdown, no explanation):"
        )
        output = result["code_outputs"][0]
        assert output["tool"] is None  # the reply went to the sandbox
        assert output["error"] is None and output["stdout"] == ""
