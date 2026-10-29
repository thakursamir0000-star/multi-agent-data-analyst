"""
Regression tests for app/main.py — the Streamlit UI, via streamlit's AppTest.

AppTest cannot drive st.file_uploader, so the "dataset loaded" state is set
up with the same DatasetRegistry calls the upload branch makes.
"""

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from core.datasets import DatasetRegistry

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app" / "main.py")


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def reset_core_tools_flag():
    import agents.llm

    agents.llm._CORE_TOOLS = None
    yield
    agents.llm._CORE_TOOLS = None


def _loaded_app() -> AppTest:
    """An AppTest whose session already holds the sample dataset."""
    registry = DatasetRegistry(data_dir=ROOT / "sample_data")
    info = registry.register_file("sales_sample.csv")
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["registry"] = registry
    at.session_state["dataset_id"] = info.dataset_id
    at.session_state["df"] = registry.get_dataframe(info.dataset_id)
    at.session_state["profile"] = registry.get_profile(info.dataset_id)
    at.session_state["profile_text"] = registry.get_profile_text(info.dataset_id)
    return at


def _llm(content: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=content)
    return llm


class TestApp:
    def test_starts_without_dataset(self):
        at = AppTest.from_file(APP, default_timeout=60).run()
        assert not at.exception
        assert any("Upload a CSV or Excel file" in info.value for info in at.info)
        assert at.session_state["registry"] is None
        assert at.session_state["dataset_id"] is None

    def test_dataset_loaded_shows_chat(self):
        at = _loaded_app().run()
        assert not at.exception
        assert any("Ask a Question" in md.value for md in at.markdown)
        assert len(at.chat_input) == 1
        assert any("500" in md.value and "Rows" in md.value for md in at.markdown)

    def test_question_runs_graph_with_core_tool(self):
        tool_call = json.dumps({
            "tool": "make_chart",
            "arguments": {"chart_type": "bar", "x": "category", "y": "revenue"},
        })
        coder_llm = _llm(tool_call)
        llms = {
            "agents.planner.ChatGroq": _llm('["Chart revenue by category"]'),
            "agents.coder.ChatGroq": coder_llm,
            "agents.analyst.ChatGroq": _llm("Services leads revenue."),
            "agents.critic.ChatGroq": _llm(
                '{"verdict": "PASS", "issues": [], "suggestion": "", "is_ambiguous": false}'
            ),
        }
        at = _loaded_app().run()
        with ExitStack() as stack:
            for target, llm in llms.items():
                stack.enter_context(patch(target, return_value=llm))
            stack.enter_context(patch("tools.observability._write_to_disk"))
            at.chat_input[0].set_value("Revenue by category?").run()

        assert not at.exception
        history = at.session_state["chat_history"]
        assert history[0] == {"role": "user", "content": "Revenue by category?"}
        assert history[1] == {"role": "assistant", "content": "Services leads revenue."}
        assert history[2]["content"] == "__PLOT__"
        assert history[2]["plot_base64"].startswith("iVBORw0KGgo")  # PNG from make_chart
        # The Coder was offered the tools, with the registry's table name
        system_prompt = coder_llm.invoke.call_args[0][0][0].content
        assert "the dataset is the table sales_sample" in system_prompt
        output = at.session_state["graph_state"]["code_outputs"][0]
        assert output["tool"] == "make_chart" and output["error"] is None
