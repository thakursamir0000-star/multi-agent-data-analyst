"""
Integration tests for agents/graph.py — full graph wiring.

Tests the routing logic and conditional edges without real LLM calls.
"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from agents.graph import route_after_coder, route_after_critic


# ── Routing Logic Tests ─────────────────────────────────────────────

class TestRouteAfterCoder:
    def test_more_tasks_routes_to_coder(self):
        state = {"sub_tasks": ["Task 1", "Task 2"], "current_task_index": 1}
        assert route_after_coder(state) == "coder"

    def test_all_tasks_done_routes_to_analyst(self):
        state = {"sub_tasks": ["Task 1", "Task 2"], "current_task_index": 2}
        assert route_after_coder(state) == "analyst"

    def test_empty_tasks_routes_to_analyst(self):
        state = {"sub_tasks": [], "current_task_index": 0}
        assert route_after_coder(state) == "analyst"


class TestRouteAfterCritic:
    def test_pass_routes_to_end(self):
        state = {
            "verified_insight": "All good",
            "needs_human_input": False,
        }
        assert route_after_critic(state) == "end"

    def test_fail_with_retries_routes_to_coder(self):
        state = {
            "verified_insight": "",
            "needs_human_input": False,
            "critic_feedback": "Fix the numbers",
            "retry_count": 1,
        }
        assert route_after_critic(state) == "coder"

    def test_escalate_routes_to_human_input(self):
        state = {
            "verified_insight": "",
            "needs_human_input": True,
            "human_question": "Please clarify",
        }
        assert route_after_critic(state) == "human_input"

    def test_empty_verified_insight_is_falsy(self):
        """Empty string for verified_insight should NOT route to end."""
        state = {
            "verified_insight": "",
            "needs_human_input": False,
            "critic_feedback": "Retry",
        }
        assert route_after_critic(state) != "end"


# ── Graph Construction Tests ────────────────────────────────────────

class TestGraphConstruction:
    def test_graph_builds_without_error(self):
        from agents.graph import build_graph

        graph = build_graph()
        assert graph is not None

    def test_initial_state_creation(self):
        from agents.graph import create_initial_state

        state = create_initial_state(
            user_query="Test query",
            dataframe_profile="Test profile",
        )
        assert state["user_query"] == "Test query"
        assert state["dataframe_profile"] == "Test profile"
        assert state["sub_tasks"] == []
        assert state["retry_count"] == 0
        assert state["needs_human_input"] is False
        assert state["code_outputs"] == []
        assert state["trace_log"] == []


# ── End-to-End Run (Mocked LLMs) ────────────────────────────────────

def _llm_returning(content: str, input_tokens: int, output_tokens: int) -> MagicMock:
    """A mock LLM whose invoke() returns an AIMessage with usage metadata."""
    from langchain_core.messages import AIMessage

    llm = MagicMock()
    llm.invoke.return_value = AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )
    return llm


class TestGraphEndToEnd:
    def test_full_run_records_tokens_without_leaking_private_key(self):
        import pandas as pd

        from agents.graph import build_graph, create_initial_state

        df = pd.DataFrame({"a": [10, 20, 30]})
        llms = {
            "agents.planner.ChatGroq": _llm_returning('["Sum column \'a\'"]', 50, 10),
            "agents.coder.ChatGroq": _llm_returning('print(df["a"].sum())', 80, 12),
            "agents.analyst.ChatGroq": _llm_returning("The total of a is 60.", 90, 20),
            "agents.critic.ChatGroq": _llm_returning(
                '{"verdict": "PASS", "issues": [], "suggestion": "", "is_ambiguous": false}',
                70, 15,
            ),
        }
        with ExitStack() as stack:
            for target, llm in llms.items():
                stack.enter_context(patch(target, return_value=llm))
            stack.enter_context(patch("tools.observability._write_to_disk"))

            graph = build_graph(df=df)
            final = graph.invoke(
                create_initial_state("What is the total of a?", "a (int64)"),
                {"configurable": {"thread_id": "test-e2e"}},
            )

        assert final["verified_insight"] == "The total of a is 60."
        assert "60" in final["code_outputs"][0]["stdout"]
        assert "_tokens_used" not in final

        tokens = {e["node_name"]: e["tokens_used"] for e in final["trace_log"]}
        assert tokens == {"Planner": 60, "Coder": 92, "Analyst": 110, "Critic": 85}


# ── Retry Isolation ─────────────────────────────────────────────────

def _llm_sequence(*contents: str) -> MagicMock:
    """A mock LLM whose successive invoke() calls return these contents."""
    from langchain_core.messages import AIMessage

    llm = MagicMock()
    llm.invoke.side_effect = [AIMessage(content=c) for c in contents]
    return llm


_FAIL = (
    '{"verdict": "FAIL", "issues": ["Totals look wrong"], '
    '"suggestion": "Recompute the totals", "is_ambiguous": false}'
)
_FAIL_AMBIGUOUS = (
    '{"verdict": "FAIL", "issues": ["Which year?"], '
    '"suggestion": "", "is_ambiguous": true}'
)
_PASS = '{"verdict": "PASS", "issues": [], "suggestion": "", "is_ambiguous": false}'


class TestRetryIsolation:
    """Each attempt (plan or Critic retry) must see only its own task outputs."""

    def test_code_outputs_reducer(self):
        from agents.state import NewAttempt, _append_or_new_attempt

        old = [{"task_index": 0}, {"task_index": 1}]
        assert _append_or_new_attempt(old, [{"task_index": 2}]) == old + [{"task_index": 2}]
        replaced = _append_or_new_attempt(old, NewAttempt([{"task_index": 0, "new": True}]))
        assert replaced == [{"task_index": 0, "new": True}]
        assert type(replaced) is list  # the marker type does not persist in state
        assert old == [{"task_index": 0}, {"task_index": 1}]  # not mutated

    def _patched(self, stack, planner, coder, analyst, critic):
        llms = {
            "agents.planner.ChatGroq": planner,
            "agents.coder.ChatGroq": coder,
            "agents.analyst.ChatGroq": analyst,
            "agents.critic.ChatGroq": critic,
        }
        for target, llm in llms.items():
            stack.enter_context(patch(target, return_value=llm))
        stack.enter_context(patch("tools.observability._write_to_disk"))
        stack.enter_context(patch("agents.critic.get_max_retries", return_value=2))

    @staticmethod
    def _prompt(llm, call):
        return llm.invoke.call_args_list[call][0][0][-1].content

    def test_critic_retry_outputs_are_isolated(self):
        import pandas as pd

        from agents.graph import build_graph, create_initial_state

        planner = _llm_sequence('["Task one sums a", "Task two sums b"]')
        coder = _llm_sequence(
            'print("attempt1-task1")', 'print("attempt1-task2")',
            'print("attempt2-task1")', 'print("attempt2-task2")',
        )
        analyst = _llm_sequence("Draft one.", "Draft two.")
        critic = _llm_sequence(_FAIL, _PASS)

        with ExitStack() as stack:
            self._patched(stack, planner, coder, analyst, critic)
            graph = build_graph(df=pd.DataFrame({"a": [1], "b": [2]}))
            final = graph.invoke(
                create_initial_state("Sum a and b", "a (int64), b (int64)"),
                {"configurable": {"thread_id": "test-retry-isolation"}},
            )

        # Attempt 1 ran both tasks and the Analyst saw both outputs
        first_analysis = self._prompt(analyst, 0)
        assert "attempt1-task1" in first_analysis and "attempt1-task2" in first_analysis

        # The retry: Analyst and Critic see only attempt-2 outputs
        retry_analysis = self._prompt(analyst, 1)
        assert "attempt2-task1" in retry_analysis and "attempt2-task2" in retry_analysis
        assert "attempt1" not in retry_analysis
        assert retry_analysis.count("### Sub-task") == 2
        retry_verification = self._prompt(critic, 1)
        assert "attempt2-task1" in retry_verification and "attempt1" not in retry_verification

        # Final state holds exactly the retry's outputs, once each, in order
        outputs = final["code_outputs"]
        assert [o["stdout"].strip() for o in outputs] == ["attempt2-task1", "attempt2-task2"]
        assert [o["task_index"] for o in outputs] == [0, 1]

        # Retry context survived: the retry's first task got the Critic's
        # feedback, and the retry was counted
        assert "## Critic Feedback (fix this)\nRecompute the totals" in self._prompt(coder, 2)
        assert final["retry_count"] == 1
        assert final["verified_insight"] == "Draft two."

    def test_replan_after_human_input_outputs_are_isolated(self):
        import pandas as pd
        from langgraph.types import Command

        from agents.graph import build_graph, create_initial_state

        planner = _llm_sequence('["Task one sums a", "Task two sums b"]', '["Task three sums a for 2024"]')
        coder = _llm_sequence(
            'print("plan1-task1")', 'print("plan1-task2")', 'print("plan2-task1")',
        )
        analyst = _llm_sequence("Draft one.", "Draft two.")
        critic = _llm_sequence(_FAIL_AMBIGUOUS, _PASS)
        config = {"configurable": {"thread_id": "test-replan-isolation"}}

        with ExitStack() as stack:
            self._patched(stack, planner, coder, analyst, critic)
            graph = build_graph(df=pd.DataFrame({"a": [1], "b": [2]}))
            graph.invoke(create_initial_state("Sum a and b", "a, b"), config)
            assert graph.get_state(config).next == ("human_input",)
            final = graph.invoke(Command(resume="Only 2024"), config)

        replan_analysis = self._prompt(analyst, 1)
        assert "plan2-task1" in replan_analysis and "plan1" not in replan_analysis
        assert [o["stdout"].strip() for o in final["code_outputs"]] == ["plan2-task1"]
        assert final["verified_insight"] == "Draft two."


# ── Graph with the Core Layer ───────────────────────────────────────

class TestGraphWithRegistry:
    @pytest.fixture
    def registry_and_id(self):
        import pandas as pd

        from core.datasets import DatasetRegistry

        registry = DatasetRegistry()
        info = registry.register_dataframe(
            pd.DataFrame({"category": ["A", "B", "A"], "revenue": [10.0, 20.0, 30.0]}), "sales"
        )
        return registry, info.dataset_id

    def test_builds_with_registry(self, registry_and_id):
        from agents.graph import build_graph

        registry, dataset_id = registry_and_id
        assert build_graph(registry=registry, dataset_id=dataset_id) is not None

    def test_registry_and_id_required_together(self, registry_and_id):
        from agents.graph import build_graph

        registry, dataset_id = registry_and_id
        with pytest.raises(ValueError, match="together"):
            build_graph(registry=registry)
        with pytest.raises(ValueError, match="together"):
            build_graph(dataset_id=dataset_id)

    @pytest.mark.parametrize("bad_id", ["ds_0000000000000000", "../../secret.txt"])
    def test_unknown_dataset_fails_fast(self, registry_and_id, bad_id):
        from agents.graph import build_graph
        from core.errors import DatasetNotFoundError

        registry, _ = registry_and_id
        with pytest.raises(DatasetNotFoundError):
            build_graph(registry=registry, dataset_id=bad_id)

    def _run(self, registry, dataset_id, coder_reply):
        from agents.graph import build_graph, create_initial_state

        coder_llm = _llm_returning(coder_reply, 80, 12)
        llms = {
            "agents.planner.ChatGroq": _llm_returning('["Sum revenue by category"]', 50, 10),
            "agents.coder.ChatGroq": coder_llm,
            "agents.analyst.ChatGroq": _llm_returning("A has 40 in revenue.", 90, 20),
            "agents.critic.ChatGroq": _llm_returning(
                '{"verdict": "PASS", "issues": [], "suggestion": "", "is_ambiguous": false}',
                70, 15,
            ),
        }
        with ExitStack() as stack:
            for target, llm in llms.items():
                stack.enter_context(patch(target, return_value=llm))
            stack.enter_context(patch("tools.observability._write_to_disk"))
            stack.enter_context(patch("agents.llm._CORE_TOOLS", None))

            graph = build_graph(registry=registry, dataset_id=dataset_id)
            final = graph.invoke(
                create_initial_state("Revenue by category?", "category, revenue"),
                {"configurable": {"thread_id": "test-core"}},
            )
        return final, coder_llm

    def test_end_to_end_with_core_tool(self, registry_and_id):
        import json

        registry, dataset_id = registry_and_id
        reply = json.dumps({
            "tool": "group_aggregate",
            "arguments": {"group_by": "category", "agg": "sum", "value_column": "revenue"},
        })

        final, coder_llm = self._run(registry, dataset_id, reply)

        output = final["code_outputs"][0]
        assert output["tool"] == "group_aggregate"
        assert '{"category":"A","sum_revenue":40.0}' in output["stdout"]
        assert final["verified_insight"] == "A has 40 in revenue."
        system_prompt = coder_llm.invoke.call_args[0][0][0].content
        assert "the dataset is the table sales" in system_prompt

    def test_end_to_end_sandbox_fallback_uses_registered_data(self, registry_and_id):
        registry, dataset_id = registry_and_id

        final, _ = self._run(registry, dataset_id, 'print(df["revenue"].sum())')

        output = final["code_outputs"][0]
        assert output["tool"] is None and output["error"] is None
        assert "60.0" in output["stdout"]  # the sandbox df came from the registry
