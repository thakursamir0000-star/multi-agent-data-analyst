"""
Integration tests for agents/graph.py — full graph wiring.

Tests the routing logic and conditional edges without real LLM calls.
"""

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
