"""
Tests for the multi-step fixes: every agent works from the full question, so
a filter (a year, a region, a threshold) is applied in each step that needs
it and the Critic can check that it was.

The benchmark showed the failure: sub-tasks run as separate snippets on the
full table, so "filter to the West" in one step and "sum units by product" in
the next answered for every region, and the Critic, which never saw the
question or the code, approved it.
"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pandas as pd
from langchain_core.messages import AIMessage

from agents.coder import CORE_TOOLS_PROMPT, coder_node
from agents.critic import CRITIC_SYSTEM_PROMPT, _format_verification_context, critic_node
from agents.planner import PLANNER_SYSTEM_PROMPT, planner_node
from agents.state import question_with_clarification

_PASS = '{"verdict": "PASS", "issues": [], "suggestion": "", "is_ambiguous": false}'
_FAIL_FILTER = ('{"verdict": "FAIL", "issues": ["Task 2 sums every region"], '
                '"suggestion": "Filter to region == \'West\' in the task that sums units", "is_ambiguous": false}')
QUESTION = "Which product sold the most units in the West region?"


def _llm(*contents: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.side_effect = [AIMessage(content=c) for c in contents]
    return llm


def _human_prompt(llm, call=0) -> str:
    return llm.invoke.call_args_list[call][0][0][-1].content


def test_question_with_clarification():
    assert question_with_clarification({"user_query": "Revenue?"}) == "Revenue?"
    assert question_with_clarification({"user_query": "Revenue?", "human_answer": "2024 only"}) == (
        "Revenue?\n\n[Clarification from user]: 2024 only")


# ── Planner ─────────────────────────────────────────────────────────

def test_planner_is_told_that_steps_do_not_share_filters():
    assert "nothing (variables, filtered tables) carries over" in PLANNER_SYSTEM_PROMPT
    assert 'Never make "filter the rows" a\n   sub-task of its own' in PLANNER_SYSTEM_PROMPT


@patch("agents.planner.ChatGroq")
def test_a_new_plan_starts_without_old_critic_feedback(mock_groq_cls):
    mock_groq_cls.return_value = _llm('["Sum quantity by product for region West"]')
    result = planner_node({"user_query": QUESTION, "dataframe_profile": "", "human_answer": "",
                           "critic_feedback": "stale"})
    assert result["critic_feedback"] == ""


# ── Coder ───────────────────────────────────────────────────────────

def test_tool_menu_warns_that_tools_use_every_row():
    assert "always\nuse EVERY row" in CORE_TOOLS_PROMPT and "query_sql with a WHERE" in CORE_TOOLS_PROMPT


@patch("agents.coder.ChatGroq")
def test_coder_sees_the_question_and_the_clarification(mock_groq_cls):
    mock_groq_cls.return_value = llm = _llm("print(1)")
    coder_node({"user_query": QUESTION, "human_answer": "Only 2025", "sub_tasks": ["Sum quantity by product"],
                "current_task_index": 0, "dataframe_profile": "", "critic_feedback": ""},
               df=pd.DataFrame({"a": [1]}))
    prompt = _human_prompt(llm)
    assert f"## User Question\n{QUESTION}\n\n[Clarification from user]: Only 2025" in prompt
    assert "always hold the FULL dataset" in prompt
    assert prompt.index("## User Question") < prompt.index("## Task")


@patch("agents.coder.ChatGroq")
def test_coder_keeps_critic_feedback_for_the_rest_of_the_attempt(mock_groq_cls):
    mock_groq_cls.return_value = llm = _llm("print(2)")
    result = coder_node({"user_query": QUESTION, "sub_tasks": ["Filter", "Sum quantity by product"],
                         "current_task_index": 1, "dataframe_profile": "", "critic_feedback": "Filter to West"},
                        df=pd.DataFrame({"a": [1]}))
    assert "## Critic Feedback (fix this)\nFilter to West" in _human_prompt(llm)
    assert "critic_feedback" not in result  # not cleared after the first task any more


# ── Critic ──────────────────────────────────────────────────────────

def test_critic_rules_cover_filters_and_real_ambiguity():
    assert "must itself apply EVERY condition in the question" in CRITIC_SYSTEM_PROMPT
    assert 'Set "is_ambiguous" to true ONLY' in CRITIC_SYSTEM_PROMPT


def test_verification_context_shows_question_plan_and_code():
    outputs = [
        {"task": "Filter to West", "code": "w = df[df.region == 'West']", "tool": None, "stdout": "120 rows"},
        {"task": "Units by product", "code": '{"tool": "group_aggregate"}', "tool": "group_aggregate",
         "stdout": "Accessory S 1461"},
    ]
    context = _format_verification_context("Accessory S sold 1,461 units.", outputs, QUESTION,
                                           ["Filter to West", "Units by product"])
    assert context.startswith(f"## User Question\n{QUESTION}")
    assert "## Plan\n1. Filter to West\n2. Units by product" in context
    assert "code:\n```\nw = df[df.region == 'West']\n```" in context
    assert 'tool call (group_aggregate):\n```\n{"tool": "group_aggregate"}\n```' in context
    # Old callers (question and plan omitted) still get the outputs
    assert _format_verification_context("x", outputs).startswith("## Draft Insight")


@patch("agents.critic.ChatGroq")
def test_critic_node_sends_the_question_and_plan(mock_groq_cls):
    mock_groq_cls.return_value = llm = _llm(_PASS)
    critic_node({"user_query": QUESTION, "human_answer": "2025", "sub_tasks": ["Units by product, West, 2025"],
                 "draft_insight": "Gadget Z", "code_outputs": [], "retry_count": 0})
    prompt = _human_prompt(llm)
    assert "[Clarification from user]: 2025" in prompt and "1. Units by product, West, 2025" in prompt


# ── Through the graph ───────────────────────────────────────────────

def test_critic_retry_reaches_every_task_of_the_new_attempt():
    from agents.graph import build_graph, create_initial_state

    planner = _llm('["Filter rows to region West", "Group by product, sum quantity"]')
    coder = _llm("print('a1t1')", "print('a1t2')", "print('a2t1')", "print('a2t2')")
    analyst = _llm("Accessory S sold the most.", "Gadget Z sold the most.")
    critic = _llm(_FAIL_FILTER, _PASS)
    with ExitStack() as stack:
        for target, llm in {"agents.planner.ChatGroq": planner, "agents.coder.ChatGroq": coder,
                            "agents.analyst.ChatGroq": analyst, "agents.critic.ChatGroq": critic}.items():
            stack.enter_context(patch(target, return_value=llm))
        stack.enter_context(patch("tools.observability._write_to_disk"))
        stack.enter_context(patch("agents.critic.get_max_retries", return_value=2))
        final = build_graph(df=pd.DataFrame({"a": [1]})).invoke(
            create_initial_state(QUESTION, "region, product, quantity"),
            {"configurable": {"thread_id": "test-feedback-every-task"}},
        )

    feedback = "## Critic Feedback (fix this)\nFilter to region == 'West' in the task that sums units"
    # Before the fix only the retry's first task (call 2) got the feedback;
    # the second task (call 3), where the sum happens, ran without it.
    assert feedback in _human_prompt(coder, 2) and feedback in _human_prompt(coder, 3)
    assert feedback not in _human_prompt(coder, 0)
    assert all(QUESTION in _human_prompt(coder, call) for call in range(4))
    assert QUESTION in _human_prompt(critic, 0) and "print('a1t2')" in _human_prompt(critic, 0)
    assert final["verified_insight"] == "Gadget Z sold the most." and final["retry_count"] == 1
