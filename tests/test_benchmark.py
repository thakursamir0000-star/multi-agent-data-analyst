"""Tests for evaluation/benchmark.py — ground truth, answer matching and scoring (no LLM)."""

import math

import pandas as pd
import pytest

from evaluation import benchmark
from evaluation.benchmark import (
    DATA_FILE,
    build_cases,
    is_correct,
    matches_names,
    matches_number,
    numbers_in,
    summarize,
)


def test_ground_truth_is_computed_for_every_question():
    cases = build_cases(pd.read_csv(DATA_FILE))
    assert len(cases) == 15 and len({c["id"] for c in cases}) == 15
    for case in cases:
        assert ("number" in case) != ("names" in case)
        if "number" in case:
            assert math.isfinite(float(case["number"]))
        else:
            assert case["names"] and all(case["names"])


def test_numbers_in_reads_commas_scales_and_minus_signs():
    assert numbers_in("Revenue was $776,013.95 (about 776K) and −0.06") == [
        (776013.95, 2, False), (776000.0, 0, True), (-0.06, 2, False),
    ]
    assert numbers_in("1.2 million units")[0][:2] == (1_200_000.0, 1)


@pytest.mark.parametrize("text, expected, kwargs, ok", [
    ("Total revenue is $776,013.95.", 776013.95, {}, True),
    ("Total revenue is about $776K.", 776013.95, {}, True),
    ("Total revenue is $776,014.", 776013.95, {}, True),
    ("Total revenue is $770,000.", 776013.95, {}, False),
    ("The average discount is 9.3%.", 0.0932377, {"ratio": True}, True),
    ("The average discount is 0.09.", 0.0932377, {"ratio": True}, True),
    ("The average discount is 0.", 0.0932377, {"ratio": True}, False),  # a stray 0 is not an answer
    ("A weak negative correlation (r = -0.058).", -0.0578889, {"absolute": True}, True),
    ("A weak correlation of 0.06.", -0.0578889, {"absolute": True}, True),
    ("There are 500 transactions.", 500, {}, True),
    ("There are 499 transactions.", 500, {}, False),
])
def test_matches_number(text, expected, kwargs, ok):
    assert matches_number(text, expected, **kwargs) is ok


def test_matches_names():
    assert matches_names("Top: Service Premium, Gadget Z and Service Basic", ["Service Premium", "Gadget Z", "Service Basic"])
    assert not matches_names("Top: Service Premium and Gadget Z", ["Service Premium", "Gadget Z", "Service Basic"])
    assert matches_names("The best month was January 2025.", [["2025-01", "January 2025"]])


def test_is_correct_dispatches_on_case_type():
    assert is_correct({"number": 11}, "There are 11 distinct products.")
    assert is_correct({"names": ["North"]}, "The **North** region leads.")


def test_run_case_scores_a_fake_pipeline(monkeypatch):
    class FakeGraph:
        def invoke(self, state, config):
            return {
                "verified_insight": "There are 11 distinct products.",
                "retry_count": 1,
                "code_outputs": [{"tool": "describe"}, {"tool": None}],
                "trace_log": [{"tokens_used": 120}, {"tokens_used": 80}],
            }

    import agents.graph

    monkeypatch.setattr(agents.graph, "build_graph", lambda **kwargs: FakeGraph())
    result = benchmark.run_case({"id": 9, "question": "How many products?", "number": 11}, None, "ds", "profile")
    assert result["correct"] and result["verified"] and not result["escalated"]
    assert (result["retries"], result["steps"], result["core_tool_steps"], result["tokens"]) == (1, 2, 1, 200)


def test_clarification_and_errors_count_as_misses(monkeypatch):
    import agents.graph

    class AskingGraph:
        def invoke(self, state, config):
            return {"__interrupt__": ["Which column?"], "draft_insight": "11"}

    monkeypatch.setattr(agents.graph, "build_graph", lambda **kwargs: AskingGraph())
    asked = benchmark.run_case({"id": 1, "question": "q", "number": 11}, None, "ds", "p")
    assert asked["escalated"] and not asked["correct"]

    def broken(**kwargs):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(agents.graph, "build_graph", broken)
    failed = benchmark.run_case({"id": 2, "question": "q", "number": 11}, None, "ds", "p")
    assert failed["error"] == "RuntimeError: rate limited" and not failed["correct"]


def test_summary_metrics():
    base = {"answer": "", "escalated": False, "retries": 0, "steps": 2, "core_tool_steps": 1,
            "tokens": 100, "latency_s": 2.0, "error": None}
    cases = [
        {**base, "id": 1, "question": "a", "correct": True, "verified": True},
        {**base, "id": 2, "question": "b", "correct": False, "verified": True, "retries": 2},
        {**base, "id": 3, "question": "c", "correct": True, "verified": False, "latency_s": 6.0},
        {**base, "id": 4, "question": "d", "correct": False, "verified": False, "escalated": True},
    ]
    s = summarize(cases)
    assert s["accuracy"] == 0.5 and s["verified_rate"] == 0.5 and s["verified_but_wrong"] == 1
    assert s["escalation_rate"] == 0.25 and s["mean_retries"] == 0.5 and s["core_tool_share"] == 0.5
    assert s["latency_s_p50"] == 2.0 and s["tokens_per_question"] == 100
    table = benchmark.markdown(s, cases, "test-model")
    assert "Verified but wrong" in table
    assert "#2 b → (empty answer)" in table and "#4 d → asked for clarification" in table
