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
    assert len(cases) == 22 and len({c["id"] for c in cases}) == 22
    assert [c["level"] for c in cases].count("multi_step") == 7
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


DAILY = ("RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model "
         "openai/gpt-oss-120b ... on tokens per day (TPD): Limit 200000, Used 199114, Requested 1125. "
         "Please try again in 1m43.248s.'}}")
MINUTE = ("RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached ... on tokens "
          "per minute (TPM): Limit 8000, Used 7900, Requested 900. Please try again in 6.5s.'}}")


def test_rate_limit_classification():
    assert benchmark.rate_limit_kind(DAILY) == "daily"
    assert benchmark.rate_limit_kind(MINUTE) == "minute"
    assert benchmark.rate_limit_kind("ValueError: bad") is None
    assert benchmark.rate_limit_kind(None) is None
    assert benchmark.retry_after_seconds(DAILY) == pytest.approx(103.248)
    assert benchmark.retry_after_seconds(MINUTE) == pytest.approx(6.5)
    assert benchmark.retry_after_seconds("no hint") == 30.0


def _result(case, error=None, correct=True):
    return {"id": case["id"], "question": case["question"], "answer": "" if error else "ok",
            "correct": correct and not error, "verified": not error, "escalated": False, "retries": 0,
            "steps": 1, "core_tool_steps": 1, "tokens": 10, "latency_s": 1.0, "error": error}


CASES = [{"id": i, "question": f"q{i}"} for i in range(1, 6)]


def test_daily_limit_stops_the_run():
    calls = []

    def runner(case):
        calls.append(case["id"])
        return _result(case, error=DAILY if case["id"] == 3 else None)

    results, reason = benchmark.run_all(CASES, runner, sleep=lambda s: pytest.fail("no waiting on daily"))
    assert calls == [1, 2, 3] and len(results) == 3
    assert "daily token limit" in reason and "2 not run" in reason


def test_minute_limit_is_waited_out_once():
    attempts, waits = {}, []

    def runner(case):
        attempts[case["id"]] = attempts.get(case["id"], 0) + 1
        first_try_of_2 = case["id"] == 2 and attempts[2] == 1
        return _result(case, error=MINUTE if first_try_of_2 else None)

    results, reason = benchmark.run_all(CASES, runner, sleep=waits.append)
    assert reason is None and attempts[2] == 2 and waits == [pytest.approx(8.5)]
    assert all(r["error"] is None for r in results)


def test_errors_are_excluded_from_quality_metrics():
    results = [_result(CASES[0]), _result(CASES[1], correct=False), _result(CASES[2], error=DAILY)]
    s = summarize(results, planned=5)
    assert (s["questions"], s["scored"], s["errors"]) == (5, 2, 1)
    assert s["accuracy"] == 0.5  # 1 of the 2 scored, not 1 of 3
    table = benchmark.markdown(s, results, "m", "Groq's daily token limit was reached")
    assert "2 of 5 questions scored" in table and "⚠️ Groq's daily token limit" in table


def test_levels_repeats_and_cost():
    def r(i, repeat, correct, level="basic"):
        return {**_result({"id": i, "question": f"q{i}"}, correct=correct), "repeat": repeat, "level": level,
                "input_tokens": 1_000_000, "output_tokens": 100_000}

    results = [r(1, 1, True), r(2, 1, False, "multi_step"),   # run 1: 50%
               r(1, 2, True), r(2, 2, True, "multi_step")]     # run 2: 100%
    s = summarize(results, price={"input": 0.15, "output": 0.75})
    assert s["accuracy"] == 0.75 and s["accuracy_basic"] == 1.0 and s["accuracy_multi_step"] == 0.5
    assert s["accuracy_std"] == 0.25          # runs at 0.5 and 1.0
    assert s["consistent_share"] == 0.5       # q1 always right, q2 flipped
    assert s["cost_per_question_usd"] == pytest.approx(0.225)    # 1M in * 0.15 + 0.1M out * 0.75
    assert s["cost_per_correct_usd"] == pytest.approx(0.3)       # 0.9 total / 3 correct


def test_pricing_file_covers_the_priced_ci_models():
    prices = benchmark.load_prices()
    for model in ("openai/gpt-oss-120b", "openai/gpt-oss-20b"):
        assert prices[model]["input"] > 0 and prices[model]["output"] > 0


def test_a_model_without_a_list_price_shows_no_cost():
    # qwen/qwen3.8-27b is benchmarked in CI but has no price in pricing.json.
    priced = summarize([_result(CASES[0])], price={"input": 0.1, "output": 0.5})
    unpriced = summarize([_result(CASES[0])], price=None)
    reports = [{"model": m, "summary": s, "results": [], "stop_reason": None}
               for m, s in (("openai/gpt-oss-20b", priced), ("qwen/qwen3.8-27b", unpriced))]
    row = next(line for line in benchmark.report_markdown(reports).splitlines() if "`qwen/qwen3.8-27b`" in line)
    assert row.count("| – |") >= 2  # cost per question and per correct answer


def test_comparison_table_has_one_row_per_model():
    s = summarize([_result(CASES[0])], price={"input": 0.1, "output": 0.5})
    reports = [{"model": m, "summary": s, "results": [], "stop_reason": None} for m in ("a/big", "b/small")]
    text = benchmark.report_markdown(reports)
    assert text.count("| `a/big` |") == 1 and text.count("| `b/small` |") == 1
    assert "Cost / correct answer" in text
    single = benchmark.report_markdown(reports[:1])
    assert "Cost / correct answer" not in single and "**Model:** `a/big`" in single


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
