"""Tests for the agent-eval parts of evaluation/benchmark.py: the held-out suite,
step checks, failure types, run history and the regression gate (no LLM)."""

import json

import pandas as pd
import pytest

from evaluation import benchmark
from evaluation.benchmark import (
    SUITES,
    build_cases,
    build_support_cases,
    failure_type,
    filter_checks,
    gate,
    history_entry,
    history_table,
    summarize,
)


def _cases(suite):
    spec = SUITES[suite]
    return spec["cases"](pd.read_csv(spec["file"]))


# ── the held-out suite ──────────────────────────────────────────────

def test_support_suite_has_its_own_questions_and_ground_truth():
    cases = _cases("support")
    levels = [c["level"] for c in cases]
    assert len(cases) == 21 and (levels.count("basic"), levels.count("multi_step")) == (8, 13)
    assert not {c["id"] for c in cases} & {c["id"] for c in _cases("sales")}
    truth = {c["id"]: c.get("number", c.get("names")) for c in cases}
    # Pinned so a regenerated CSV can't silently change the held-out answers
    assert truth[101] == 600 and truth[109] == 28 and truth[116] == 16
    assert truth[111] == ["Shipping"] and truth[118] == ["Low"] and truth[121] == ["Billing"]
    assert truth[117][0][0] == "2024-12"


def test_every_filtered_support_question_differs_from_the_unfiltered_answer():
    d = pd.read_csv(SUITES["support"]["file"])
    by_id = {c["id"]: c for c in build_support_cases(d)}
    whole_table = {
        111: (d["reopened"] == "Yes").groupby(d["team"]).mean().idxmin(),
        113: d.loc[d["team"] == "Technical", "channel"].value_counts().idxmax(),
        118: d["priority"].value_counts().idxmax(),
        121: d.groupby("team")["resolution_hours"].mean().idxmin(),
    }
    for case_id, unfiltered in whole_table.items():
        assert by_id[case_id]["names"] != [unfiltered]
    assert all(c.get("filters") for c in by_id.values() if c["level"] == "multi_step")


def test_sales_multi_step_questions_carry_their_filters():
    multi = [c for c in build_cases(pd.read_csv(SUITES["sales"]["file"])) if c["level"] == "multi_step"]
    assert {c["id"] for c in multi if c.get("filters")} == {16, 17, 19, 20, 21, 22}


# ── step checks and failure types ───────────────────────────────────

WEST_2024 = {"filters": [["West"], ["2024"]]}


def test_filters_must_be_applied_in_one_step():
    one_step = filter_checks(WEST_2024, ["Sum units for West in 2024"],
                             ["df[(df.region == 'West') & (df.date.str[:4] == '2024')].quantity.sum()"])
    assert one_step == {"plan_has_filters": True, "code_has_filters": True, "missing_filters": []}

    # Steps don't share variables: filtering in step 1 doesn't filter step 2.
    split = filter_checks(WEST_2024, ["Filter to West", "Sum units by product"],
                          ["w = df[df.region == 'West']", "df[df.date.str[:4] == '2024'].quantity.sum()"])
    assert split["plan_has_filters"] is False and split["code_has_filters"] is False
    assert len(split["missing_filters"]) == 1


def test_comparisons_may_spread_their_periods_over_steps():
    case = {"filters": [["2024"], ["2025"]], "filters_any_step": True}
    checks = filter_checks(case, ["H1 2024 revenue", "H1 2025 revenue"], ["... '2024-06-30' ...", "... '2025-06-30' ..."])
    assert checks["code_has_filters"] is True


def test_questions_without_filters_are_not_checked():
    assert filter_checks({}, ["anything"], ["df.revenue.sum()"]) == {
        "plan_has_filters": None, "code_has_filters": None, "missing_filters": []}


def _miss(**overrides):
    base = {"correct": False, "error": None, "escalated": False, "answer": "Gadget Z sold the most.",
            "code_has_filters": None, "missing_filters": []}
    return {**base, **overrides}


@pytest.mark.parametrize("result, expected", [
    (_miss(correct=True), None),
    (_miss(error="RateLimitError: 429"), None),        # API errors are counted separately
    (_miss(escalated=True, answer=""), "asked_for_clarification"),
    (_miss(answer="  "), "no_answer"),
    (_miss(code_has_filters=False, missing_filters=["West"]), "ignored_filter"),
    (_miss(code_has_filters=False, missing_filters=["2024"]), "wrong_period"),
    (_miss(code_has_filters=False, missing_filters=["Gadgets", "2024"]), "ignored_filter"),
    (_miss(code_has_filters=True), "wrong_result"),
    (_miss(), "wrong_result"),
])
def test_failure_types(result, expected):
    assert failure_type(result) == expected


def test_run_case_records_what_each_agent_did(monkeypatch):
    class FakeGraph:
        def invoke(self, state, config):
            return {
                "draft_insight": "Accessory S sold the most units (1,461).",
                "sub_tasks": ["Filter rows to region West", "Group by product, sum quantity"],
                "code_outputs": [{"code": "w = df[df.region == 'West']", "tool": None},
                                 {"code": '{"tool": "group_aggregate"}', "tool": "group_aggregate"}],
                "trace_log": [],
            }

    import agents.graph

    monkeypatch.setattr(agents.graph, "build_graph", lambda **kwargs: FakeGraph())
    case = {"id": 21, "question": "Which product sold the most units in the West region?",
            "names": ["Gadget Z"], "filters": [["West"]]}
    result = benchmark.run_case(case, None, "ds", "profile")
    assert not result["correct"] and result["plan"][0] == "Filter rows to region West"
    assert result["step_tools"] == [None, "group_aggregate"]
    # The West filter is in step 1, so the step that applies it covers every condition
    assert result["code_has_filters"] is True and result["failure"] == "wrong_result"


def test_clarification_text_is_kept_for_the_miss_list(monkeypatch):
    class AskingGraph:
        def invoke(self, state, config):
            return {"__interrupt__": ["?"], "human_question": "Which year do you mean?"}

    import agents.graph

    monkeypatch.setattr(agents.graph, "build_graph", lambda **kwargs: AskingGraph())
    result = benchmark.run_case({"id": 11, "question": "q", "number": 1}, None, "ds", "p")
    assert result["failure"] == "asked_for_clarification" and result["clarification"] == "Which year do you mean?"
    table = benchmark.markdown(summarize([result]), [result], "m")
    assert "#11 q → asked for clarification: Which year do you mean?" in table


def _scored(i, correct, level="multi_step", **extra):
    return {"id": i, "question": f"q{i}", "answer": "x", "correct": correct, "verified": True, "escalated": False,
            "retries": 0, "steps": 1, "core_tool_steps": 0, "tokens": 10, "latency_s": 1.0, "error": None,
            "level": level, **extra}


def test_summary_counts_step_checks_and_failures():
    results = [
        _scored(1, True, "basic"),
        _scored(2, True, plan_has_filters=True, code_has_filters=True),
        _scored(3, False, plan_has_filters=False, code_has_filters=False, failure="ignored_filter",
                missing_filters=["West"]),
        _scored(4, False, plan_has_filters=True, code_has_filters=True, failure="wrong_result"),
    ]
    s = summarize(results)
    assert (s["correct"], s["correct_basic"], s["scored_multi_step"], s["correct_multi_step"]) == (2, 1, 3, 1)
    assert s["filter_questions"] == 3 and s["plan_filter_rate"] == 0.667 and s["code_filter_rate"] == 0.667
    assert s["failures"] == {"ignored_filter": 1, "wrong_result": 1}
    text = benchmark.markdown(s, results, "m")
    assert "Filters applied in one code step" in text and "ignored a filter 1" in text
    assert "**ignored a filter** (West)" in text and "**wrong result**" in text
    assert "verified_but_wrong" not in text and "| 3 | 0.667 | 0.667 | 2 |" in text


# ── history and the regression gate ─────────────────────────────────

def _report(correct, total=22, errors=0, suite="sales", model="openai/gpt-oss-120b"):
    results = [_scored(i, i <= correct) for i in range(1, total - errors + 1)]
    results += [{**_scored(i, False), "error": "RateLimitError"} for i in range(total - errors + 1, total + 1)]
    return {"model": model, "suite": suite, "summary": summarize(results, planned=total), "results": results}


def test_history_round_trip_and_trend_table(tmp_path):
    path = tmp_path / "history.jsonl"
    entry = history_entry(_report(18), label="after", commit="abc1234", date="2026-11-20")
    benchmark.append_history([entry], path)
    benchmark.append_history([history_entry(_report(8, errors=7), date="2026-11-21")], path)
    rows = benchmark.load_history(path)
    assert rows[0]["label"] == "after" and rows[0]["correct"] == 18 and rows[1]["errors"] == 7
    table = history_table(rows)
    assert "| 2026-11-20 | abc1234 | after | sales | `openai/gpt-oss-120b` | 18 / 22 |" in table
    assert "| 8 / 15 |" in table  # scored questions only; the 7 errors have their own column


def test_committed_history_and_baseline_are_valid():
    rows = benchmark.load_history()
    assert [r["correct"] for r in rows[:2]] == [17, 18]
    assert "sales" in history_table(rows)
    baseline = benchmark.load_baseline()
    assert baseline["baselines"]["sales:openai/gpt-oss-120b"]["accuracy"] == pytest.approx(17 / 22, abs=0.001)


BASELINE = {"max_drop": 0.1, "min_scored_share": 0.9,
            "baselines": {"sales:openai/gpt-oss-120b": {"accuracy": 0.773}}}


def test_gate_passes_within_tolerance_and_fails_on_a_drop():
    failures, notes = gate([_report(15)], BASELINE)         # 0.682 ≥ 0.673
    assert failures == [] and "≥ 0.673" in notes[0]
    failures, _ = gate([_report(14)], BASELINE)             # 0.636
    assert failures and "below 0.673" in failures[0]


def test_gate_fails_a_run_with_too_many_api_errors():
    # The 2026-11-14 run: 7 of 22 questions hit rate limits; its accuracy can't be compared
    failures, _ = gate([_report(11, errors=7)], BASELINE)
    assert failures and "only 15 of 22 questions were scored" in failures[0]


def test_gate_skips_suites_and_models_without_a_baseline():
    failures, notes = gate([_report(5, suite="support"), _report(5, model="openai/gpt-oss-20b")], BASELINE)
    assert failures == [] and len(notes) == 2 and all("no baseline yet" in n for n in notes)


def test_api_errors_print_as_errors_not_misses(capsys):
    daily = "RateLimitError: 429 ... tokens per day (TPD): Limit 200000"
    case = {"id": 106, "question": "q"}
    benchmark.run_all([case], lambda c: {**_scored(106, False), "error": daily}, sleep=lambda s: None)
    out = capsys.readouterr().out
    assert "#106 ERR" in out and "tokens per day" in out and "MISS" not in out


def test_history_entry_is_json_serializable():
    json.dumps(history_entry(_report(20), commit=benchmark.current_commit()))


# ── saving progress and resuming ────────────────────────────────────

def test_each_finished_question_is_saved_as_it_completes(tmp_path):
    path = benchmark.progress_path("support", "gemini-4-hydrogen", "before", 1, root=tmp_path)
    assert path.name == "support__gemini-4-hydrogen__before__x1.jsonl"
    saved = []
    benchmark.run_all([{"id": 1, "question": "q"}, {"id": 2, "question": "q"}],
                      lambda c: _scored(c["id"], True),
                      on_result=lambda r: (saved.append(r["id"]), benchmark.save_progress(path, r)))
    assert saved == [1, 2] and len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_resume_reuses_finished_questions_and_retries_errors(tmp_path):
    path = tmp_path / "run.jsonl"
    benchmark.save_progress(path, {**_scored(1, True), "repeat": 1})
    benchmark.save_progress(path, {**_scored(2, False), "repeat": 1, "error": "RateLimitError 429 PerDay"})
    benchmark.save_progress(path, {**_scored(3, False), "repeat": 1})
    benchmark.save_progress(path, {**_scored(3, True), "repeat": 1})   # a later line wins
    done = benchmark.start_progress(path, resume=True)
    assert set(done) == {(1, 1), (3, 1)} and done[(3, 1)]["correct"] is True  # #2 must run again


def test_a_fresh_run_keeps_the_old_progress_file(tmp_path):
    path = tmp_path / "run.jsonl"
    benchmark.save_progress(path, _scored(1, True))
    assert benchmark.start_progress(path, resume=False) == {}
    assert not path.exists() and len(list(tmp_path.glob("run.*.old"))) == 1


def test_request_counts_are_summarized():
    results = [{**_scored(1, True), "requests": 4, "wait_s": 0.0},
               {**_scored(2, False), "requests": 6, "wait_s": 12.0},
               {**_scored(3, False), "requests": 2, "error": "RateLimitError 429"}]
    s = summarize(results)
    assert s["requests"] == 12  # failed requests count toward a free quota too
    assert s["requests_per_question"] == 5.0 and s["wait_s_per_question"] == 6.0
    assert "LLM requests: 12 (5.0 per scored question)" in benchmark.markdown(s, results, "m")


def test_openrouter_billing_and_providers_are_summarized():
    results = [{**_scored(1, True), "requests": 5, "billed_usd": 0.0004, "served_by": {"PrimeIntellect": 5}},
               {**_scored(2, False), "requests": 4, "billed_usd": 0.0003, "served_by": {"PrimeIntellect": 4}}]
    s = summarize(results)
    assert s["billed_usd"] == pytest.approx(0.0007) and s["served_by"] == {"PrimeIntellect": 9}
    assert "Answered by (as reported by OpenRouter): PrimeIntellect 9; billed $0.000700" in benchmark.markdown(s, results, "m")
    assert "Answered by" not in benchmark.markdown(summarize([_scored(3, True)]), [], "m")  # Groq / Gemini runs


def test_openrouter_has_its_own_price_entry():
    prices = benchmark.load_prices()
    assert prices["openrouter:openai/gpt-oss-120b"] == {"input": 0.035, "output": 0.075}
    assert prices["openai/gpt-oss-120b"]["input"] == 0.15  # Groq's entry is unchanged
