"""
End-to-end benchmark for the multi-agent pipeline.

    python evaluation/benchmark.py                    # sales questions (needs the provider's API key)
    python evaluation/benchmark.py --suite support    # held-out support-ticket questions
    python evaluation/benchmark.py --suite support --label after --resume   # finish an interrupted run
    LLM_PROVIDER=gemini python evaluation/benchmark.py   # Gemini instead of Groq (GEMINI_API_KEY)
    python evaluation/benchmark.py --limit 5          # first 5 questions
    python evaluation/benchmark.py --level multi_step --repeats 3
    python evaluation/benchmark.py --models openai/gpt-oss-120b,openai/gpt-oss-20b
    python evaluation/benchmark.py --label after --gate   # label the run; fail on a regression
    python evaluation/benchmark.py --history          # trend table of recorded runs, no LLM
    python evaluation/benchmark.py --dry-run          # print questions + ground truth, no LLM

Two suites, each a CSV whose questions have answers computed in pandas:
- "sales" (sample_data/sales_sample.csv, 22 questions) is the development
  set: agent changes are made by studying its failures;
- "support" (sample_data/support_tickets.csv, 21 questions) is held out:
  unrelated data, written before those changes, to check they generalize.

Questions are "basic" (one aggregation or lookup) or "multi_step" (filter,
group and compare). --repeats runs each question several times and reports
the spread, because LLM answers vary between runs. --models compares models
side by side, with cost per question from evaluation/pricing.json.

Step checks: for a question that filters rows (a year, a region, a
threshold), the benchmark checks whether the Planner's sub-tasks mention every
condition and whether one code step applies all of them; steps run as
separate snippets on the full table, so a filter in one step does not carry
over to the next. Each miss gets a failure type (asked for clarification,
ignored a filter, wrong time period, wrong result, no answer); API errors are
counted separately. Every run is appended to evaluation/history.jsonl, and
--gate fails the run when accuracy drops below evaluation/baseline.json.

Rate limits: each LLM call is paced (LLM_REQUESTS_PER_MINUTE) and retried
within bounds by agents/llm.py; a question that still hits a per-minute limit
is retried with exponential backoff. A daily limit stops the run (the
remaining questions can't succeed until it resets). Every finished question
is saved to evaluation/results/progress/ as it completes, and --resume reuses
those answers, so an interrupted run never repeats a finished question. Only
finished runs go into the history, and a run in which no question could be
scored exits with an error.

The full graph (planner → coder → analyst → critic) answers
it exactly as the app and the MCP analyze_question tool do, and the final
insight counts as correct when it contains the expected number (allowing for
normal rounding, %, K/M) or every expected name.

Reported: answer accuracy (overall and per level), how often the critic
verified the answer (and how often it verified a wrong one), clarification
requests, retries, how much of the work used deterministic core tools vs
generated pandas, the step checks and failure types, latency p50/p95, tokens
and cost per question.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import unicodedata
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.llm import rate_limit_kind, request_stats, retry_after_seconds  # noqa: E402,F401
from core.datasets import DatasetRegistry  # noqa: E402

DATA_FILE = ROOT / "sample_data" / "sales_sample.csv"
SUPPORT_FILE = ROOT / "sample_data" / "support_tickets.csv"
EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
PRICING_FILE = EVAL_DIR / "pricing.json"
HISTORY_FILE = EVAL_DIR / "history.jsonl"
BASELINE_FILE = EVAL_DIR / "baseline.json"
LEVELS = ("basic", "multi_step")


# ── questions with pandas ground truth ──────────────────────────────

def build_cases(df: pd.DataFrame) -> list[dict]:
    """Questions and their exact answers, computed from the data itself."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    month = d.groupby(d["date"].dt.to_period("M"))["revenue"].sum().idxmax()
    top3 = d.groupby("product")["revenue"].sum().nlargest(3).index.tolist()
    y2025 = d[d["date"].dt.year == 2025]
    margin = d.groupby("category")["profit"].sum() / d.groupby("category")["revenue"].sum()
    h1_2024 = d.loc[d["date"].between("2024-01-01", "2024-06-30"), "revenue"].sum()
    h1_2025 = d.loc[d["date"].between("2025-01-01", "2025-06-30"), "revenue"].sum()
    gadgets_2024 = d[(d["category"] == "Gadgets") & (d["date"].dt.year == 2024)]
    basic = [
        {"id": 1, "question": "What is the total revenue?", "number": d["revenue"].sum()},
        {"id": 2, "question": "Which region has the highest total profit?",
         "names": [d.groupby("region")["profit"].sum().idxmax()]},
        {"id": 3, "question": "What is the average discount?", "number": d["discount"].mean(), "ratio": True},
        {"id": 4, "question": "How many transactions are in the dataset?", "number": len(d)},
        {"id": 5, "question": "Which product category generates the most revenue?",
         "names": [d.groupby("category")["revenue"].sum().idxmax()]},
        {"id": 6, "question": "What is the total quantity sold in the South region?",
         "number": d.loc[d["region"] == "South", "quantity"].sum(), "filters": [["South"]]},
        {"id": 7, "question": "What is the median unit price?", "number": d["unit_price"].median()},
        {"id": 8, "question": "Which product has the highest total profit?",
         "names": [d.groupby("product")["profit"].sum().idxmax()]},
        {"id": 9, "question": "How many distinct products are there?", "number": d["product"].nunique()},
        {"id": 10, "question": "What is the Pearson correlation between discount and profit?",
         "number": d["discount"].corr(d["profit"]), "absolute": True},
        {"id": 11, "question": "What was the total revenue in 2024?",
         "number": d.loc[d["date"].dt.year == 2024, "revenue"].sum(), "filters": [["2024"]]},
        {"id": 12, "question": "Which month had the highest total revenue?",
         "names": [[str(month), month.strftime("%B %Y"), month.strftime("%b %Y")]]},
        {"id": 13, "question": "What is the overall profit margin (total profit divided by total revenue)?",
         "number": d["profit"].sum() / d["revenue"].sum(), "ratio": True},
        {"id": 14, "question": "How many transactions are missing a region?", "number": d["region"].isna().sum()},
        {"id": 15, "question": "What are the top 3 products by total revenue?", "names": top3},
    ]
    multi_step = [
        {"id": 16, "question": "Which region had the highest total profit in 2025?",
         "names": [y2025.groupby("region")["profit"].sum().idxmax()], "filters": [["2025"]]},
        {"id": 17, "question": "What was the average discount on transactions with revenue above 1,000?",
         "number": d.loc[d["revenue"] > 1000, "discount"].mean(), "ratio": True,
         "filters": [["1000", "1,000", "1_000", "1e3"]]},
        {"id": 18, "question": "Which product category has the highest profit margin (total profit divided by total revenue)?",
         "names": [margin.idxmax()]},
        {"id": 19, "question": "By what percentage did total revenue change from the first half of 2024 "
                               "(January to June) to the first half of 2025?",
         "number": (h1_2025 - h1_2024) / h1_2024, "ratio": True,
         "filters": [["2024"], ["2025"]], "filters_any_step": True},
        {"id": 20, "question": "How many transactions in the North region had a discount of at least 15%?",
         "number": ((d["region"] == "North") & (d["discount"] >= 0.15)).sum(),
         "filters": [["North"], ["0.15", "15%", "15 %", "15 percent"]]},
        {"id": 21, "question": "Which product sold the most units in the West region?",
         "names": [d[d["region"] == "West"].groupby("product")["quantity"].sum().idxmax()], "filters": [["West"]]},
        {"id": 22, "question": "What is the average profit per transaction for the Gadgets category in 2024?",
         "number": gadgets_2024["profit"].mean(), "filters": [["Gadgets"], ["2024"]]},
    ]
    return [{**c, "level": "basic"} for c in basic] + [{**c, "level": "multi_step"} for c in multi_step]


def build_support_cases(df: pd.DataFrame) -> list[dict]:
    """The held-out suite: questions about sample_data/support_tickets.csv.

    Every filtered multi-step question has a different answer from the same
    question asked of the whole table, so a skipped filter shows up as a miss.
    """
    d = df.copy()
    d["created"] = pd.to_datetime(d["created"])
    year = d["created"].dt.year
    reopened = d["reopened"] == "Yes"
    in_2024 = d[year == 2024]
    month = in_2024.groupby(in_2024["created"].dt.to_period("M")).size().idxmax()
    high_urgent = d[d["priority"].isin(["High", "Urgent"])]
    q1_technical = d[(d["team"] == "Technical") & d["created"].between("2025-01-01", "2025-03-31")]
    h2_2024 = d["created"].between("2024-07-01", "2024-12-31").sum()
    h1_2025 = d["created"].between("2025-01-01", "2025-06-30").sum()
    chat_2025 = d[(d["channel"] == "Chat") & (year == 2025) & d["satisfaction"].notna()]
    phone_urgent = d[(d["channel"] == "Phone") & (d["priority"] == "Urgent")]
    under_24h = ["< 24", "<24", "lt(24)", "under 24", "less than 24", "below 24"]
    at_most_2 = ["<= 2", "<=2", "< 3", "<3", "le(2)", "lt(3)", "isin([1, 2])", "isin([1,2])",
                 "2 or lower", "2 or less", "at most 2"]
    basic = [
        {"id": 101, "question": "How many tickets are there?", "number": len(d)},
        {"id": 102, "question": "What is the average resolution time in hours?", "number": d["resolution_hours"].mean()},
        {"id": 103, "question": "Which team handled the most tickets?", "names": [d["team"].value_counts().idxmax()]},
        {"id": 104, "question": "What is the average satisfaction score?", "number": d["satisfaction"].mean()},
        {"id": 105, "question": "What share of tickets were reopened?", "number": reopened.mean(), "ratio": True},
        {"id": 106, "question": "Which channel has the highest average satisfaction score?",
         "names": [d.groupby("channel")["satisfaction"].mean().idxmax()]},
        {"id": 107, "question": "What is the median first response time in minutes?",
         "number": d["first_response_min"].median()},
        {"id": 108, "question": "How many tickets have no satisfaction score?", "number": d["satisfaction"].isna().sum()},
    ]
    multi_step = [
        {"id": 109, "question": "How many Urgent tickets were created in 2025?",
         "number": ((d["priority"] == "Urgent") & (year == 2025)).sum(), "filters": [["Urgent"], ["2025"]]},
        {"id": 110, "question": "What was the average resolution time, in hours, for Billing tickets created in 2024?",
         "number": d.loc[(d["team"] == "Billing") & (year == 2024), "resolution_hours"].mean(),
         "filters": [["Billing"], ["2024"]]},
        {"id": 111, "question": "Which team had the lowest reopen rate among High and Urgent priority tickets?",
         "names": [(high_urgent["reopened"] == "Yes").groupby(high_urgent["team"]).mean().idxmin()],
         "filters": [["High"], ["Urgent"]]},
        {"id": 112, "question": "What is the average satisfaction score of tickets resolved in under 24 hours?",
         "number": d.loc[d["resolution_hours"] < 24, "satisfaction"].mean(), "filters": [under_24h]},
        {"id": 113, "question": "Which channel received the most Technical tickets in the first quarter of 2025 "
                                "(January to March)?",
         "names": [q1_technical["channel"].value_counts().idxmax()], "filters": [["Technical"], ["2025"]]},
        {"id": 114, "question": "By what percentage did the number of tickets change from the second half of 2024 "
                                "(July to December) to the first half of 2025 (January to June)?",
         "number": (h1_2025 - h2_2024) / h2_2024, "ratio": True,
         "filters": [["2024"], ["2025"]], "filters_any_step": True},
        {"id": 115, "question": "What is the average first response time, in minutes, for Phone tickets with "
                                "Urgent priority?",
         "number": phone_urgent["first_response_min"].mean(), "filters": [["Phone"], ["Urgent"]]},
        {"id": 116, "question": "How many Shipping tickets were reopened?",
         "number": ((d["team"] == "Shipping") & reopened).sum(), "filters": [["Shipping"], ["yes"]]},
        {"id": 117, "question": "Which month of 2024 had the most tickets?",
         "names": [[str(month), month.strftime("%B %Y"), month.strftime("%b %Y")]], "filters": [["2024"]]},
        {"id": 118, "question": "Among tickets with a satisfaction score of 2 or lower, which priority level is "
                                "most common?",
         "names": [d.loc[d["satisfaction"] <= 2, "priority"].value_counts().idxmax()], "filters": [at_most_2]},
        {"id": 119, "question": "What is the average resolution time, in hours, for Email tickets that were reopened?",
         "number": d.loc[(d["channel"] == "Email") & reopened, "resolution_hours"].mean(),
         "filters": [["Email"], ["yes"]]},
        {"id": 120, "question": "Of the Chat tickets created in 2025 that have a satisfaction score, what percentage "
                                "scored 5?",
         "number": (chat_2025["satisfaction"] == 5).mean(), "ratio": True, "filters": [["Chat"], ["2025"]]},
        {"id": 121, "question": "Which team resolves Urgent tickets fastest on average?",
         "names": [d[d["priority"] == "Urgent"].groupby("team")["resolution_hours"].mean().idxmin()],
         "filters": [["Urgent"]]},
    ]
    return [{**c, "level": "basic"} for c in basic] + [{**c, "level": "multi_step"} for c in multi_step]


SUITES = {
    "sales": {"file": DATA_FILE, "table": "sales_sample", "cases": build_cases},
    "support": {"file": SUPPORT_FILE, "table": "support_tickets", "cases": build_support_cases},
}


# ── answer matching ─────────────────────────────────────────────────

_NUMBER = re.compile(r"[-−]?\d[\d,]*(?:\.\d+)?")
_SCALE = (("thousand", 1e3), ("million", 1e6), ("billion", 1e9), ("k", 1e3), ("m", 1e6), ("b", 1e9))


def numbers_in(text: str) -> list[tuple[float, int, bool]]:
    """(value, decimals written, scaled by K/M/B) for every number in the text."""
    found = []
    for m in _NUMBER.finditer(text):
        raw = m.group(0).replace("−", "-").rstrip(",")
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        after = text[m.end():m.end() + 9].lower().lstrip()
        scaled = False
        for word, factor in _SCALE:
            if after.startswith(word) and (len(word) > 1 or not after[1:2].isalpha()):
                value, scaled = value * factor, True
                break
        found.append((value, decimals, scaled))
    return found


def matches_number(text: str, expected: float, ratio: bool = False, absolute: bool = False) -> bool:
    """True if the text states `expected` (rounded as written; ratios may be given as %)."""
    targets = [float(expected)] + ([float(expected) * 100] if ratio else [])
    for value, decimals, scaled in numbers_in(text):
        for target in targets:
            v, t = (abs(value), abs(target)) if absolute else (value, target)
            if scaled and math.isclose(v, t, rel_tol=0.01):
                return True
            if not scaled and v != 0 and (round(t, decimals) == v or math.isclose(v, t, rel_tol=0.001)):
                return True
    return False


# gpt-oss often writes no-break spaces ("Service Premium"), non-breaking
# hyphens and zero-width characters, which plain substring matching misses.
_INVISIBLE = dict.fromkeys(map(ord, "­​‌‍⁠﻿"))
_HYPHENS = str.maketrans(dict.fromkeys("‐‑‒–—―−", "-"))


def fold_typography(text: str) -> str:
    """Plain-ASCII spacing and hyphens, lower case, so names can be compared."""
    text = unicodedata.normalize("NFKC", text).translate(_INVISIBLE).translate(_HYPHENS)
    return " ".join(text.split()).lower()


def matches_names(text: str, names: list) -> bool:
    """Every expected name appears; a nested list means "any of these spellings"."""
    folded = fold_typography(text)
    return all(any(fold_typography(str(alt)) in folded for alt in (n if isinstance(n, list) else [n])) for n in names)


def is_correct(case: dict, answer: str) -> bool:
    if "number" in case:
        return matches_number(answer, case["number"], case.get("ratio", False), case.get("absolute", False))
    return matches_names(answer, case["names"])


# ── step checks and failure types ───────────────────────────────────

def _mentions(text: str, alternatives: list[str]) -> bool:
    folded = text.lower()
    return any(alt.lower() in folded for alt in alternatives)


def filter_checks(case: dict, plan: list[str], step_code: list[str]) -> dict:
    """Did the plan and the code carry the question's row filters?

    A case's "filters" lists the conditions the question sets (a year, a
    region, a threshold), each as alternative spellings. The plan passes when
    its sub-tasks mention every condition. Steps run as separate snippets on
    the full table, so the code passes only when one step applies them all
    (or, with "filters_any_step", when each appears in some step: comparisons
    of two periods may compute each period separately). A proxy based on the
    text of the code, so a mention without an actual filter still passes.
    """
    groups = case.get("filters")
    if not groups:
        return {"plan_has_filters": None, "code_has_filters": None, "missing_filters": []}
    if case.get("filters_any_step"):
        covered = "\n".join(step_code)
    else:
        covered = max(step_code, key=lambda code: sum(_mentions(code, g) for g in groups), default="")
    missing = [g[0] for g in groups if not _mentions(covered, g)]
    return {
        "plan_has_filters": all(_mentions("\n".join(plan), g) for g in groups),
        "code_has_filters": not missing,
        "missing_filters": missing,
    }


_YEAR = re.compile(r"^(?:19|20)\d\d$")
FAILURE_LABELS = {
    "asked_for_clarification": "asked for clarification",
    "ignored_filter": "ignored a filter",
    "wrong_period": "wrong time period",
    "wrong_result": "wrong result",
    "no_answer": "no answer",
}


def failure_type(result: dict) -> str | None:
    """Why a scored question was missed (None if it was right or hit an API error)."""
    if result["correct"] or result.get("error"):
        return None
    if result.get("escalated"):
        return "asked_for_clarification"
    if not result.get("answer", "").strip():
        return "no_answer"
    missing = result.get("missing_filters") or []
    if result.get("code_has_filters") is False and missing:
        return "wrong_period" if all(_YEAR.match(m) for m in missing) else "ignored_filter"
    return "wrong_result"


# ── running the pipeline ────────────────────────────────────────────

def run_case(case: dict, registry: DatasetRegistry, dataset_id: str, profile_text: str) -> dict:
    from agents.graph import build_graph, create_initial_state

    stats_before = request_stats()
    start = time.perf_counter()
    try:
        graph = build_graph(registry=registry, dataset_id=dataset_id)
        final = graph.invoke(create_initial_state(case["question"], profile_text),
                             {"configurable": {"thread_id": uuid.uuid4().hex}})
        error = None
    except Exception as e:  # an API error shouldn't stop the whole benchmark
        final, error = {}, f"{type(e).__name__}: {e}"
    latency = time.perf_counter() - start
    stats = request_stats()

    escalated = bool(final.get("__interrupt__"))
    answer = "" if escalated else (final.get("verified_insight") or final.get("draft_insight") or "")
    outputs = final.get("code_outputs", [])
    trace = final.get("trace_log", [])
    plan = [str(task) for task in final.get("sub_tasks", [])]
    step_code = [str(o.get("code") or "") for o in outputs]
    result = {
        "id": case["id"],
        "level": case.get("level", "basic"),
        "question": case["question"],
        "answer": answer,
        "correct": bool(answer) and is_correct(case, answer),
        "verified": bool(final.get("verified_insight")),
        "escalated": escalated,
        "retries": final.get("retry_count", 0),
        "steps": len(outputs),
        "core_tool_steps": sum(1 for o in outputs if o.get("tool")),
        "tokens": sum(e.get("tokens_used", 0) for e in trace),
        "input_tokens": sum(e.get("input_tokens", 0) for e in trace),
        "output_tokens": sum(e.get("output_tokens", 0) for e in trace),
        "latency_s": round(latency, 2),
        # LLM requests sent (retries included) and time spent waiting on rate limits
        "requests": int(stats["requests"] - stats_before["requests"]),
        "wait_s": round(stats["wait_s"] - stats_before["wait_s"], 1),
        "error": error,
        # What each agent did, for the step checks and for reading the misses
        "plan": plan,
        "step_code": [code[:1500] for code in step_code],
        "step_tools": [o.get("tool") for o in outputs],
        "clarification": (final.get("human_question") or "")[:500] if escalated else "",
        **filter_checks(case, plan, step_code),
    }
    result["failure"] = failure_type(result)
    return result


def run_all(cases: list[dict], runner, sleep=time.sleep, max_wait: float = 240.0,
            minute_retries: int = 3, on_result=None) -> tuple[list[dict], str | None]:
    """Run every case; retry per-minute limits, stop at a daily limit.

    A question that still hits a per-minute limit after the client's own
    retries is retried up to `minute_retries` times: first after the
    provider's suggested wait, then with exponential backoff from a full
    minute (60 s, 120 s, ...), since one question can use more tokens than a
    free-tier minute allows. `on_result` is called with each finished result
    (to save progress). Returns (results, stop_reason). After a daily limit
    the remaining questions can't succeed until the quota resets, so they
    aren't attempted.
    """
    results = []
    for case in cases:
        result = runner(case)
        for attempt in range(minute_retries):
            if rate_limit_kind(result["error"]) != "minute":
                break
            if attempt == 0:
                wait = min(retry_after_seconds(result["error"]) + 2, max_wait)
            else:
                wait = min(60.0 * 2 ** (attempt - 1), max_wait)
            print(f"   per-minute rate limit; waiting {wait:.0f}s and retrying #{case['id']}")
            sleep(wait)
            result = runner(case)
        result["repeat"] = case.get("repeat", 1)
        run_label = f" (run {result['repeat']})" if result["repeat"] > 1 else ""
        status = "OK  " if result["correct"] else ("ERR " if result["error"] else "MISS")
        why = f" ({result['failure']})" if result.get("failure") else ""
        print(f"#{result['id']:>2}{run_label} {status} {result['latency_s']:>5.1f}s  {case['question']}{why}")
        if result["error"]:
            print(f"     {result['error'][:200]}")
        results.append(result)
        if on_result is not None:
            on_result(result)
        if rate_limit_kind(result["error"]) == "daily":
            skipped = len(cases) - len(results)
            return results, (f"The provider's daily limit was reached after {len(results)} question(s); "
                             f"{skipped} not run. Finish with --resume after the quota resets.")
    return results, None


# ── saving progress, so an interrupted run can resume ───────────────

PROGRESS_DIR = RESULTS_DIR / "progress"


def progress_path(suite: str, model: str, label: str, repeats: int, root: Path = PROGRESS_DIR) -> Path:
    """One file per run: suite, model, label and number of repeats."""
    slug = re.sub(r"[^A-Za-z0-9._]+", "-", f"{suite}__{model}__{label or 'run'}__x{repeats}").strip("-")
    return root / f"{slug}.jsonl"


def load_progress(path: Path) -> dict[tuple[int, int], dict]:
    """Finished questions of an earlier attempt at this run, by (id, repeat).

    Questions that ended in an API error are left out, so a resumed run
    tries them again; the latest line for a question wins.
    """
    done: dict[tuple[int, int], dict] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        result = json.loads(line)
        key = (result["id"], result.get("repeat", 1))
        if result.get("error"):
            done.pop(key, None)
        else:
            done[key] = result
    return done


def save_progress(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, default=str) + "\n")


def start_progress(path: Path, resume: bool) -> dict[tuple[int, int], dict]:
    """The finished questions to reuse; a fresh run keeps an old file as *.old."""
    if resume:
        return load_progress(path)
    if path.exists():
        path.replace(path.with_name(f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}.old"))
    return {}


def load_prices(path: Path = PRICING_FILE) -> dict[str, dict[str, float]]:
    """USD per 1M input/output tokens, per model (see pricing.json)."""
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("models", {})
    except (OSError, json.JSONDecodeError):
        return {}


def case_cost(case: dict, price: dict[str, float]) -> float:
    return (case.get("input_tokens", 0) * price["input"] + case.get("output_tokens", 0) * price["output"]) / 1e6


def summarize(cases: list[dict], planned: int | None = None, price: dict[str, float] | None = None) -> dict:
    """Quality metrics over the questions that got an answer from the pipeline;
    infrastructure errors (API/rate limits) are counted separately."""
    def mean(values, digits=3):
        return round(float(np.mean(values)), digits) if values else None

    scored = [c for c in cases if not c["error"]]
    steps = sum(c["steps"] for c in scored)
    latencies = [c["latency_s"] for c in scored]
    filtered = [c for c in scored if c.get("code_has_filters") is not None]
    summary = {
        "questions": planned if planned is not None else len(cases),
        "scored": len(scored),
        "correct": sum(bool(c["correct"]) for c in scored),
        **{f"{what}_{level}": sum(1 for c in scored if c.get("level", "basic") == level
                                  and (what == "scored" or c["correct"]))
           for level in LEVELS for what in ("scored", "correct")},
        "accuracy": mean([float(c["correct"]) for c in scored]),
        **{f"accuracy_{level}": mean([float(c["correct"]) for c in scored if c.get("level", "basic") == level])
           for level in LEVELS},
        "verified_rate": mean([float(c["verified"]) for c in scored]),
        "verified_but_wrong": sum(c["verified"] and not c["correct"] for c in scored),
        "escalation_rate": mean([float(c["escalated"]) for c in scored]),
        "escalations": sum(bool(c["escalated"]) for c in scored),
        "mean_retries": mean([c["retries"] for c in scored]),
        "core_tool_share": round(sum(c["core_tool_steps"] for c in scored) / steps, 3) if steps else None,
        "latency_s_p50": round(float(np.percentile(latencies, 50)), 2) if latencies else None,
        "latency_s_p95": round(float(np.percentile(latencies, 95)), 2) if latencies else None,
        "tokens_per_question": mean([c["tokens"] for c in scored], 1),
        # Every request counts toward a free tier's daily quota, failed ones included
        "requests": sum(c.get("requests", 0) for c in cases),
        "requests_per_question": mean([c.get("requests", 0) for c in scored], 1),
        "wait_s_per_question": mean([c.get("wait_s", 0.0) for c in scored], 1),
        "filter_questions": len(filtered),
        "plan_filter_rate": mean([float(bool(c["plan_has_filters"])) for c in filtered]),
        "code_filter_rate": mean([float(bool(c["code_has_filters"])) for c in filtered]),
        "failures": dict(Counter(c["failure"] for c in scored if c.get("failure"))),
        "accuracy_std": None,
        "consistent_share": None,
        "cost_per_question_usd": None,
        "cost_per_correct_usd": None,
        "errors": len(cases) - len(scored),
    }

    repeats = sorted({c.get("repeat", 1) for c in scored})
    if len(repeats) > 1:
        per_run = [np.mean([c["correct"] for c in scored if c.get("repeat", 1) == r]) for r in repeats]
        summary["accuracy_std"] = round(float(np.std(per_run)), 3)
        outcomes: dict[int, list[bool]] = {}
        for c in scored:
            outcomes.setdefault(c["id"], []).append(c["correct"])
        # Same result (always right or always wrong) in every run
        summary["consistent_share"] = mean([float(all(v) or not any(v)) for v in outcomes.values()])

    if price and scored:
        costs = [case_cost(c, price) for c in scored]
        correct = sum(c["correct"] for c in scored)
        summary["cost_per_question_usd"] = round(float(np.mean(costs)), 5)
        summary["cost_per_correct_usd"] = round(sum(costs) / correct, 5) if correct else None
    return summary


def _fmt(value, suffix="") -> str:
    return "–" if value is None else f"{value}{suffix}"


def _accuracy(s: dict) -> str:
    return _fmt(s["accuracy"]) + (f" ± {s['accuracy_std']}" if s["accuracy_std"] is not None else "")


def failure_counts(failures: dict | None) -> str:
    if not failures:
        return "–"
    return " · ".join(f"{FAILURE_LABELS.get(k, k)} {n}" for k, n in sorted(failures.items(), key=lambda kv: -kv[1]))


def _miss_reason(c: dict) -> str:
    if c["error"]:
        return c["error"]
    if c["escalated"]:
        asked = " ".join((c.get("clarification") or "").split())[:200]
        return "asked for clarification" + (f": {asked}" if asked else "")
    what = " ".join(c["answer"].split())[:220] or "(empty answer)"
    failure = c.get("failure")
    if failure in ("ignored_filter", "wrong_period"):
        what += f" · **{FAILURE_LABELS[failure]}** ({', '.join(c.get('missing_filters', []))})"
    elif failure == "wrong_result":
        what += " · **wrong result**"
    return what


def markdown(summary: dict, cases: list[dict], model: str, stop_reason: str | None = None,
             dataset: str = "sales_sample.csv") -> str:
    """One model's results on one suite: the metric tables and every miss."""
    s = summary
    cost = f"${s['cost_per_question_usd']:.4f}" if s["cost_per_question_usd"] is not None else "–"
    lines = [
        f"**Model:** `{model}` · {s['scored']} of {s['questions']} questions scored on `{dataset}`", "",
    ]
    if stop_reason:
        lines += [f"> ⚠️ {stop_reason}", ""]
    lines += [
        "| Accuracy | Basic | Multi-step | Verified by critic | Verified but wrong | Clarification requests | "
        "Mean retries | Core-tool share | Latency p50 / p95 | Tokens / question | Cost / question | Errors |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        f"| {_accuracy(s)} | {_fmt(s['accuracy_basic'])} | {_fmt(s['accuracy_multi_step'])} | "
        f"{_fmt(s['verified_rate'])} | {s['verified_but_wrong']} | {_fmt(s['escalation_rate'])} | "
        f"{_fmt(s['mean_retries'])} | {_fmt(s['core_tool_share'])} | "
        f"{_fmt(s['latency_s_p50'], 's')} / {_fmt(s['latency_s_p95'], 's')} | {_fmt(s['tokens_per_question'])} | "
        f"{cost} | {s['errors']} |",
    ]
    if s.get("filter_questions") or s.get("failures"):
        lines += [
            "",
            "| Questions that filter rows | Filters in the plan | Filters applied in one code step | "
            "Critic approved a wrong answer | Failure types |",
            "|---|---|---|---|---|",
            f"| {s.get('filter_questions', 0)} | {_fmt(s.get('plan_filter_rate'))} | "
            f"{_fmt(s.get('code_filter_rate'))} | {s['verified_but_wrong']} | {failure_counts(s.get('failures'))} |",
        ]
    if s.get("requests"):
        lines += ["", f"LLM requests: {s['requests']} ({_fmt(s.get('requests_per_question'))} per scored question); "
                      f"{_fmt(s.get('wait_s_per_question'), ' s')} per question spent waiting on rate limits "
                      "(included in latency)."]
    if s["consistent_share"] is not None:
        lines += ["", f"Same outcome in every run for {s['consistent_share']:.0%} of questions."]
    misses = [c for c in cases if not c["correct"]]
    if misses:
        lines += ["", f"<details><summary>{len(misses)} incorrect or unanswered</summary>", ""]
        for c in misses:
            run = f" (run {c['repeat']})" if c.get("repeat", 1) > 1 else ""
            lines.append(f"- #{c['id']}{run} {c['question']} → {_miss_reason(c).replace('|', '/')}")
        lines += ["", "</details>"]
    return "\n".join(lines)


def comparison_table(reports: list[dict]) -> str:
    """One row per model: quality, speed and cost side by side."""
    lines = [
        "| Model | Suite | Accuracy | Basic | Multi-step | Verified but wrong | Latency p50 | Tokens / question | "
        "Cost / question | Cost / correct answer | Scored |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        s = r["summary"]
        cost_q = f"${s['cost_per_question_usd']:.4f}" if s["cost_per_question_usd"] is not None else "–"
        cost_c = f"${s['cost_per_correct_usd']:.4f}" if s["cost_per_correct_usd"] is not None else "–"
        lines.append(
            f"| `{r['model']}` | {r.get('suite', 'sales')} | {_accuracy(s)} | {_fmt(s['accuracy_basic'])} | {_fmt(s['accuracy_multi_step'])} | "
            f"{s['verified_but_wrong']} | {_fmt(s['latency_s_p50'], 's')} | {_fmt(s['tokens_per_question'])} | "
            f"{cost_q} | {cost_c} | {s['scored']}/{s['questions']} |"
        )
    return "\n".join(lines)


def report_markdown(reports: list[dict]) -> str:
    sections = [comparison_table(reports)] if len(reports) > 1 else []
    sections += [markdown(r["summary"], r["results"], r["model"], r["stop_reason"],
                          r.get("dataset", "sales_sample.csv")) for r in reports]
    return "\n\n".join(sections)


# ── run history and the regression gate ────────────────────────────

def current_commit() -> str:
    """Short commit of the code under test, "-dirty" when tracked files changed."""
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"][:7]
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""
    return head + ("-dirty" if dirty else "")


def history_entry(report: dict, label: str = "", commit: str = "", date: str | None = None) -> dict:
    """One line of evaluation/history.jsonl: a run of one model on one suite."""
    s = report["summary"]
    keys = ("questions", "scored", "errors", "correct", "scored_basic", "correct_basic", "scored_multi_step",
            "correct_multi_step", "accuracy", "verified_but_wrong", "escalations", "filter_questions",
            "plan_filter_rate", "code_filter_rate", "failures", "tokens_per_question", "requests_per_question",
            "latency_s_p50")
    return {
        "date": date or time.strftime("%Y-%m-%d", time.gmtime()),
        "commit": commit,
        "label": label,
        "suite": report.get("suite", "sales"),
        "model": report["model"],
        **{k: s.get(k) for k in keys},
    }


def append_history(entries: list[dict], path: Path = HISTORY_FILE) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, default=str) + "\n")


def load_history(path: Path = HISTORY_FILE) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _count(correct, scored) -> str:
    return "–" if correct is None or not scored else f"{correct} / {scored}"


def history_table(entries: list[dict]) -> str:
    """Trend table: one row per recorded run, oldest first."""
    lines = [
        "| Date | Commit | Label | Suite | Model | Correct | Basic | Multi-step | Filters applied | "
        "Verified but wrong | Clarifications | Failure types | Errors |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        lines.append(
            f"| {e.get('date', '')} | {e.get('commit') or '–'} | {e.get('label') or '–'} | {e.get('suite', '')} | "
            f"`{e.get('model', '')}` | {_count(e.get('correct'), e.get('scored'))} | "
            f"{_count(e.get('correct_basic'), e.get('scored_basic'))} | "
            f"{_count(e.get('correct_multi_step'), e.get('scored_multi_step'))} | "
            f"{_fmt(e.get('code_filter_rate'))} | {_fmt(e.get('verified_but_wrong'))} | "
            f"{_fmt(e.get('escalations'))} | {failure_counts(e.get('failures'))} | {_fmt(e.get('errors'))} |"
        )
    return "\n".join(lines)


def load_baseline(path: Path = BASELINE_FILE) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def gate(reports: list[dict], baseline: dict) -> tuple[list[str], list[str]]:
    """Compare each run with its baseline ("<suite>:<model>" in baseline.json).

    Returns (failures, notes). A run fails when its accuracy is more than
    max_drop below the baseline, or when too few questions were scored to
    tell (API errors such as rate limits): an incomplete run must not pass.
    """
    max_drop = float(baseline.get("max_drop", 0.10))
    min_share = float(baseline.get("min_scored_share", 0.9))
    failures, notes = [], []
    for r in reports:
        key = f"{r.get('suite', 'sales')}:{r['model']}"
        ref = baseline.get("baselines", {}).get(key)
        s = r["summary"]
        if ref is None:
            notes.append(f"{key}: no baseline yet, not gated")
            continue
        floor = round(float(ref["accuracy"]) - max_drop, 3)
        if s["questions"] and s["scored"] < min_share * s["questions"]:
            failures.append(f"{key}: only {s['scored']} of {s['questions']} questions were scored "
                            f"(API errors), too few to compare with the baseline")
        elif s["accuracy"] is None or s["accuracy"] < floor:
            failures.append(f"{key}: accuracy {s['accuracy']} is below {floor} "
                            f"(baseline {ref['accuracy']} minus {max_drop})")
        else:
            notes.append(f"{key}: accuracy {s['accuracy']} ≥ {floor} (baseline {ref['accuracy']} minus {max_drop})")
    return failures, notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=[*SUITES, "all"], default="sales",
                        help="question set: sales (development), support (held out) or all")
    parser.add_argument("--limit", type=int, help="only the first N questions (after --level)")
    parser.add_argument("--level", choices=[*LEVELS, "all"], default="all")
    parser.add_argument("--repeats", type=int, default=1, help="run every question N times")
    parser.add_argument("--models", help="comma-separated models of the active provider (LLM_PROVIDER) to compare")
    parser.add_argument("--model", help="a single model (same as --models with one entry)")
    parser.add_argument("--label", default="", help="note stored with the run in history.jsonl, e.g. before/after")
    parser.add_argument("--resume", action="store_true",
                        help="reuse the questions an interrupted run with the same suite, model, label and "
                             "repeats already finished (saved in evaluation/results/progress/)")
    parser.add_argument("--gate", action="store_true", help="fail if accuracy drops below evaluation/baseline.json")
    parser.add_argument("--no-history", action="store_true", help="don't append the run to history.jsonl")
    parser.add_argument("--history", action="store_true", help="print the trend table of recorded runs and exit")
    parser.add_argument("--dry-run", action="store_true", help="print questions and ground truth only")
    args = parser.parse_args()
    try:  # Windows consoles default to a code page without "–", "≥" or "⚠"
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    if args.history:
        print(history_table(load_history()))
        return

    suites = []
    for name in (list(SUITES) if args.suite == "all" else [args.suite]):
        spec = SUITES[name]
        df = pd.read_csv(spec["file"])
        cases = [c for c in spec["cases"](df) if args.level in ("all", c["level"])][: args.limit]
        suites.append((name, spec, df, cases))
    if args.dry_run:
        for name, spec, _, cases in suites:
            print(f"== {name} ({spec['file'].name}, {len(cases)} questions)")
            for c in cases:
                truth = c.get("number", c.get("names"))
                print(f"#{c['id']:>3} [{c['level']}] {c['question']}  ->  {truth}")
        return

    from tools.config import get_env

    import agents.llm as agent_llm

    if not get_env(agent_llm.api_key_name()):
        raise SystemExit(f"The benchmark needs {agent_llm.api_key_name()} (or use --dry-run).")
    models = [m.strip() for m in (args.models or args.model or agent_llm.get_model()).split(",") if m.strip()]
    commit = current_commit()
    repeats = max(args.repeats, 1)
    prices = load_prices()
    registry = DatasetRegistry(data_dir=None, max_bytes=50 * 1024 * 1024)
    datasets = {}
    for name, spec, df, _ in suites:
        dataset_id = registry.register_dataframe(df, spec["table"]).dataset_id
        datasets[name] = (dataset_id, registry.get_profile_text(dataset_id))

    reports = []
    for model in models:
        agent_llm._MODEL = model  # every node reads the model at call time
        for name, spec, _, cases in suites:
            dataset_id, profile_text = datasets[name]
            # Run-major order: if a daily limit stops the run, the earlier runs are complete.
            jobs = [{**c, "repeat": r} for r in range(1, repeats + 1) for c in cases]
            path = progress_path(name, model, args.label, repeats)
            done = start_progress(path, args.resume)
            todo = [j for j in jobs if (j["id"], j["repeat"]) not in done]
            print(f"\n== {agent_llm.get_provider()} {model} · {name} ({len(jobs)} runs"
                  + (f", {len(jobs) - len(todo)} reused from {path.name}" if done else "") + ")")
            if {r.get("commit") for r in done.values()} - {commit}:
                print(f"   note: reused answers come from commit(s) {sorted({r.get('commit') for r in done.values()})}"
                      f", this is {commit}")

            def save(result, path=path):
                result["commit"] = commit
                save_progress(path, result)

            new, stop_reason = run_all(
                todo, lambda case, ds=dataset_id, pt=profile_text: run_case(case, registry, ds, pt), on_result=save)
            finished = {**done, **{(r["id"], r["repeat"]): r for r in new}}
            results = [finished[(j["id"], j["repeat"])] for j in jobs if (j["id"], j["repeat"]) in finished]
            summary = summarize(results, planned=len(jobs), price=prices.get(model))
            reports.append({"model": model, "suite": name, "dataset": spec["file"].name, "summary": summary,
                            "stop_reason": stop_reason, "results": results})

    text = report_markdown(reports)
    # Only finished runs go into the history; a run stopped by a daily limit is resumed instead.
    entries = [history_entry(r, label=args.label, commit=commit) for r in reports if not r["stop_reason"]]
    text += ("\n\n<details><summary>History entries (evaluation/history.jsonl)</summary>\n\n```json\n"
             + "\n".join(json.dumps(e, default=str) for e in entries) + "\n```\n\n</details>")
    failures, notes = gate(reports, load_baseline()) if args.gate else ([], [])
    if args.gate:
        text += "\n\n**Regression gate:** " + ("failed" if failures else "passed") + "\n" + "".join(
            f"\n- {line}" for line in failures + notes)
    # Save first: the measured answers must survive any problem with printing.
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "benchmark.json").write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    if not args.no_history:
        append_history(entries)
    print("\n" + text)
    if any(r["stop_reason"] for r in reports):
        print("\nNot finished: rerun the same command with --resume after the quota resets. "
              "Finished questions are kept in evaluation/results/progress/.")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write("## Multi-agent benchmark\n\n" + text + "\n")
    if all(r["summary"]["scored"] == 0 for r in reports):
        # Nothing was measured: fail loudly instead of reporting accuracy 0.0 as a pass.
        raise SystemExit("No question could be scored (see errors above).")
    if failures:
        raise SystemExit("Regression gate failed:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    main()
