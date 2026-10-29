"""
End-to-end benchmark for the multi-agent pipeline.

    python evaluation/benchmark.py              # run all questions (needs GROQ_API_KEY)
    python evaluation/benchmark.py --limit 5    # first 5 questions
    python evaluation/benchmark.py --level multi_step --repeats 3
    python evaluation/benchmark.py --models openai/gpt-oss-120b,openai/gpt-oss-20b
    python evaluation/benchmark.py --dry-run    # print questions + ground truth, no LLM

Questions are "basic" (one aggregation or lookup) or "multi_step" (filter,
group and compare). --repeats runs each question several times and reports
the spread, because LLM answers vary between runs. --models compares models
side by side, with cost per question from evaluation/pricing.json.

A per-minute Groq rate limit is waited out once per question; a daily limit
stops the run (the remaining questions can't succeed until it resets), and a
run in which no question could be scored exits with an error.

Each question about sample_data/sales_sample.csv has a ground truth computed
directly in pandas. The full graph (planner → coder → analyst → critic) answers
it exactly as the app and the MCP analyze_question tool do, and the final
insight counts as correct when it contains the expected number (allowing for
normal rounding, %, K/M) or every expected name.

Reported: answer accuracy (overall and per level), how often the critic
verified the answer (and how often it verified a wrong one), clarification
requests, retries, how much of the work used deterministic core tools vs
generated pandas, latency p50/p95, tokens and cost per question.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.datasets import DatasetRegistry  # noqa: E402

DATA_FILE = ROOT / "sample_data" / "sales_sample.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
PRICING_FILE = Path(__file__).resolve().parent / "pricing.json"
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
         "number": d.loc[d["region"] == "South", "quantity"].sum()},
        {"id": 7, "question": "What is the median unit price?", "number": d["unit_price"].median()},
        {"id": 8, "question": "Which product has the highest total profit?",
         "names": [d.groupby("product")["profit"].sum().idxmax()]},
        {"id": 9, "question": "How many distinct products are there?", "number": d["product"].nunique()},
        {"id": 10, "question": "What is the Pearson correlation between discount and profit?",
         "number": d["discount"].corr(d["profit"]), "absolute": True},
        {"id": 11, "question": "What was the total revenue in 2024?",
         "number": d.loc[d["date"].dt.year == 2024, "revenue"].sum()},
        {"id": 12, "question": "Which month had the highest total revenue?",
         "names": [[str(month), month.strftime("%B %Y"), month.strftime("%b %Y")]]},
        {"id": 13, "question": "What is the overall profit margin (total profit divided by total revenue)?",
         "number": d["profit"].sum() / d["revenue"].sum(), "ratio": True},
        {"id": 14, "question": "How many transactions are missing a region?", "number": d["region"].isna().sum()},
        {"id": 15, "question": "What are the top 3 products by total revenue?", "names": top3},
    ]
    multi_step = [
        {"id": 16, "question": "Which region had the highest total profit in 2025?",
         "names": [y2025.groupby("region")["profit"].sum().idxmax()]},
        {"id": 17, "question": "What was the average discount on transactions with revenue above 1,000?",
         "number": d.loc[d["revenue"] > 1000, "discount"].mean(), "ratio": True},
        {"id": 18, "question": "Which product category has the highest profit margin (total profit divided by total revenue)?",
         "names": [margin.idxmax()]},
        {"id": 19, "question": "By what percentage did total revenue change from the first half of 2024 "
                               "(January to June) to the first half of 2025?",
         "number": (h1_2025 - h1_2024) / h1_2024, "ratio": True},
        {"id": 20, "question": "How many transactions in the North region had a discount of at least 15%?",
         "number": ((d["region"] == "North") & (d["discount"] >= 0.15)).sum()},
        {"id": 21, "question": "Which product sold the most units in the West region?",
         "names": [d[d["region"] == "West"].groupby("product")["quantity"].sum().idxmax()]},
        {"id": 22, "question": "What is the average profit per transaction for the Gadgets category in 2024?",
         "number": gadgets_2024["profit"].mean()},
    ]
    return [{**c, "level": "basic"} for c in basic] + [{**c, "level": "multi_step"} for c in multi_step]


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


def matches_names(text: str, names: list) -> bool:
    """Every expected name appears; a nested list means "any of these spellings"."""
    lower = text.lower()
    return all(any(str(alt).lower() in lower for alt in (n if isinstance(n, list) else [n])) for n in names)


def is_correct(case: dict, answer: str) -> bool:
    if "number" in case:
        return matches_number(answer, case["number"], case.get("ratio", False), case.get("absolute", False))
    return matches_names(answer, case["names"])


# ── running the pipeline ────────────────────────────────────────────

def run_case(case: dict, registry: DatasetRegistry, dataset_id: str, profile_text: str) -> dict:
    from agents.graph import build_graph, create_initial_state

    start = time.perf_counter()
    try:
        graph = build_graph(registry=registry, dataset_id=dataset_id)
        final = graph.invoke(create_initial_state(case["question"], profile_text),
                             {"configurable": {"thread_id": uuid.uuid4().hex}})
        error = None
    except Exception as e:  # an API error shouldn't stop the whole benchmark
        final, error = {}, f"{type(e).__name__}: {e}"
    latency = time.perf_counter() - start

    escalated = bool(final.get("__interrupt__"))
    answer = "" if escalated else (final.get("verified_insight") or final.get("draft_insight") or "")
    outputs = final.get("code_outputs", [])
    trace = final.get("trace_log", [])
    return {
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
        "error": error,
    }


_RETRY_AFTER = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s")


def rate_limit_kind(error: str | None) -> str | None:
    """"daily" / "minute" for a Groq rate-limit error, else None."""
    if not error or not ("429" in error or "rate limit" in error.lower()):
        return None
    return "daily" if "per day" in error else "minute"


def retry_after_seconds(error: str, default: float = 30.0) -> float:
    match = _RETRY_AFTER.search(error)
    return int(match.group(1) or 0) * 60 + float(match.group(2)) if match else default


def run_all(cases: list[dict], runner, sleep=time.sleep, max_wait: float = 120.0) -> tuple[list[dict], str | None]:
    """Run every case; wait out per-minute limits once, stop at a daily limit.

    Returns (results, stop_reason). After a daily limit the remaining
    questions can't succeed until the quota resets, so they aren't attempted.
    """
    results = []
    for case in cases:
        result = runner(case)
        if rate_limit_kind(result["error"]) == "minute":
            wait = min(retry_after_seconds(result["error"]) + 2, max_wait)
            print(f"   per-minute rate limit; waiting {wait:.0f}s and retrying #{case['id']}")
            sleep(wait)
            result = runner(case)
        result["repeat"] = case.get("repeat", 1)
        run_label = f" (run {result['repeat']})" if result["repeat"] > 1 else ""
        print(f"#{result['id']:>2}{run_label} {'OK  ' if result['correct'] else 'MISS'} "
              f"{result['latency_s']:>5.1f}s  {case['question']}")
        results.append(result)
        if rate_limit_kind(result["error"]) == "daily":
            skipped = len(cases) - len(results)
            return results, (f"Groq's daily token limit was reached after {len(results)} question(s); "
                             f"{skipped} not run. Rerun after the quota resets or with --models <another model>.")
    return results, None


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
    summary = {
        "questions": planned if planned is not None else len(cases),
        "scored": len(scored),
        "accuracy": mean([float(c["correct"]) for c in scored]),
        **{f"accuracy_{level}": mean([float(c["correct"]) for c in scored if c.get("level", "basic") == level])
           for level in LEVELS},
        "verified_rate": mean([float(c["verified"]) for c in scored]),
        "verified_but_wrong": sum(c["verified"] and not c["correct"] for c in scored),
        "escalation_rate": mean([float(c["escalated"]) for c in scored]),
        "mean_retries": mean([c["retries"] for c in scored]),
        "core_tool_share": round(sum(c["core_tool_steps"] for c in scored) / steps, 3) if steps else None,
        "latency_s_p50": round(float(np.percentile(latencies, 50)), 2) if latencies else None,
        "latency_s_p95": round(float(np.percentile(latencies, 95)), 2) if latencies else None,
        "tokens_per_question": mean([c["tokens"] for c in scored], 1),
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


def markdown(summary: dict, cases: list[dict], model: str, stop_reason: str | None = None) -> str:
    """One model's results: the metric table and every miss."""
    s = summary
    cost = f"${s['cost_per_question_usd']:.4f}" if s["cost_per_question_usd"] is not None else "–"
    lines = [
        f"**Model:** `{model}` · {s['scored']} of {s['questions']} questions scored on `sales_sample.csv`", "",
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
    if s["consistent_share"] is not None:
        lines += ["", f"Same outcome in every run for {s['consistent_share']:.0%} of questions."]
    misses = [c for c in cases if not c["correct"]]
    if misses:
        lines += ["", f"<details><summary>{len(misses)} incorrect or unanswered</summary>", ""]
        for c in misses:
            what = c["error"] or ("asked for clarification" if c["escalated"]
                                  else " ".join(c["answer"].split())[:220] or "(empty answer)")
            run = f" (run {c['repeat']})" if c.get("repeat", 1) > 1 else ""
            lines.append(f"- #{c['id']}{run} {c['question']} → {what.replace('|', '/')}")
        lines += ["", "</details>"]
    return "\n".join(lines)


def comparison_table(reports: list[dict]) -> str:
    """One row per model: quality, speed and cost side by side."""
    lines = [
        "| Model | Accuracy | Basic | Multi-step | Verified but wrong | Latency p50 | Tokens / question | "
        "Cost / question | Cost / correct answer | Scored |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        s = r["summary"]
        cost_q = f"${s['cost_per_question_usd']:.4f}" if s["cost_per_question_usd"] is not None else "–"
        cost_c = f"${s['cost_per_correct_usd']:.4f}" if s["cost_per_correct_usd"] is not None else "–"
        lines.append(
            f"| `{r['model']}` | {_accuracy(s)} | {_fmt(s['accuracy_basic'])} | {_fmt(s['accuracy_multi_step'])} | "
            f"{s['verified_but_wrong']} | {_fmt(s['latency_s_p50'], 's')} | {_fmt(s['tokens_per_question'])} | "
            f"{cost_q} | {cost_c} | {s['scored']}/{s['questions']} |"
        )
    return "\n".join(lines)


def report_markdown(reports: list[dict]) -> str:
    sections = [comparison_table(reports)] if len(reports) > 1 else []
    sections += [markdown(r["summary"], r["results"], r["model"], r["stop_reason"]) for r in reports]
    return "\n\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only the first N questions (after --level)")
    parser.add_argument("--level", choices=[*LEVELS, "all"], default="all")
    parser.add_argument("--repeats", type=int, default=1, help="run every question N times")
    parser.add_argument("--models", help="comma-separated Groq models to compare")
    parser.add_argument("--model", help="a single Groq model (same as --models with one entry)")
    parser.add_argument("--dry-run", action="store_true", help="print questions and ground truth only")
    args = parser.parse_args()

    df = pd.read_csv(DATA_FILE)
    cases = [c for c in build_cases(df) if args.level in ("all", c["level"])][: args.limit]
    if args.dry_run:
        for c in cases:
            truth = c.get("number", c.get("names"))
            print(f"#{c['id']:>2} [{c['level']}] {c['question']}  ->  {truth}")
        return

    from tools.config import get_env

    if not get_env("GROQ_API_KEY"):
        raise SystemExit("The benchmark needs GROQ_API_KEY (or use --dry-run).")
    import agents.llm as agent_llm

    models = [m.strip() for m in (args.models or args.model or agent_llm.get_model()).split(",") if m.strip()]
    prices = load_prices()
    registry = DatasetRegistry(data_dir=None, max_bytes=50 * 1024 * 1024)
    dataset_id = registry.register_dataframe(df, "sales_sample").dataset_id
    profile_text = registry.get_profile_text(dataset_id)
    # Run-major order: if a daily limit stops the run, the earlier runs are complete.
    jobs = [{**c, "repeat": r} for r in range(1, max(args.repeats, 1) + 1) for c in cases]

    reports = []
    for model in models:
        agent_llm._MODEL = model  # every node reads the model at call time
        print(f"\n== {model} ({len(jobs)} runs)")
        results, stop_reason = run_all(jobs, lambda case: run_case(case, registry, dataset_id, profile_text))
        summary = summarize(results, planned=len(jobs), price=prices.get(model))
        reports.append({"model": model, "summary": summary, "stop_reason": stop_reason, "results": results})

    text = report_markdown(reports)
    print("\n" + text)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "benchmark.json").write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write("## Multi-agent benchmark\n\n" + text + "\n")
    if all(r["summary"]["scored"] == 0 for r in reports):
        # Nothing was measured: fail loudly instead of reporting accuracy 0.0 as a pass.
        raise SystemExit("No question could be scored (see errors above).")


if __name__ == "__main__":
    main()
