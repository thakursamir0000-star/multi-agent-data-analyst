"""
End-to-end benchmark for the multi-agent pipeline.

    python evaluation/benchmark.py              # run all questions (needs GROQ_API_KEY)
    python evaluation/benchmark.py --limit 5    # first 5 questions
    python evaluation/benchmark.py --dry-run    # print questions + ground truth, no LLM

Each question about sample_data/sales_sample.csv has a ground truth computed
directly in pandas. The full graph (planner → coder → analyst → critic) answers
it exactly as the app and the MCP analyze_question tool do, and the final
insight counts as correct when it contains the expected number (allowing for
normal rounding, %, K/M) or every expected name.

Reported: answer accuracy, how often the critic verified the answer (and how
often it verified a wrong one), clarification requests, retries, how much of
the work used deterministic core tools vs generated pandas, latency p50/p95
and tokens per question.
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


# ── questions with pandas ground truth ──────────────────────────────

def build_cases(df: pd.DataFrame) -> list[dict]:
    """Questions and their exact answers, computed from the data itself."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    month = d.groupby(d["date"].dt.to_period("M"))["revenue"].sum().idxmax()
    top3 = d.groupby("product")["revenue"].sum().nlargest(3).index.tolist()
    return [
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
    return {
        "id": case["id"],
        "question": case["question"],
        "answer": answer,
        "correct": bool(answer) and is_correct(case, answer),
        "verified": bool(final.get("verified_insight")),
        "escalated": escalated,
        "retries": final.get("retry_count", 0),
        "steps": len(outputs),
        "core_tool_steps": sum(1 for o in outputs if o.get("tool")),
        "tokens": sum(e.get("tokens_used", 0) for e in final.get("trace_log", [])),
        "latency_s": round(latency, 2),
        "error": error,
    }


def summarize(cases: list[dict]) -> dict:
    def mean(values):
        return round(float(np.mean(values)), 3) if values else None

    steps = sum(c["steps"] for c in cases)
    latencies = [c["latency_s"] for c in cases]
    return {
        "questions": len(cases),
        "accuracy": mean([float(c["correct"]) for c in cases]),
        "verified_rate": mean([float(c["verified"]) for c in cases]),
        "verified_but_wrong": sum(c["verified"] and not c["correct"] for c in cases),
        "escalation_rate": mean([float(c["escalated"]) for c in cases]),
        "mean_retries": mean([c["retries"] for c in cases]),
        "core_tool_share": round(sum(c["core_tool_steps"] for c in cases) / steps, 3) if steps else None,
        "latency_s_p50": round(float(np.percentile(latencies, 50)), 2) if latencies else None,
        "latency_s_p95": round(float(np.percentile(latencies, 95)), 2) if latencies else None,
        "tokens_per_question": mean([c["tokens"] for c in cases]),
        "errors": sum(1 for c in cases if c["error"]),
    }


def markdown(summary: dict, cases: list[dict], model: str) -> str:
    s = summary
    lines = [
        f"**Model:** `{model}` · {s['questions']} questions on `sales_sample.csv`", "",
        "| Accuracy | Verified by critic | Verified but wrong | Clarification requests | Mean retries | "
        "Core-tool share | Latency p50 / p95 | Tokens / question | Errors |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| {s['accuracy']} | {s['verified_rate']} | {s['verified_but_wrong']} | {s['escalation_rate']} | "
        f"{s['mean_retries']} | {s['core_tool_share']} | {s['latency_s_p50']}s / {s['latency_s_p95']}s | "
        f"{s['tokens_per_question']} | {s['errors']} |",
    ]
    misses = [c for c in cases if not c["correct"]]
    if misses:
        lines += ["", f"<details><summary>{len(misses)} incorrect or unanswered</summary>", ""]
        for c in misses:
            what = c["error"] or ("asked for clarification" if c["escalated"]
                                  else " ".join(c["answer"].split())[:220] or "(empty answer)")
            lines.append(f"- #{c['id']} {c['question']} → {what.replace('|', '/')}")
        lines += ["", "</details>"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--dry-run", action="store_true", help="print questions and ground truth only")
    args = parser.parse_args()

    df = pd.read_csv(DATA_FILE)
    cases = build_cases(df)[: args.limit]
    if args.dry_run:
        for c in cases:
            truth = c.get("number", c.get("names"))
            print(f"#{c['id']:>2} {c['question']}  ->  {truth}")
        return

    from tools.config import get_env

    if not get_env("GROQ_API_KEY"):
        raise SystemExit("The benchmark needs GROQ_API_KEY (or use --dry-run).")
    from agents.llm import get_model

    registry = DatasetRegistry(data_dir=None, max_bytes=50 * 1024 * 1024)
    dataset_id = registry.register_dataframe(df, "sales_sample").dataset_id
    profile_text = registry.get_profile_text(dataset_id)

    results = []
    for case in cases:
        result = run_case(case, registry, dataset_id, profile_text)
        print(f"#{result['id']:>2} {'OK  ' if result['correct'] else 'MISS'} {result['latency_s']:>5.1f}s  {case['question']}")
        results.append(result)

    summary = summarize(results)
    table = markdown(summary, results, get_model())
    print("\n" + table)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "benchmark.json").write_text(
        json.dumps({"summary": summary, "cases": results}, indent=2, default=str), encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write("## Multi-agent benchmark\n\n" + table + "\n")


if __name__ == "__main__":
    main()
