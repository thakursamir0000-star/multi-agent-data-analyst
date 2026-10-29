"""
MCP prompts: reusable instructions for a client's model.

They describe how to use this server's deterministic tools; they are not
agents and contain no LangGraph logic.
"""

from __future__ import annotations

from mcp.server import MCPServer

from mcp_server.context import ServerState


def register_prompts(server: MCPServer, state: ServerState) -> None:
    config = state.config
    session_note = (
        "Pass the same `session` token (from open_session) to every tool call.\n"
        if config.transport == "streamable-http"
        else ""
    )
    fallback = (
        "- Only if no tool fits (derived columns, date parsing, reshaping), use run_pandas; "
        "prefer the deterministic tools because their numbers are exact.\n"
        if config.allow_run_pandas
        else ""
    )

    @server.prompt(title="Analyze a dataset")
    def analyze_dataset(dataset_id: str, question: str = "") -> str:
        """Analyze a registered dataset with the deterministic tools."""
        goal = f"Answer this question: {question}" if question.strip() else (
            "Give an overview: what the data contains, its key figures and notable patterns."
        )
        return (
            f"Analyze the registered dataset `{dataset_id}`. {goal}\n\n"
            f"{session_note}"
            "Work with the server's tools and cite only numbers they return:\n"
            "1. Call get_dataset_schema (columns, types, SQL table name) and "
            "get_dataset_profile (missing values, sample values).\n"
            "2. Use the deterministic tools for the numbers:\n"
            "- describe for summary statistics;\n"
            "- group_aggregate for totals, averages or counts per group;\n"
            "- correlate for relationships between numeric columns;\n"
            "- detect_anomalies for outliers;\n"
            "- query_sql (one read-only SELECT on the table name) for filters, joins and rankings;\n"
            "- make_chart when a chart helps.\n"
            f"{fallback}"
            "3. Write a short answer: the direct answer first, then supporting figures. "
            "Say which tool produced each number, and state any limitation (missing values, "
            "truncated results)."
        )

    @server.prompt(title="Data quality report")
    def data_quality_report(dataset_id: str) -> str:
        """Assess the data quality of a registered dataset."""
        return (
            f"Write a data-quality report for the registered dataset `{dataset_id}`.\n\n"
            f"{session_note}"
            "Gather evidence with the server's tools and cite only numbers they return:\n"
            "1. Completeness: get_dataset_profile - null counts and percentages per column.\n"
            "2. Types and consistency: get_dataset_schema - is each dtype what the values suggest "
            "(e.g. dates or numbers stored as text)? Check sample values.\n"
            "3. Uniqueness: query_sql - duplicate rows (COUNT(*) vs COUNT(DISTINCT ...)) and "
            "candidate key columns; unique counts from the profile.\n"
            "4. Validity: describe - impossible ranges (negative quantities, prices of 0).\n"
            "5. Outliers: detect_anomalies on each important numeric column.\n\n"
            "Report: an overall rating (good / fair / poor), then one section per dimension "
            "with the evidence, and a prioritized list of fixes."
        )
