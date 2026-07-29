"""
DataFrame Store — simple module-level store for the active DataFrame.

LangGraph's StateGraph only keeps keys defined in the TypedDict schema
and cannot serialize DataFrames.  This module provides a simple global
reference that the Coder node reads at execution time.

This is safe for Streamlit (single-threaded per session, synchronous
graph execution per query).
"""

from __future__ import annotations

import pandas as pd

_current_df: pd.DataFrame | None = None


def set_df(df: pd.DataFrame) -> None:
    """Set the active DataFrame."""
    global _current_df
    _current_df = df


def get_df() -> pd.DataFrame | None:
    """Get the active DataFrame."""
    return _current_df


def clear_df() -> None:
    """Clear the active DataFrame."""
    global _current_df
    _current_df = None
