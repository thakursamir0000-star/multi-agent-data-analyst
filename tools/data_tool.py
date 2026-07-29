"""
DataTool — CSV/Excel loader and auto-profiler.

Loads tabular data into a pandas DataFrame and generates a compact
column-level profile suitable for injection into LLM prompts.
"""

from __future__ import annotations

import io
from typing import Any, BinaryIO, Union

import pandas as pd


class DataTool:
    """Loads and profiles tabular datasets for the agent graph."""

    # Supported extensions → reader mapping
    _READERS = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
    }

    MAX_SAMPLE_VALUES = 5  # max unique values shown per column in profile

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        source: Union[str, BinaryIO],
        *,
        file_name: str | None = None,
    ) -> pd.DataFrame:
        """Load a CSV or Excel file into a DataFrame.

        Parameters
        ----------
        source : str | BinaryIO
            A file path (str) or a file-like object (e.g. Streamlit
            ``UploadedFile``).
        file_name : str, optional
            Original filename — used when *source* is a buffer to
            determine the file type.  Ignored when *source* is a path.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        ValueError
            If the file extension is unsupported.
        """
        ext = cls._resolve_extension(source, file_name)
        reader = cls._READERS.get(ext)
        if reader is None:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(cls._READERS)}"
            )

        # If source is a file-like buffer, ensure we read from the start
        if hasattr(source, "seek"):
            source.seek(0)

        return reader(source)

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    @classmethod
    def profile(cls, df: pd.DataFrame) -> dict[str, Any]:
        """Generate a compact per-column profile of the DataFrame.

        Returns a dict keyed by column name, each containing:
        - dtype: pandas dtype as string
        - null_count / null_pct: missing-value stats
        - cardinality: number of unique values
        - sample_values: up to MAX_SAMPLE_VALUES examples
        - stats: numeric summary (min/max/mean/median/std) if applicable

        Also includes top-level keys:
        - _rows: total row count
        - _columns: total column count
        """
        profile: dict[str, Any] = {
            "_rows": len(df),
            "_columns": len(df.columns),
        }

        for col in df.columns:
            series = df[col]
            col_info: dict[str, Any] = {
                "dtype": str(series.dtype),
                "null_count": int(series.isna().sum()),
                "null_pct": round(series.isna().mean() * 100, 2),
                "cardinality": int(series.nunique()),
            }

            # Sample values (converted to native Python types for JSON safety)
            unique_vals = series.dropna().unique()
            samples = unique_vals[: cls.MAX_SAMPLE_VALUES].tolist()
            col_info["sample_values"] = [
                cls._safe_convert(v) for v in samples
            ]

            # Numeric stats
            if pd.api.types.is_numeric_dtype(series):
                desc = series.describe()
                col_info["stats"] = {
                    "min": cls._safe_convert(desc.get("min")),
                    "max": cls._safe_convert(desc.get("max")),
                    "mean": cls._safe_convert(desc.get("mean")),
                    "median": cls._safe_convert(series.median()),
                    "std": cls._safe_convert(desc.get("std")),
                }

            profile[col] = col_info

        return profile

    @classmethod
    def profile_to_text(cls, profile: dict[str, Any]) -> str:
        """Convert a profile dict to a compact text block for LLM prompts.

        Keeps the representation short to conserve tokens.
        """
        lines = [
            f"Dataset: {profile['_rows']} rows × {profile['_columns']} columns",
            "",
        ]
        for key, info in profile.items():
            if key.startswith("_"):
                continue
            dtype = info["dtype"]
            nulls = info["null_count"]
            card = info["cardinality"]
            samples = info["sample_values"]
            line = f"  • {key} ({dtype}) — {nulls} nulls, {card} unique"
            if samples:
                sample_str = ", ".join(str(s) for s in samples[:3])
                line += f"  e.g. [{sample_str}]"
            if "stats" in info:
                s = info["stats"]
                line += (
                    f"  range=[{s['min']}..{s['max']}] "
                    f"mean={s['mean']} std={s['std']}"
                )
            lines.append(line)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_extension(
        source: Union[str, BinaryIO], file_name: str | None
    ) -> str:
        """Return the lowercase file extension (e.g. '.csv')."""
        import os

        if isinstance(source, str):
            _, ext = os.path.splitext(source)
            return ext.lower()
        if file_name:
            _, ext = os.path.splitext(file_name)
            return ext.lower()
        # Streamlit UploadedFile has a .name attribute
        if hasattr(source, "name"):
            _, ext = os.path.splitext(source.name)
            return ext.lower()
        raise ValueError(
            "Cannot determine file type. Pass file_name= for buffers."
        )

    @staticmethod
    def _safe_convert(value: Any) -> Any:
        """Convert numpy/pandas scalars to native Python types."""
        import numpy as np

        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return round(float(value), 4)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (pd.Timestamp,)):
            return value.isoformat()
        return value
