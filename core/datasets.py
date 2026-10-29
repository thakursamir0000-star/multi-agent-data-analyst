"""
DatasetRegistry — the controlled entry point for datasets.

Datasets enter the registry only through:
  - register_file():      a bare file name inside the approved data directory
  - register_upload():    bytes from an upload mechanism (e.g. Streamlit)
  - register_dataframe(): an in-memory DataFrame (tests / programmatic use)

Everything else refers to datasets by dataset ID. A dataset ID is never
interpreted as a path, and a file name is only opened after it has been
validated and confirmed to resolve to a regular file directly inside the
approved directory — so inputs like "../../secret.txt" or "C:\\Users\\..."
cannot load anything.

Parsing and profiling reuse tools.data_tool.DataTool.

Isolation boundary: a registry is the unit of data visibility — anything
registered in it can be read by every tool call on it, and query_sql can
read (and join) all of its datasets. So a registry must never be shared by
unrelated users:
  - The Streamlit app creates one registry per browser session (kept in
    st.session_state), so sessions cannot see each other's uploads.
  - The future MCP server (Phase 2) must likewise scope registries per
    client/session. A remote MCP server must not use one global,
    process-wide registry for unrelated clients.
"""

from __future__ import annotations

import copy
import hashlib
import io
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Sequence, Union

import pandas as pd

from core.errors import DatasetError, DatasetNotFoundError
from core.schemas import (
    ColumnInfo,
    ColumnProfile,
    DatasetInfo,
    DatasetProfile,
    DatasetSource,
    NumericSummary,
    column_kind,
    to_json_value,
)
from tools.data_tool import DataTool

# The file formats DataTool.load can read
ALLOWED_EXTENSIONS = frozenset({".csv", ".xlsx", ".xls"})
DEFAULT_MAX_BYTES = 50 * 1024 * 1024

_DATASET_ID_RE = re.compile(r"^ds_[0-9a-f]{16}$")
# Bare file names only: no separators, drive letters, ':' streams or leading dots
_FILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,127}$")
_MAX_NAME_CHARS = 128


@dataclass
class _Entry:
    info: DatasetInfo
    df: pd.DataFrame
    profile: dict[str, Any]
    profile_text: str


class DatasetRegistry:
    """Maps stable dataset IDs to approved, parsed datasets.

    Parameters
    ----------
    data_dir : str | Path, optional
        The approved directory for ``register_file``. Without it, only
        uploads and in-memory DataFrames can be registered.
    max_bytes : int
        Maximum size of a file or upload.
    """

    def __init__(
        self,
        data_dir: Union[str, Path, None] = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self._data_dir: Path | None = None
        if data_dir is not None:
            resolved = Path(data_dir).resolve()
            if not resolved.is_dir():
                raise DatasetError(f"Approved data directory does not exist: {data_dir}")
            self._data_dir = resolved
        self._max_bytes = max_bytes
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_file(self, file_name: str) -> DatasetInfo:
        """Register a file from the approved data directory by bare name."""
        if self._data_dir is None:
            raise DatasetError("No approved data directory is configured")
        name = self._validate_file_name(file_name)

        try:
            resolved = (self._data_dir / name).resolve(strict=True)
        except (OSError, RuntimeError):
            raise DatasetError(
                f"Dataset file '{name}' was not found in the approved data directory"
            ) from None
        # Also rejects symlinks that point outside the directory
        if resolved.parent != self._data_dir or not resolved.is_file():
            raise DatasetError(
                f"'{name}' is not a file in the approved data directory"
            )
        if resolved.stat().st_size > self._max_bytes:
            raise DatasetError(self._too_large_message(name))

        return self._register_bytes(resolved.read_bytes(), name, source="file")

    def register_upload(
        self, data: Union[bytes, bytearray, BinaryIO], file_name: str
    ) -> DatasetInfo:
        """Register uploaded content. ``file_name`` is only a label and is
        used to pick the parser; it is never opened as a path."""
        if hasattr(data, "read"):
            if hasattr(data, "seek"):
                data.seek(0)
            data = data.read()
        if not isinstance(data, (bytes, bytearray)):
            raise DatasetError("Upload content must be bytes")
        name = self._label(file_name)
        if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise DatasetError(self._unsupported_message(name))
        return self._register_bytes(bytes(data), name, source="upload")

    def register_dataframe(self, df: pd.DataFrame, name: str) -> DatasetInfo:
        """Register an in-memory DataFrame (a copy is stored)."""
        if not isinstance(df, pd.DataFrame):
            raise DatasetError("register_dataframe expects a pandas DataFrame")
        label = self._label(name)
        try:
            content = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        except TypeError:  # unhashable cells, e.g. lists
            content = df.to_csv().encode("utf-8")
        content += repr((list(df.columns), [str(t) for t in df.dtypes])).encode("utf-8")

        dataset_id = self._make_id(label, content)
        with self._lock:
            if dataset_id in self._entries:
                return self._entries[dataset_id].info
        return self._add(dataset_id, label, "dataframe", df.copy())

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def list_datasets(self) -> list[DatasetInfo]:
        """All registered datasets, in registration order."""
        with self._lock:
            return [entry.info for entry in self._entries.values()]

    def get(self, dataset_id: str) -> DatasetInfo:
        """Metadata for a registered dataset."""
        return self._entry(dataset_id).info

    def schema(self, dataset_id: str) -> list[ColumnInfo]:
        """Column names, dtypes and kinds for a registered dataset."""
        return self._entry(dataset_id).info.columns

    def get_dataframe(self, dataset_id: str) -> pd.DataFrame:
        """A copy of the dataset, so callers cannot modify the registered data."""
        return self._entry(dataset_id).df.copy()

    def get_profile(self, dataset_id: str) -> dict[str, Any]:
        """The DataTool profile dict for a registered dataset."""
        return copy.deepcopy(self._entry(dataset_id).profile)

    def get_profile_text(self, dataset_id: str) -> str:
        """The compact profile text used in LLM prompts."""
        return self._entry(dataset_id).profile_text

    def profile(self, dataset_id: str) -> DatasetProfile:
        """The DataTool profile as a typed, JSON-serializable model."""
        entry = self._entry(dataset_id)
        columns = []
        for info in entry.info.columns:
            raw = entry.profile[info.name] if info.name in entry.profile else {}
            stats = raw.get("stats")
            columns.append(
                ColumnProfile(
                    name=info.name,
                    dtype=info.dtype,
                    kind=info.kind,
                    null_count=info.null_count,
                    null_pct=float(raw.get("null_pct", 0.0)),
                    unique_count=info.unique_count,
                    sample_values=[to_json_value(v) for v in raw.get("sample_values", [])],
                    stats=None if stats is None else NumericSummary(
                        **{k: _number(stats.get(k)) for k in ("min", "max", "mean", "median", "std")}
                    ),
                )
            )
        return DatasetProfile(
            dataset_id=dataset_id,
            name=entry.info.name,
            rows=entry.info.rows,
            columns=columns,
            summary_text=entry.profile_text,
        )

    def remove(self, dataset_id: str) -> None:
        self._validate_id(dataset_id)
        with self._lock:
            if self._entries.pop(dataset_id, None) is None:
                raise DatasetNotFoundError(f"Unknown dataset ID: {dataset_id!r}")

    def available_files(self) -> list[str]:
        """Files in the approved data directory that can be registered."""
        if self._data_dir is None:
            return []
        return sorted(
            path.name
            for path in self._data_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in ALLOWED_EXTENSIONS
            and _FILE_NAME_RE.match(path.name)
        )

    def _tables(self, dataset_ids: Sequence[str] | None = None) -> dict[str, pd.DataFrame]:
        """table_name → DataFrame for core.sql (read-only use, not copied).

        All datasets, or only ``dataset_ids`` (each must be registered).
        """
        if dataset_ids is None:
            with self._lock:
                return {e.info.table_name: e.df for e in self._entries.values()}
        entries = [self._entry(dataset_id) for dataset_id in dataset_ids]
        return {e.info.table_name: e.df for e in entries}

    def _frame(self, dataset_id: str) -> pd.DataFrame:
        """The registered DataFrame for core.tools (read-only use, not copied)."""
        return self._entry(dataset_id).df

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _entry(self, dataset_id: str) -> _Entry:
        self._validate_id(dataset_id)
        with self._lock:
            entry = self._entries.get(dataset_id)
        if entry is None:
            raise DatasetNotFoundError(f"Unknown dataset ID: {dataset_id!r}")
        return entry

    @staticmethod
    def _validate_id(dataset_id: Any) -> None:
        if not isinstance(dataset_id, str) or not _DATASET_ID_RE.match(dataset_id):
            shown = repr(dataset_id)[:80]
            raise DatasetNotFoundError(
                f"Invalid dataset ID: {shown}. Expected an ID like 'ds_0123456789abcdef'"
            )

    @staticmethod
    def _validate_file_name(file_name: Any) -> str:
        if (
            not isinstance(file_name, str)
            or not _FILE_NAME_RE.match(file_name)
            or ".." in file_name
        ):
            shown = repr(file_name)[:80]
            raise DatasetError(
                f"Invalid dataset file name: {shown}. Use a bare file name "
                f"from the approved data directory (no paths)"
            )
        if Path(file_name).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise DatasetError(DatasetRegistry._unsupported_message(file_name))
        return file_name

    @staticmethod
    def _label(name: Any) -> str:
        """A display label: last path component, control characters removed."""
        if not isinstance(name, str):
            raise DatasetError("Dataset name must be a string")
        label = re.split(r"[\\/]", name)[-1]
        label = "".join(ch for ch in label if ch.isprintable()).strip()
        if not label:
            raise DatasetError("Dataset name must not be empty")
        return label[:_MAX_NAME_CHARS]

    @staticmethod
    def _make_id(name: str, content: bytes) -> str:
        """Stable ID: the same name and content always get the same ID."""
        digest = hashlib.sha256(name.encode("utf-8") + b"\x00" + content)
        return f"ds_{digest.hexdigest()[:16]}"

    def _register_bytes(self, data: bytes, name: str, source: DatasetSource) -> DatasetInfo:
        if not data:
            raise DatasetError(f"'{name}' is empty")
        if len(data) > self._max_bytes:
            raise DatasetError(self._too_large_message(name))

        dataset_id = self._make_id(name, data)
        with self._lock:
            if dataset_id in self._entries:
                return self._entries[dataset_id].info
        try:
            df = DataTool.load(io.BytesIO(data), file_name=name)
        except Exception as e:  # parser errors vary by format
            raise DatasetError(f"Could not read '{name}': {e}") from e
        return self._add(dataset_id, name, source, df)

    def _add(
        self, dataset_id: str, name: str, source: DatasetSource, df: pd.DataFrame
    ) -> DatasetInfo:
        profile = DataTool.profile(df)
        columns = [
            ColumnInfo(
                name=str(col),
                dtype=profile[col]["dtype"],
                kind=column_kind(df[col]),
                null_count=profile[col]["null_count"],
                unique_count=profile[col]["cardinality"],
            )
            for col in df.columns
        ]
        with self._lock:
            if dataset_id in self._entries:  # registered concurrently
                return self._entries[dataset_id].info
            info = DatasetInfo(
                dataset_id=dataset_id,
                name=name,
                table_name=self._unique_table_name(name),
                source=source,
                rows=len(df),
                columns=columns,
            )
            self._entries[dataset_id] = _Entry(
                info=info,
                df=df,
                profile=profile,
                profile_text=DataTool.profile_to_text(profile),
            )
        return info

    def _unique_table_name(self, name: str) -> str:
        """A SQL-safe table name derived from the dataset name (lock held)."""
        base = re.sub(r"[^a-z0-9]+", "_", Path(name).stem.lower()).strip("_")[:48]
        if not base:
            base = "dataset"
        if base[0].isdigit():
            base = f"t_{base}"
        taken = {e.info.table_name for e in self._entries.values()}
        candidate, n = base, 2
        while candidate in taken:
            candidate, n = f"{base}_{n}", n + 1
        return candidate

    def _too_large_message(self, name: str) -> str:
        mb = 1024 * 1024
        limit = (
            f"{self._max_bytes // mb} MB" if self._max_bytes >= mb
            else f"{self._max_bytes} bytes"
        )
        return f"'{name}' is larger than the {limit} limit"

    @staticmethod
    def _unsupported_message(name: str) -> str:
        return (
            f"Unsupported file type for '{name}'. "
            f"Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )


def _number(value: Any) -> float | None:
    """A JSON-native number (booleans as 0/1), or None."""
    value = to_json_value(value)
    return None if value is None or isinstance(value, str) else float(value)
