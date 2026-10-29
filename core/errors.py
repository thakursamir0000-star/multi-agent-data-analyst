"""
Errors raised by the core layer.

All subclass CoreError (a ValueError), so callers — the LangGraph app now,
the MCP server later — can turn any of them into a clean error message.
"""

from __future__ import annotations


class CoreError(ValueError):
    """Base class for invalid requests to the core layer."""


class DatasetError(CoreError):
    """A dataset could not be registered (bad name, path, type, size or content)."""


class DatasetNotFoundError(DatasetError):
    """No registered dataset has the requested ID."""


class ColumnError(CoreError):
    """A requested column is missing or has the wrong type for the operation."""


class QueryError(CoreError):
    """A SQL query was rejected or failed."""
