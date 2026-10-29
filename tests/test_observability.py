"""
Tests for tools/observability.py — trace logging and token accounting.
"""

from unittest.mock import patch

import pytest

from tools.observability import TOKEN_USAGE_KEY, log_node


@pytest.fixture
def disk_entries():
    """Capture entries written to disk instead of appending to logs/."""
    entries = []
    with patch(
        "tools.observability._write_to_disk",
        side_effect=lambda entry, *args, **kwargs: entries.append(entry),
    ):
        yield entries


class TestTokenAccounting:
    def test_usage_recorded_and_key_removed(self, disk_entries):
        @log_node("Test")
        def node(state):
            return {
                "draft_insight": "done",
                TOKEN_USAGE_KEY: {
                    "input_tokens": 100,
                    "output_tokens": 25,
                    "total_tokens": 125,
                },
            }

        result = node({})

        # The private key never reaches LangGraph state
        assert TOKEN_USAGE_KEY not in result
        assert result["draft_insight"] == "done"

        entry = result["trace_log"][0]
        assert entry["input_tokens"] == 100
        assert entry["output_tokens"] == 25
        assert entry["tokens_used"] == 125
        # The same counts are written to the JSONL log
        assert disk_entries == [entry]

    def test_usage_not_in_output_summary(self, disk_entries):
        @log_node("Test")
        def node(state):
            return {"x": 1, TOKEN_USAGE_KEY: {"total_tokens": 5}}

        entry = node({})["trace_log"][0]
        assert TOKEN_USAGE_KEY not in entry["output_summary"]

    def test_missing_usage_defaults_to_zero(self, disk_entries):
        @log_node("Test")
        def node(state):
            return {"x": 1}

        entry = node({})["trace_log"][0]
        assert entry["input_tokens"] == 0
        assert entry["output_tokens"] == 0
        assert entry["tokens_used"] == 0

    def test_error_still_logged(self, disk_entries):
        @log_node("Test")
        def node(state):
            raise ValueError("boom")

        with pytest.raises(ValueError):
            node({})

        assert disk_entries[0]["error"] == "boom"
        assert disk_entries[0]["tokens_used"] == 0
