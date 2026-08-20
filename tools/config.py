"""
Config — centralised env / secrets reader.

Works both locally (python-dotenv) and on Streamlit Cloud (st.secrets).
All agents and the app import get_env(key) instead of os.getenv().
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

_st_secrets: dict[str, str] | None = None


def _load_st_secrets() -> dict[str, str]:
    """Lazily load Streamlit secrets (returns empty dict outside Streamlit)."""
    global _st_secrets
    if _st_secrets is not None:
        return _st_secrets
    try:
        import streamlit as st  # noqa: F811
        _st_secrets = {k: str(v) for k, v in st.secrets.items()}
    except Exception:
        _st_secrets = {}
    return _st_secrets


def get_env(key: str, default: str = "") -> str:
    """Read a config value: Streamlit secrets → env var → default."""
    # 1. Streamlit secrets
    secrets = _load_st_secrets()
    if key in secrets:
        return secrets[key]
    # 2. OS / .env
    val = os.getenv(key)
    if val is not None:
        return val
    return default
