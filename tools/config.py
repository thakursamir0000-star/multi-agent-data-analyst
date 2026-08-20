"""
Config — centralised env / secrets reader.

Works both locally (python-dotenv) and on Streamlit Cloud (st.secrets).
All agents and the app import get_env(key) instead of os.getenv().
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


def get_env(key: str, default: str = "") -> str:
    """Read a config value: Streamlit secrets -> env var -> default."""
    # 1. Try Streamlit secrets (works on Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    # 2. OS / .env
    val = os.getenv(key)
    if val is not None:
        return val
    return default
