"""
Multi-Agent Data Analyst — Streamlit Frontend

Features:
  - CSV/Excel file upload with auto-profiling
  - Chat interface for natural-language queries
  - Collapsible agent trace panel (sidebar) with per-node details
  - Human-in-the-loop: pause/resume when Critic escalates
  - Inline plot rendering from agent code outputs
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `agents` and `tools` are importable
# regardless of how Streamlit launches this file.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import base64
import uuid
from io import BytesIO
from typing import Any

import streamlit as st
from langgraph.types import Command

from tools.config import get_env

# ── Page Configuration ──────────────────────────────────────────────

st.set_page_config(
    page_title="Multi-Agent Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Styling ──────────────────────────────────────────────────

st.markdown("""
<style>
    /* Global dark theme refinements */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Header area */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        font-family: 'Inter', sans-serif;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        color: #a0aec0;
        font-size: 1rem;
        margin-bottom: 1.5rem;
        font-family: 'Inter', sans-serif;
    }

    /* Chat messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
    }

    /* File uploader */
    .stFileUploader {
        border: 2px dashed rgba(102, 126, 234, 0.4) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    /* Sidebar trace panel */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%) !important;
    }

    .trace-node {
        background: rgba(255, 255, 255, 0.05);
        border-left: 3px solid #667eea;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }

    .trace-node-pass {
        border-left-color: #48bb78;
    }

    .trace-node-fail {
        border-left-color: #f56565;
    }

    .trace-node-active {
        border-left-color: #ecc94b;
        animation: pulse 1.5s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .badge-pass { background: #48bb78; color: #1a202c; }
    .badge-fail { background: #f56565; color: #1a202c; }
    .badge-running { background: #ecc94b; color: #1a202c; }

    /* Profile cards */
    .profile-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Metrics row */
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)


# ── Session State Initialization ────────────────────────────────────

def _init_session_state() -> None:
    """Initialize all session_state keys on first run."""
    defaults = {
        "df": None,
        "profile": None,
        "profile_text": "",
        "chat_history": [],
        "trace_logs": [],
        "graph": None,
        "thread_id": str(uuid.uuid4()),
        "awaiting_human_input": False,
        "human_question": "",
        "graph_state": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()


# ── Helper Functions ────────────────────────────────────────────────

def _load_graph():
    """Build the compiled graph with the current DataFrame.

    Always rebuilds when graph is None (set to None on each new query)
    so the Coder node captures the current DataFrame via closure.
    """
    if st.session_state.graph is None:
        import importlib
        import agents.coder as _coder_mod
        import agents.graph as _graph_mod
        importlib.reload(_coder_mod)
        importlib.reload(_graph_mod)
        st.session_state.graph = _graph_mod.build_graph(df=st.session_state.df)
    return st.session_state.graph


def _render_plot(plot_base64: str) -> None:
    """Render a base64-encoded plot image."""
    image_bytes = base64.b64decode(plot_base64)
    st.image(BytesIO(image_bytes), use_container_width=True)


def _render_trace_sidebar() -> None:
    """Render the agent trace panel in the sidebar."""
    with st.sidebar:
        st.markdown("## 🔍 Agent Trace")
        st.markdown("---")

        traces = st.session_state.trace_logs
        if not traces:
            st.info("Run a query to see the agent trace here.")
            return

        for i, entry in enumerate(traces):
            node = entry.get("node_name", "Unknown")
            latency = entry.get("latency_ms", 0)
            error = entry.get("error")
            timestamp = entry.get("timestamp", "")

            # Node icon mapping
            icons = {
                "Planner": "🧠",
                "Coder": "💻",
                "Analyst": "📊",
                "Critic": "🔍",
                "HumanInput": "🤚",
            }
            icon = icons.get(node, "⚙️")

            # Status badge
            if error:
                badge = '<span class="badge badge-fail">ERROR</span>'
                node_class = "trace-node trace-node-fail"
            else:
                badge = '<span class="badge badge-pass">OK</span>'
                node_class = "trace-node trace-node-pass"

            with st.expander(f"{icon} {node} — {latency:.0f}ms", expanded=False):
                st.markdown(f"{badge}", unsafe_allow_html=True)

                if entry.get("input_summary"):
                    st.text(f"Input: {entry['input_summary']}")
                if entry.get("output_summary"):
                    st.text(f"Output: {entry['output_summary'][:200]}")
                if entry.get("tokens_used"):
                    st.metric("Tokens", entry["tokens_used"])
                if error:
                    st.error(error)


def _run_analysis(query: str) -> None:
    """Run the full multi-agent analysis pipeline."""
    graph = _load_graph()
    thread_id = st.session_state.thread_id

    from agents.graph import create_initial_state

    initial_state = create_initial_state(
        user_query=query,
        dataframe_profile=st.session_state.profile_text,
    )

    config = {"configurable": {"thread_id": thread_id}}

    with st.status("🤖 Agents are working...", expanded=True) as status:
        try:
            # Stream graph updates for real-time trace
            final_state = None
            for event in graph.stream(initial_state, config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    if node_name == "__interrupt__":
                        # Human-in-the-loop pause
                        st.session_state.awaiting_human_input = True
                        if isinstance(node_output, list) and node_output:
                            st.session_state.human_question = str(
                                node_output[0].value
                                if hasattr(node_output[0], "value")
                                else node_output[0]
                            )
                        status.update(
                            label="⏸️ Waiting for your input...",
                            state="error",
                        )
                        return

                    st.write(f"✅ **{node_name}** completed")

                    # Update trace logs
                    if isinstance(node_output, dict):
                        trace_entries = node_output.get("trace_log", [])
                        st.session_state.trace_logs.extend(trace_entries)

                        # Show progress
                        if node_output.get("sub_tasks"):
                            n = len(node_output["sub_tasks"])
                            st.write(f"   📋 Planned {n} sub-tasks")
                        if node_output.get("code_outputs"):
                            for co in node_output["code_outputs"]:
                                if co.get("error"):
                                    st.write(f"   ⚠️ Code error in: {co['task']}")
                                else:
                                    st.write(f"   ✅ Executed: {co['task']}")

                    final_state = node_output

            # Get final state from checkpointer
            full_state = graph.get_state(config)
            if full_state and full_state.values:
                final_state = full_state.values
                st.session_state.graph_state = final_state

            status.update(label="✅ Analysis complete!", state="complete")

        except Exception as e:
            status.update(label=f"❌ Error: {str(e)}", state="error")
            st.error(f"Analysis failed: {str(e)}")
            return

    # Display results
    if final_state:
        insight = final_state.get("verified_insight") or final_state.get(
            "draft_insight", ""
        )
        if insight:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": insight}
            )

            # Display any plots
            code_outputs = final_state.get("code_outputs", [])
            for co in code_outputs:
                if co.get("plot_base64"):
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": "__PLOT__",
                            "plot_base64": co["plot_base64"],
                        }
                    )


def _resume_with_human_input(answer: str) -> None:
    """Resume the graph after human-in-the-loop pause."""
    graph = _load_graph()
    thread_id = st.session_state.thread_id
    config = {"configurable": {"thread_id": thread_id}}

    st.session_state.awaiting_human_input = False
    st.session_state.human_question = ""

    with st.status("🤖 Resuming analysis with your input...", expanded=True) as status:
        try:
            final_state = None
            for event in graph.stream(
                Command(resume=answer), config, stream_mode="updates"
            ):
                for node_name, node_output in event.items():
                    if node_name == "__interrupt__":
                        st.session_state.awaiting_human_input = True
                        if isinstance(node_output, list) and node_output:
                            st.session_state.human_question = str(
                                node_output[0].value
                                if hasattr(node_output[0], "value")
                                else node_output[0]
                            )
                        status.update(
                            label="⏸️ Waiting for your input again...",
                            state="error",
                        )
                        return

                    st.write(f"✅ **{node_name}** completed")
                    if isinstance(node_output, dict):
                        trace_entries = node_output.get("trace_log", [])
                        st.session_state.trace_logs.extend(trace_entries)
                    final_state = node_output

            full_state = graph.get_state(config)
            if full_state and full_state.values:
                final_state = full_state.values

            status.update(label="✅ Analysis complete!", state="complete")

        except Exception as e:
            status.update(label=f"❌ Error: {str(e)}", state="error")
            st.error(f"Resume failed: {str(e)}")
            return

    if final_state:
        insight = final_state.get("verified_insight") or final_state.get(
            "draft_insight", ""
        )
        if insight:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": insight}
            )
            code_outputs = final_state.get("code_outputs", [])
            for co in code_outputs:
                if co.get("plot_base64"):
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": "__PLOT__",
                            "plot_base64": co["plot_base64"],
                        }
                    )


# ── Main Layout ─────────────────────────────────────────────────────

def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<p class="main-header">🤖 Multi-Agent Data Analyst</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload a dataset, ask questions in plain English, '
        "and let specialized AI agents analyze your data.</p>",
        unsafe_allow_html=True,
    )

    # ── File Upload Section ─────────────────────────────────────────
    with st.container():
        uploaded_file = st.file_uploader(
            "📁 Upload your dataset",
            type=["csv", "xlsx", "xls"],
            help="Supports CSV and Excel files up to 50 MB",
            key="file_uploader",
        )

        if uploaded_file and st.session_state.df is None:
                max_mb = int(get_env("MAX_UPLOAD_MB", "50"))
            if uploaded_file.size > max_mb * 1024 * 1024:
                st.error(f"File too large. Maximum size: {max_mb} MB")
            else:
                with st.spinner("Loading and profiling dataset..."):
                    from tools.data_tool import DataTool

                    try:
                        df = DataTool.load(uploaded_file, file_name=uploaded_file.name)
                        profile = DataTool.profile(df)
                        profile_text = DataTool.profile_to_text(profile)

                        st.session_state.df = df
                        st.session_state.profile = profile
                        st.session_state.profile_text = profile_text

                        st.success(
                            f"✅ Loaded **{profile['_rows']}** rows × "
                            f"**{profile['_columns']}** columns"
                        )
                    except Exception as e:
                        st.error(f"Failed to load file: {e}")

    # ── Dataset Preview ─────────────────────────────────────────────
    if st.session_state.df is not None:
        with st.expander("📊 Dataset Preview & Profile", expanded=False):
            col1, col2, col3 = st.columns(3)
            profile = st.session_state.profile
            with col1:
                st.markdown(
                    f'<div class="profile-card"><div class="metric-value">'
                    f'{profile["_rows"]:,}</div>Rows</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f'<div class="profile-card"><div class="metric-value">'
                    f'{profile["_columns"]}</div>Columns</div>',
                    unsafe_allow_html=True,
                )
            with col3:
                total_nulls = sum(
                    info.get("null_count", 0)
                    for key, info in profile.items()
                    if not key.startswith("_")
                )
                st.markdown(
                    f'<div class="profile-card"><div class="metric-value">'
                    f"{total_nulls:,}</div>Missing Values</div>",
                    unsafe_allow_html=True,
                )

            st.dataframe(
                st.session_state.df.head(20),
                use_container_width=True,
                height=300,
            )

            st.code(st.session_state.profile_text, language="text")

    # ── Trace Sidebar ───────────────────────────────────────────────
    _render_trace_sidebar()

    # ── Chat Interface ──────────────────────────────────────────────
    if st.session_state.df is not None:
        st.markdown("---")
        st.markdown("### 💬 Ask a Question")

        # Render chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                if msg["content"] == "__PLOT__" and msg.get("plot_base64"):
                    _render_plot(msg["plot_base64"])
                else:
                    st.markdown(msg["content"])

        # Human-in-the-loop input
        if st.session_state.awaiting_human_input:
            st.warning(f"🤚 **Agent needs your help:**\n\n{st.session_state.human_question}")
            human_answer = st.text_input(
                "Your response:",
                key="human_response_input",
                placeholder="Type your clarification here...",
            )
            if st.button("Submit Response", key="submit_human_response"):
                if human_answer:
                    st.session_state.chat_history.append(
                        {"role": "user", "content": human_answer}
                    )
                    _resume_with_human_input(human_answer)
                    st.rerun()

        # Normal chat input
        else:
            if query := st.chat_input(
                "Ask about your data...",
                key="chat_input",
            ):
                st.session_state.chat_history.append(
                    {"role": "user", "content": query}
                )
                # Reset for new query
                st.session_state.trace_logs = []
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.graph = None  # Force fresh graph

                with st.chat_message("user"):
                    st.markdown(query)

                with st.chat_message("assistant"):
                    _run_analysis(query)

                st.rerun()
    else:
        st.info("👆 Upload a CSV or Excel file to get started.")


if __name__ == "__main__":
    main()
