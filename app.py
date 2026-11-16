from datetime import datetime
import importlib
import os
from typing import Optional
import uuid
from dotenv import load_dotenv
import pandas as pd
import streamlit as st

load_dotenv()

import agent
from agent import DataAnalysisAgentManager
import utils

# Ensure modules stay synchronized during Streamlit live updates
importlib.reload(utils)
importlib.reload(agent)

st.set_page_config(
    page_title="Autonomous Data Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .hero-container {
        padding: 1.3rem 1.6rem;
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 12px;
        color: #FFFFFF;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
    }
    .hero-title {
        font-size: 1.7rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-subtitle {
        font-size: 0.92rem;
        color: #94A3B8;
        font-weight: 400;
        margin-bottom: 0;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-online {
        background-color: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
    }
    .status-offline {
        background-color: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FECACA;
    }

    .meta-bar {
        font-size: 0.78rem;
        color: #64748B;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }

    .stExpander {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        margin-top: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
if not API_KEY:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            API_KEY = st.secrets["GEMINI_API_KEY"]
        elif "GOOGLE_API_KEY" in st.secrets:
            API_KEY = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
SAMPLE_DATASET_PATH = os.path.join(os.path.dirname(__file__), "sample_sales_data.csv")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "df" not in st.session_state:
    if os.path.exists(SAMPLE_DATASET_PATH):
        try:
            st.session_state.df = utils.load_csv(SAMPLE_DATASET_PATH)
            st.session_state.dataset_name = "sample_sales_data.csv"
        except Exception:
            st.session_state.df = None
            st.session_state.dataset_name = None
    else:
        st.session_state.df = None
        st.session_state.dataset_name = None

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None

if "agent_manager" not in st.session_state:
    st.session_state.agent_manager = None


def generate_markdown_report() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_name = st.session_state.dataset_name or "Dataset"
    df = st.session_state.df
    rows = len(df) if df is not None else 0
    cols = len(df.columns) if df is not None else 0

    lines = [
        "# Data Analysis Report",
        f"**Generated:** {now}  ",
        f"**Source:** `{dataset_name}` ({rows:,} rows, {cols} columns)  ",
        "\n---\n",
    ]

    for i, msg in enumerate(st.session_state.messages, start=1):
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"### Question: {content}\n")
        else:
            lines.append(f"{content}\n")
            steps = msg.get("intermediate_steps", [])
            if steps:
                lines.append("<details><summary>Executed Code</summary>\n")
                for s_idx, (action, obs) in enumerate(steps, start=1):
                    tool_input = getattr(action, "tool_input", "")
                    code_str = tool_input.get("code", str(tool_input)) if isinstance(tool_input, dict) else str(tool_input)
                    lines.append(f"**Step {s_idx} Code:**\n```python\n{code_str.strip()}\n```")
                    lines.append(f"**Output:**\n```\n{str(obs).strip()}\n```\n")
                lines.append("</details>\n")
            lines.append("\n---\n")

    return "\n".join(lines)


# Sidebar layout
with st.sidebar:
    st.markdown("### Data Analysis Agent")

    if API_KEY:
        st.markdown(
            '<div class="status-badge status-online">● Agent Connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge status-offline">● API Key Missing</div>',
            unsafe_allow_html=True,
        )
        st.caption("Add `GEMINI_API_KEY` to your `.env` file to start.")

    st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
    st.divider()

    st.markdown("#### Dataset")
    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Upload any tabular CSV dataset.",
    )

    if uploaded_file is not None and st.session_state.dataset_name != uploaded_file.name:
        try:
            st.session_state.df = utils.load_csv(uploaded_file)
            st.session_state.dataset_name = uploaded_file.name
            st.session_state.agent_manager = None
            st.session_state.messages = []
            st.success(f"Loaded `{uploaded_file.name}`")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading file: {e}")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Sample Data", key="btn_sample_data", use_container_width=True, help="Load built-in sales dataset"):
            if os.path.exists(SAMPLE_DATASET_PATH):
                st.session_state.df = utils.load_csv(SAMPLE_DATASET_PATH)
                st.session_state.dataset_name = "sample_sales_data.csv"
                st.session_state.agent_manager = None
                st.session_state.messages = []
                st.rerun()

    with col_btn2:
        if st.button("Clear Chat", key="btn_clear_chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if st.session_state.df is not None:
        df = st.session_state.df
        profile = utils.generate_data_profile(df)

        st.divider()
        with st.expander("Dataset Inspection", expanded=True):
            st.markdown(f"**File:** `{st.session_state.dataset_name}`")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Rows", f"{profile.get('rows', len(df)):,}")
            with c2:
                st.metric("Columns", f"{profile.get('columns', len(df.columns))}")

            mem_str = profile.get("memory_str", "N/A")
            comp_pct = profile.get("completeness_pct", 100.0)
            st.caption(f"Memory: **{mem_str}** • Completeness: **{comp_pct}%**")

            tab1, tab2, tab3 = st.tabs(["Preview", "Schema", "Stats"])
            with tab1:
                st.dataframe(profile["head_df"], use_container_width=True, height=200)
            with tab2:
                st.dataframe(profile["schema_df"], use_container_width=True, height=200)
            with tab3:
                if not profile["stats_df"].empty:
                    st.dataframe(profile["stats_df"], use_container_width=True, height=200)

    if st.session_state.messages:
        st.divider()
        report = generate_markdown_report()
        st.download_button(
            label="Download Report (.md)",
            data=report,
            file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            key="btn_download_report",
            use_container_width=True,
        )


# Main Content Area
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">📊 Autonomous Data Analysis Agent</div>
        <div class="hero-subtitle">
            Natural language exploratory analysis, statistical verification, and interactive Plotly charts.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not API_KEY:
    st.error("Missing Gemini API Key. Please add `GEMINI_API_KEY=your_key` to your `.env` file to proceed.")
    st.stop()

if st.session_state.df is None:
    st.info("Upload a CSV file in the sidebar or click 'Sample Data' to get started.")
    st.stop()

df = st.session_state.df
profile = utils.generate_data_profile(df)

# KPI Summary
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
total_rows = profile.get("rows", len(df))
total_cols = profile.get("columns", len(df.columns))
numeric_count = len(profile.get("numeric_cols", []))
comp_pct = profile.get("completeness_pct", 100.0)
missing_cells = profile.get("missing_cells", 0)
dup_count = profile.get("duplicate_count", 0)

with kpi1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{total_rows:,}</div><div class="metric-label">Total Records</div></div>', unsafe_allow_html=True)
with kpi2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{total_cols}</div><div class="metric-label">Columns ({numeric_count} numeric)</div></div>', unsafe_allow_html=True)
with kpi3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{comp_pct}%</div><div class="metric-label">Completeness ({missing_cells:,} missing)</div></div>', unsafe_allow_html=True)
with kpi4:
    dup_str = f"{dup_count:,} duplicates" if dup_count > 0 else "0 Duplicates (Clean)"
    st.markdown(f'<div class="metric-card"><div class="metric-value">Verified</div><div class="metric-label">{dup_str}</div></div>', unsafe_allow_html=True)

st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

# Sample prompt buttons
st.markdown("##### Quick Inquiries:")
col1, col2, col3 = st.columns(3)
prompt_clicked: Optional[str] = None

with col1:
    if st.button("📊 Key trends and anomalies", key="btn_prompt_trends", use_container_width=True):
        prompt_clicked = "Summarize key trends, anomalies, and performance metrics in this dataset."
with col2:
    if st.button("🔍 Missing values and outliers", key="btn_prompt_outliers", use_container_width=True):
        prompt_clicked = "Check for missing values, distribution anomalies, and statistical outliers."
with col3:
    if st.button("📈 Top 5 categories by value", key="btn_prompt_categories", use_container_width=True):
        prompt_clicked = "Plot the top 5 categories by total value with insights."

st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

# Chat conversation feed
for m_idx, msg in enumerate(st.session_state.messages):
    role = msg.get("role", "user")
    content = msg.get("content", "")

    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="📊"):
            exec_time = msg.get("execution_time")
            step_count = len(msg.get("intermediate_steps", []))
            if exec_time:
                st.markdown(
                    f'<div class="meta-bar">⚡ Computed in <b>{exec_time}s</b> &bull; {step_count} step(s) &bull; Verified in-memory</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(content)

            for f_idx, fig_item in enumerate(msg.get("figures", [])):
                fig_type = fig_item.get("type")
                fig_obj = fig_item.get("fig")
                if fig_type == "plotly":
                    st.plotly_chart(
                        fig_obj,
                        use_container_width=True,
                        key=f"hist_plot_{m_idx}_{f_idx}",
                    )
                elif fig_type == "matplotlib":
                    st.pyplot(fig_obj)

            steps = msg.get("intermediate_steps", [])
            if steps:
                with st.expander("View Agent Reasoning & Executed Code", expanded=False):
                    for idx, (action, observation) in enumerate(steps, start=1):
                        tool_input = getattr(action, "tool_input", "")
                        code_str = tool_input.get("code", str(tool_input)) if isinstance(tool_input, dict) else str(tool_input)

                        st.markdown(f"**Step {idx} Code:**")
                        st.code(code_str.strip(), language="python")

                        st.markdown("**Output:**")
                        st.code(str(observation).strip(), language="text")

                        if idx < len(steps):
                            st.divider()

# Input and execution
user_query = st.chat_input("Ask a question about your data (e.g. 'Compare sales across categories' or 'Plot profit over time')...")
active_query = prompt_clicked or user_query

if active_query:
    st.session_state.messages.append({"role": "user", "content": active_query})

    with st.chat_message("user"):
        st.markdown(active_query)

    if (
        st.session_state.agent_manager is None
        or st.session_state.agent_manager.df is not df
    ):
        st.session_state.agent_manager = DataAnalysisAgentManager(
            df=df,
            api_key=API_KEY,
            model_name="gemini-3.1-flash-lite",
            temperature=0.0,
        )

    agent_mgr: DataAnalysisAgentManager = st.session_state.agent_manager

    with st.chat_message("assistant", avatar="📊"):
        with st.spinner("Analyzing data and generating charts..."):
            result = agent_mgr.run(
                query=active_query,
                chat_history_records=st.session_state.messages[:-1],
            )

        exec_time = result.get("execution_time", 0.0)
        step_count = len(result.get("intermediate_steps", []))

        st.markdown(
            f'<div class="meta-bar">⚡ Computed in <b>{exec_time}s</b> &bull; {step_count} step(s) &bull; Verified in-memory</div>',
            unsafe_allow_html=True,
        )
        st.markdown(result["output"])

        live_turn_id = len(st.session_state.messages)
        for f_idx, fig_item in enumerate(result["figures"]):
            fig_type = fig_item.get("type")
            fig_obj = fig_item.get("fig")
            if fig_type == "plotly":
                st.plotly_chart(
                    fig_obj,
                    use_container_width=True,
                    key=f"live_plot_{live_turn_id}_{f_idx}_{uuid.uuid4().hex[:6]}",
                )
            elif fig_type == "matplotlib":
                st.pyplot(fig_obj)

        if result["intermediate_steps"]:
            with st.expander("View Agent Reasoning & Executed Code", expanded=False):
                for idx, (action, observation) in enumerate(result["intermediate_steps"], start=1):
                    tool_input = getattr(action, "tool_input", "")
                    code_str = tool_input.get("code", str(tool_input)) if isinstance(tool_input, dict) else str(tool_input)

                    st.markdown(f"**Step {idx} Code:**")
                    st.code(code_str.strip(), language="python")

                    st.markdown("**Output:**")
                    st.code(str(observation).strip(), language="text")

                    if idx < len(result["intermediate_steps"]):
                        st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["output"],
            "figures": result["figures"],
            "intermediate_steps": result["intermediate_steps"],
            "execution_time": exec_time,
        })
        st.rerun()
