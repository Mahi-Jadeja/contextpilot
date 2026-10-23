"""ContextPilot — Main Application Entry Point (st.navigation Shell)."""

import os
import sys
import warnings
from pathlib import Path

# Suppress environmental and package warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st

# ─── resolve project root ───────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.ui.theme import inject_custom_css, render_sidebar_header

# ─── page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="ContextPilot — Adaptive Multi-Modal Context Fusion",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── inject CSS theme (permanent dark navy) ─────────────────────
inject_custom_css()

# ─── sidebar branding & controls ────────────────────────────────
with st.sidebar:
    render_sidebar_header()
    st.markdown("---")

    # clean inline system status indicator
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            <span class="status-dot dot-green"></span>
            <span style="font-size: 0.82rem; font-weight: 600; color: #22C55E; letter-spacing: 0.5px;">SYSTEM OPERATIONAL</span>
        </div>
        <div style="font-size: 0.75rem; color: #8A99AD; margin-bottom: 12px;">
            Engine: PyTorch (CPU) | Storage: Cached FAISS
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.caption("ContextPilot v1.0.0 | SIT Pune Group 24")

# ─── navigation routing ────────────────────────────────────────
pg_overview      = st.Page("pages/1_dashboard.py",     title="Overview",            icon="▶")
pg_explorer      = st.Page("pages/2_explorer.py",      title="Query Explorer",     icon="🔍")
pg_evaluation    = st.Page("pages/3_analysis.py",       title="Evaluation & Analysis", icon="📊")
pg_tokenopt      = st.Page("pages/4_playground.py",     title="Token Optimizer",    icon="⚡")

pg = st.navigation(
    {
        "Main Menu": [pg_overview, pg_explorer],
        "Deep Dive": [pg_evaluation, pg_tokenopt]
    }
)

pg.run()
