"""
ContextPilot application entry point.
Sets up theme injection, page configuration, navigation routing,
and the persistent sidebar layout with system status + theme toggle.
"""

import streamlit as st

from pathlib import Path

from app.core.config import (
    CHARTS_DIR,
    METADATA_DIR,
    EMBEDDINGS_DIR,
    FAISS_DIR,
    APP_TITLE,
    APP_TAGLINE,
)
from app.ui.theme import inject_theme


# ─── page configuration ──────────────────────────────

st.set_page_config(
    page_title=f"{APP_TITLE} — {APP_TAGLINE}",
    page_icon="⛿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─── theme injection ─────────────────────────────────

inject_theme()


# ─── initialize session state defaults ───────────────

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"

if "query_history" not in st.session_state:
    st.session_state.query_history = []


# ─── sidebar header (logo + brand) ───────────────────

with st.sidebar:

    st.markdown("""
        <div style="padding: 1.25rem 0 1rem 0; border-bottom: 1px solid #1E2A44;">
            <div style="display:flex; align-items:center; gap:12px;">
                <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="16" cy="16" r="15" stroke="#69D5CB" stroke-width="1.8"/>
                    <path d="M16 6 L18 14 L26 16 L18 18 L16 26 L14 18 L6 16 L14 14 Z" fill="#69D5CB"/>
                    <rect x="9" y="20" width="3" height="5" rx="0.6" fill="#A7B4FF" opacity="0.85"/>
                    <rect x="13.75" y="17" width="3" height="8" rx="0.6" fill="#A7B4FF" opacity="0.65"/>
                    <rect x="18.5" y="22" width="3" height="3" rx="0.6" fill="#A7B4FF" opacity="0.45"/>
                </svg>
                <div>
                    <span style="font-size:1.05rem;font-weight:700;color:#F8FAFC;">{APP_TITLE}</span>
                    <br/>
                    <span style="font-size:0.72rem;color:#94A3B8;">{APP_TAGLINE}</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ─── navigation routing ──────────────────────────────

pg_overview      = st.Page("app/pages/1_dashboard.py",     title="Overview",            icon="▶")
pg_explorer      = st.Page("app/pages/2_explorer.py",      title="Query Explorer",       icon="🔍")
pg_evaluation    = st.Page("app/pages/3_analysis.py",       title="Evaluation & Analysis",icon="📊")
pg_tokenopt      = st.Page("app/pages/4_playground.py",     title="Token Optimizer",     icon="🎚️")

pg = st.navigation({
    "Home": [pg_overview],
    "Analysis": [pg_explorer, pg_evaluation],
    "Optimization": [pg_tokenopt]
})

# render the navigation — this also draws the sidebar nav list automatically
pg.run()


# ─── sidebar footer (status + toggle) ────────────────

with st.sidebar:

    st.divider()

    # lightweight status checks — no heavy loads, just filesystem reads
    chart_count   = len(list(Path(CHARTS_DIR).glob("*.png")))               if CHARTS_DIR.is_dir()   else 0
    embeddings_ok = Path(EMBEDDINGS_DIR).is_dir() and bool(list(EMBEDDINGS_DIR.glob("*")))
    faiss_ok      = Path(FAISS_DIR).is_dir()      and bool(list(FAISS_DIR.glob("*")))

    st.caption("System Status")

    st.html(f"""
        <style>.sysrow {{ padding: .3rem 0; font-size: .82rem; color: #CBD5E1;
                          display:flex; justify-content:space-between; }}
              .dot {{ height:.7rem; width:.7rem; border-radius:50%; margin-right:.5rem;
                     flex-shrink:0; display:inline-block; }}</style>

        <div class="sysrow"><span><span class="dot" style="background:#22C55E;box-shadow:0 0 6px rgba(34,197,94,.55);"></span>Chart Index</span>           <span>{chart_count} charts</span></div>
        <div class="sysrow"><span><span class="dot" style="background:{'#22C55E' if embeddings_ok else '#EF4444'};box-shadow:0 0 6px rgba(34,197,94,.55);"></span>Embeddings</span>       <span>{"Loaded" if embeddings_ok else "Missing"}</span></div>
        <div class="sysrow"><span><span class="dot" style="background:{'#22C55E' if faiss_ok else '#EF4444'};box-shadow:0 0 6px rgba(34,197,94,.55);"></span>Vector Store</span>    <span>{"Ready" if faiss_ok else "Not Built"}</span></div>
    """)

    st.divider()

    col_a, col_b = st.columns(2)
    label = "Light Mode" if st.session_state.get("theme_mode") == "dark" else "Dark Mode"
    icon  = "☀️"       if st.session_state.get("theme_mode") == "dark" else "🌙"

    with col_a:
        st.markdown(f"<span style='font-size:.82rem;'>{icon}</span>", unsafe_allow_html=True)
    with col_b:
        if st.button(label, key="theme_toggle_btn", use_container_width=True):
            new = "light" if st.session_state.theme_mode == "dark" else "dark"
            st.session_state.theme_mode = new
            st.rerun()
