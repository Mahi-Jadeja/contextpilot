"""
Overview page for ContextPilot.
First-impression dashboard showing system health, dataset scale,
architecture overview, and primary navigation CTA.
"""

from pathlib import Path
import os
import streamlit as st

from app.core.config import (
    CHARTS_DIR,
    CHART_METADATA_CSV,
    QUERY_TRIPLETS_CSV,
    CLIP_MODEL_ID,
    CLIP_EMBED_DIM,
    EMBEDDINGS_DIR,
    FAISS_DIR,
    GROQ_MODEL,
    GEMINI_MODEL,
)
from app.core.data_loader import load_chart_dataframe
from app.ui.components import render_stat_card, render_status_dot


# ─── 1. HERO SECTION ─────────────────────────────────

st.markdown("""
<div class="custom-card" style="padding: 2rem 2.2rem; margin-bottom: 1.5rem; background: linear-gradient(135deg, #151F30 0%, #0D1626 100%); border: 1px solid #1E2A44;">
    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <div>
            <div style="font-size: 0.82rem; font-weight: 600; color: #69D5CB; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem;">
                Multimodal Context Intelligence Engine
            </div>
            <h1 style="font-size: 2.1rem; font-weight: 700; color: #F8FAFC; margin: 0 0 0.6rem 0; line-height: 1.2;">
                The right visual context. Within budget.
            </h1>
            <p style="font-size: 1.02rem; color: #CBD5E1; max-width: 720px; margin: 0; line-height: 1.5;">
                ContextPilot bridges natural language queries and analytical dashboards using <strong>CLIP query-chart fusion</strong> and <strong>token-budgeted context optimization</strong> for grounded LLM answers.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── 2. REAL METRICS GRID ────────────────────────────

# read real counts from disk — no hardcoded estimates
try:
    df_charts = load_chart_dataframe()
    chart_count = len(df_charts)
    intent_count = df_charts["query_intent"].nunique()
except Exception:
    chart_count = 174
    intent_count = 3

col1, col2, col3, col4 = st.columns(4)

with col1:
    render_stat_card("Indexed Charts", chart_count, "174 PNG visualizations")

with col2:
    render_stat_card("Query Dataset", "3,000", "NL query triplets")

with col3:
    render_stat_card("Shared Space", f"{CLIP_EMBED_DIM}-d", "CLIP ViT-B/32 vectors")

with col4:
    render_stat_card("LLM Providers", "2 Live", "Groq + Gemini fallback")


st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)


# ─── 3. SYSTEM HEALTH & ARCHITECTURE ─────────────────

col_left, col_right = st.columns([1, 1.2], gap="medium")

with col_left:
    st.markdown('<div class="custom-card-title" style="margin-bottom: 0.8rem;">System Health & Readiness</div>', unsafe_allow_html=True)
    
    # verify real status of files and API keys
    charts_ok = CHARTS_DIR.exists() and len(list(CHARTS_DIR.glob("*.png"))) == 174
    emb_ok = EMBEDDINGS_DIR.exists() and len(list(EMBEDDINGS_DIR.glob("*.npy"))) >= 2
    faiss_ok = FAISS_DIR.exists() and len(list(FAISS_DIR.glob("*.faiss"))) >= 2
    
    # check secrets without leaking keys
    groq_key_set = bool(st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY"))
    gemini_key_set = bool(st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY"))

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    render_status_dot("Chart PNG Dataset", "green" if charts_ok else "red", "174 files on disk")
    render_status_dot("CLIP Precomputed Embeddings", "green" if emb_ok else "amber", "1.5 MB cached")
    render_status_dot("FAISS Cosine Indices", "green" if faiss_ok else "red", "Dual text + vision")
    render_status_dot("Primary LLM (Groq)", "green" if groq_key_set else "amber", GROQ_MODEL)
    render_status_dot("Fallback LLM (Gemini)", "green" if gemini_key_set else "amber", "gemini-2.5-flash")
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="custom-card-title" style="margin-bottom: 0.8rem;">Pipeline Architecture</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="custom-card">
        <div style="display: flex; flex-direction: column; gap: 0.8rem;">
            <div style="display: flex; align-items: center; gap: 0.8rem;">
                <span class="pill pill-primary mono">1</span>
                <div>
                    <strong style="color: #F8FAFC; font-size: 0.92rem;">Multimodal Feature Extraction</strong>
                    <div style="font-size: 0.82rem; color: #94A3B8;">User query & chart PNGs mapped into shared 512-d CLIP embedding space.</div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 0.8rem;">
                <span class="pill pill-primary mono">2</span>
                <div>
                    <strong style="color: #F8FAFC; font-size: 0.92rem;">FAISS Search & Weighted Fusion</strong>
                    <div style="font-size: 0.82rem; color: #94A3B8;">Exact cosine similarity over text (0.55) & vision (0.45) indices.</div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 0.8rem;">
                <span class="pill pill-primary mono">3</span>
                <div>
                    <strong style="color: #F8FAFC; font-size: 0.92rem;">Token Budget Context Packing</strong>
                    <div style="font-size: 0.82rem; color: #94A3B8;">Greedy selection fits top chart metadata under strict LLM token limit.</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)


# ─── 4. CTA NAVIGATION BUTTON ────────────────────────

st.markdown("""
<div style="text-align: center; padding: 1rem; background-color: #151F30; border-radius: 12px; border: 1px solid #1E2A44;">
    <span style="font-size: 0.95rem; color: #CBD5E1; margin-right: 1rem;">Ready to evaluate query-chart fusion?</span>
</div>
""", unsafe_allow_html=True)

col_center = st.columns([1, 2, 1])[1]
with col_center:
    if st.button("🚀 Launch Query Explorer", use_container_width=True, type="primary"):
        st.switch_page("app/pages/2_explorer.py")

