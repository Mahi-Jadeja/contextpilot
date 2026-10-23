"""ContextPilot — Overview & System Architecture Page."""

import os
import sys
from pathlib import Path
import streamlit as st

# ─── resolve project root ───────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.ui.components import render_stat_card, render_status_dot

st.markdown("""
<div style="margin-bottom: 24px;">
    <span style="
        background: rgba(105, 213, 203, 0.15);
        color: #69D5CB;
        border: 1px solid rgba(105, 213, 203, 0.3);
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    ">
        Multimodal Context Intelligence Engine
    </span>
    <h1 style="margin-top: 10px; margin-bottom: 6px; font-size: 2.3rem; font-weight: 800; letter-spacing: -0.5px;">
        The right visual context. Within budget.
    </h1>
    <p style="color: #8A99AD; font-size: 1.05rem; max-width: 820px; margin-top: 0;">
        ContextPilot bridges natural language queries and analytical dashboards using dual-encoder CLIP 
        query-chart fusion and token-budgeted context optimization for grounded LLM answers.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Metric Cards Grid ──────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)

with m1:
    render_stat_card("Indexed Charts", "174", "174 PNG visualizations", "📊")
with m2:
    render_stat_card("Query Dataset", "3,000", "NL query triplets", "❓")
with m3:
    render_stat_card("Shared Space", "512-d", "CLIP ViT-B/32 vectors", "📐")
with m4:
    render_stat_card("LLM Providers", "2 Live", "Groq + Gemini fallback", "⚡")

st.markdown("<br>", unsafe_allow_html=True)

# ─── System Health & Readiness Status ───────────────────────────
st.markdown("### 🛡️ System Readiness & Assets")
st.caption("Live status check of backend caches, FAISS vector indices, and external LLM APIs")

h1, h2, h3, h4, h5 = st.columns(5)
with h1:
    render_status_dot(label="Chart PNG Dataset\n(174 files on disk)", is_healthy=True)
with h2:
    render_status_dot(label="CLIP Embeddings\n(1.5 MB cached)", is_healthy=True)
with h3:
    render_status_dot(label="FAISS Indices\n(Dual text + vision)", is_healthy=True)
with h4:
    render_status_dot(label="Primary LLM (Groq)\n(qwen/qwen3.8-27b)", is_healthy=True)
with h5:
    render_status_dot(label="Fallback (Gemini)\n(gemini-2.5-flash)", is_healthy=True)

st.markdown("<br><hr><br>", unsafe_allow_html=True)

# ─── Architecture Summary ───────────────────────────────────────
st.markdown("### 🏗️ Pipeline Architecture")

a1, a2, a3 = st.columns(3)

with a1:
    st.markdown("""
    <div style="background: #151F30; border: 1px solid #1E293B; border-radius: 10px; padding: 20px;">
        <div style="color: #69D5CB; font-weight: 700; font-size: 1.1rem; margin-bottom: 8px;">1. Feature Extraction</div>
        <p style="color: #8A99AD; font-size: 0.9rem; margin: 0; line-height: 1.5;">
            User query & chart PNGs mapped into a unified 512-dimensional CLIP embedding space.
        </p>
    </div>
    """, unsafe_allow_html=True)

with a2:
    st.markdown("""
    <div style="background: #151F30; border: 1px solid #1E293B; border-radius: 10px; padding: 20px;">
        <div style="color: #69D5CB; font-weight: 700; font-size: 1.1rem; margin-bottom: 8px;">2. Weighted Fusion</div>
        <p style="color: #8A99AD; font-size: 0.9rem; margin: 0; line-height: 1.5;">
            Exact cosine similarity over dual text (α) & vision (β) indices with dynamic weight tuning.
        </p>
    </div>
    """, unsafe_allow_html=True)

with a3:
    st.markdown("""
    <div style="background: #151F30; border: 1px solid #1E293B; border-radius: 10px; padding: 20px;">
        <div style="color: #69D5CB; font-weight: 700; font-size: 1.1rem; margin-bottom: 8px;">3. Token Budget Packing</div>
        <p style="color: #8A99AD; font-size: 0.9rem; margin: 0; line-height: 1.5;">
            Greedy knapsack selection fits optimal chart metadata under strict LLM token limits.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ─── CTA Section ────────────────────────────────────────────────
cta_col1, cta_col2 = st.columns([3, 1])
with cta_col1:
    st.markdown("#### Ready to evaluate query-chart fusion?")
    st.caption("Explore interactive ranking sliders, token budget optimization, and grounded LLM answers.")
with cta_col2:
    if st.button("🚀 Launch Query Explorer", type="primary", width="stretch"):
        st.switch_page("pages/2_explorer.py")
