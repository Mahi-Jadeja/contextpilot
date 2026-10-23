"""
Query Explorer page for ContextPilot.
Core demo flow: query → fusion retrieval → token packing → grounded LLM answer.
"""

import time
import streamlit as st

from app.core.config import (
    DEFAULT_TOP_K,
    FUSION_WEIGHT_TEXT,
    DEFAULT_TOKEN_BUDGET,
    MIN_TOKEN_BUDGET,
    MAX_TOKEN_BUDGET,
)
from app.core.data_loader import get_sample_queries
from app.core.fusion import run_fusion_pipeline
from app.core.token_optimizer import pack_context_within_budget
from app.core.llm_client import generate_answer
from app.ui.components import render_chart_card, render_token_bar


# ─── PAGE HEADER ─────────────────────────────────────

st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <div style="font-size: 0.82rem; font-weight: 600; color: #69D5CB; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.2rem;">
        Multimodal Context Fusion
    </div>
    <h2 style="font-size: 1.8rem; font-weight: 700; color: #F8FAFC; margin: 0 0 0.4rem 0;">
        Query Explorer
    </h2>
    <p style="font-size: 0.95rem; color: #94A3B8; margin: 0;">
        Ask a business question to retrieve chart contexts, pack evidence under a token budget, and generate a grounded AI insight.
    </p>
</div>
""", unsafe_allow_html=True)


# ─── SIDEBAR CONTROLS ────────────────────────────────

with st.sidebar:
    st.markdown("<div style='font-size: 0.9rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.5rem;'>Fusion & Budget Controls</div>", unsafe_allow_html=True)

    top_k = st.slider("Top-K Candidates", 1, 10, DEFAULT_TOP_K, step=1)
    token_budget = st.slider("Token Budget", MIN_TOKEN_BUDGET, MAX_TOKEN_BUDGET, DEFAULT_TOKEN_BUDGET, step=50)

    st.markdown("<div style='font-size: 0.85rem; color: #CBD5E1; margin-top: 0.8rem;'>Embedding Weights</div>", unsafe_allow_html=True)
    w_text = st.slider("Text Similarity Weight", 0.0, 1.0, FUSION_WEIGHT_TEXT, step=0.05)
    # vision weight is the complement so weights always sum to 1
    w_vision = round(1.0 - w_text, 2)
    st.caption(f"Vision Weight (auto): {w_vision}")


# ─── QUERY INPUT ─────────────────────────────────────

samples = get_sample_queries(limit=5)
# key is query_text (confirmed against data_loader output)
sample_queries = [s["query_text"] for s in samples] if samples else [
    "How has total order revenue trended over the last 18 months?",
    "What is the return rate across product categories?",
    "Compare prepaid vs COD sales across regions.",
    "Which seller category generates the highest average rating?",
    "Show the distribution of customer orders by payment method.",
]

# chips write straight into the widget's own state key —
# a keyed widget ignores any other session variable
if "query_text_area" not in st.session_state:
    st.session_state.query_text_area = sample_queries[0]

st.markdown("<div style='font-size: 0.85rem; font-weight: 500; color: #CBD5E1; margin-bottom: 0.4rem;'>Showcase queries:</div>", unsafe_allow_html=True)
chip_cols = st.columns(len(sample_queries))
for idx, q_text in enumerate(sample_queries):
    with chip_cols[idx]:
        if st.button(f"Q{idx + 1}", key=f"chip_{idx}", use_container_width=True, help=q_text):
            st.session_state.query_text_area = q_text

user_query = st.text_area(
    "Enter your business question:",
    height=80,
    placeholder="e.g. Show quarterly revenue growth by product line...",
    key="query_text_area",
)

run_search = st.button("🚀 Retrieve Context & Generate Answer", type="primary")


# ─── PIPELINE — runs only on explicit button press ────

if run_search:
    query_to_process = user_query.strip()

    if not query_to_process:
        st.warning("⚠️ Enter a query or pick a showcase query above.")
    else:
        with st.spinner("Encoding query, running fusion search..."):
            t0 = time.perf_counter()
            fused_candidates = run_fusion_pipeline(
                query=query_to_process,
                top_k=top_k,
                w_text=w_text,
                w_vision=w_vision,
            )
            retrieval_ms = (time.perf_counter() - t0) * 1000

            opt_context = pack_context_within_budget(
                query=query_to_process,
                candidates=fused_candidates,
                budget=token_budget,
            )

            llm_resp = generate_answer(opt_context)

        # stash results + the settings that produced them
        st.session_state.last_fused = fused_candidates
        st.session_state.last_opt = opt_context
        st.session_state.last_llm = llm_resp
        st.session_state.retrieval_ms = retrieval_ms
        st.session_state.last_settings = (top_k, token_budget, w_text)

# honest stale signal instead of silently old results
if "last_settings" in st.session_state:
    if st.session_state.last_settings != (top_k, token_budget, w_text):
        st.caption("⚙️ Settings changed since last run — press Retrieve to apply.")


# ─── RESULTS ─────────────────────────────────────────

if "last_fused" in st.session_state:
    fused_candidates = st.session_state.last_fused
    opt_context = st.session_state.last_opt
    llm_resp = st.session_state.last_llm
    retrieval_ms = st.session_state.retrieval_ms

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    col_results, col_llm = st.columns([1.1, 1], gap="large")

    # LEFT — retrieved chart cards
    with col_results:
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
            <div class="custom-card-title">Top Retrieved Chart Contexts ({len(fused_candidates)})</div>
            <span class="mono" style="font-size: 0.8rem; color: #69D5CB;">Retrieval: {retrieval_ms:.1f} ms</span>
        </div>
        """, unsafe_allow_html=True)

        for cand in fused_candidates:
            render_chart_card(cand, rank=cand.rank)

    # RIGHT — token budget + LLM answer
    with col_llm:
        st.markdown('<div class="custom-card-title" style="margin-bottom: 0.8rem;">Token Budget & LLM Answer</div>', unsafe_allow_html=True)

        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        render_token_bar(opt_context.estimated_tokens, token_budget)
        st.markdown(f"""
        <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 0.4rem;">
            Selected: <strong style="color: #22C55E;">{len(opt_context.selected_candidates)} charts</strong> &nbsp;|&nbsp;
            Pruned: <strong style="color: #EF4444;">{len(opt_context.dropped_candidates)} charts</strong>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="custom-card">', unsafe_allow_html=True)

        # teal for groq, periwinkle for gemini, amber for mock
        prov_color = {"groq": "#69D5CB", "gemini": "#A7B4FF"}.get(llm_resp.provider, "#F59E0B")
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
            <span style="font-size: 0.85rem; font-weight: 600; color: #F8FAFC;">AI Grounded Insight</span>
            <span class="pill mono" style="background-color: rgba(105,213,203,0.15); color: {prov_color}; border: 1px solid {prov_color};">
                {llm_resp.provider.upper()} · {llm_resp.latency_ms:.0f} ms
            </span>
        </div>
        <div class="mono" style="font-size: 0.78rem; color: #64748B; margin-bottom: 0.8rem;">
            {llm_resp.model} · tokens in/out: {llm_resp.input_tokens}/{llm_resp.output_tokens}
        </div>
        """, unsafe_allow_html=True)

        # mock responses carry an explanation in error_message — show as info, not error
        if llm_resp.provider == "mock":
            st.info(llm_resp.answer)
        else:
            st.markdown(f"""
            <div style="font-size: 0.92rem; color: #F8FAFC; line-height: 1.6; background-color: #0A0E20; padding: 1rem; border-radius: 8px; border: 1px solid #1E2A44;">
                {llm_resp.answer}
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
