"""ContextPilot — Query Explorer Page."""

import os
import sys
from pathlib import Path
import streamlit as st

# ─── resolve project root ───────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.fusion import run_fusion_pipeline
from app.core.token_optimizer import pack_context_within_budget
from app.core.llm_client import generate_answer
from app.ui.components import (
    render_chart_card,
    render_token_bar,
    render_pill
)

st.markdown("""
<div style="margin-bottom: 20px;">
    <h1 style="margin: 0; font-size: 2rem;">🔍 Multi-Modal Context Fusion Explorer</h1>
    <p style="color: #8A99AD; margin-top: 5px;">
        Search financial charts, adjust hybrid ranking weights in real time, pack token budget, and observe grounded LLM reasoning.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── 5 Curated Showcase Queries ─────────────────────────────────
CURATED_QUERIES = [
    {
        "label": "💰 Category Revenue",
        "query": "Which product category generates the most revenue?"
    },
    {
        "label": "📦 Return Rate",
        "query": "What is the return rate for each product category?"
    },
    {
        "label": "📈 Revenue Trend",
        "query": "What is the weekly revenue trend over time?"
    },
    {
        "label": "📊 Sub-Category Volume",
        "query": "Which sub-categories have the highest order volume?"
    },
    {
        "label": "📅 Monthly Category Drift",
        "query": "How has order volume changed per category over months?"
    }
]

# ─── Sidebar Controls ───────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Pipeline Settings")
    top_k = st.slider("Top-K Candidates", min_value=1, max_value=10, value=4, step=1)
    token_budget = st.slider("Token Budget Limit", min_value=100, max_value=1000, value=500, step=50)

    st.markdown("---")
    st.markdown("### ⚖️ Fusion Weights")
    alpha_weight = st.slider("Text Weight (α)", min_value=0.0, max_value=1.0, value=0.55, step=0.05)
    beta_weight = round(1.0 - alpha_weight, 2)
    st.caption(f"Vision Weight (β): **{beta_weight}**")

# ─── Query Input & Suggestion Chips ─────────────────────────────
if "query_input" not in st.session_state:
    st.session_state.query_input = CURATED_QUERIES[0]["query"]

st.markdown("##### 💡 Curated Analytical Showcase Queries")
chip_cols = st.columns(5)

def set_selected_query(q_text: str):
    st.session_state.query_input = q_text

for idx, item in enumerate(CURATED_QUERIES):
    with chip_cols[idx]:
        st.button(
            item["label"],
            key=f"chip_{idx}",
            help=item["query"],
            width="stretch",
            on_click=set_selected_query,
            args=(item["query"],)
        )

user_query = st.text_area(
    "Enter Financial Query:",
    key="query_input",
    height=80
)

col_btn, _ = st.columns([1, 3])
with col_btn:
    run_pressed = st.button("🚀 Run Fusion Pipeline", type="primary", width="stretch")

# ─── Pipeline Execution ────────────────────────────────────────
if run_pressed or st.session_state.get("has_run_explorer", False):
    if not user_query or not user_query.strip():
        st.warning("⚠️ Please enter a query or select a showcase query above before running.")
    else:
        st.session_state.has_run_explorer = True

        with st.spinner("Executing hybrid retrieval, vision scoring, dynamic fusion, and context packing..."):
            fused_candidates = run_fusion_pipeline(
                query=user_query.strip(),
                top_k=top_k,
                w_text=alpha_weight,
                w_vision=beta_weight
            )

            opt_context = pack_context_within_budget(
                query=user_query.strip(),
                candidates=fused_candidates,
                token_budget=token_budget
            )

            llm_response = generate_answer(opt_context)

        st.markdown("---")

        # 2-column layout: Left = Retrieved Charts, Right = Token Budget & LLM Answer
        left_col, right_col = st.columns([1, 1], gap="large")

        packed_filenames = [c.record.filename for c in opt_context.selected_candidates]

        with left_col:
            st.markdown("### 📊 Retrieved & Ranked Context Charts")
            st.caption(f"Showing Top-{len(fused_candidates)} retrieved visual context pairs (Ranked by Fused Score)")

            for candidate in fused_candidates:
                is_packed = candidate.record.filename in packed_filenames
                render_chart_card(candidate=candidate, is_packed=is_packed)

        with right_col:
            st.markdown("### ⚡ Dynamic Token Budget & LLM Response")

            # Token usage visual bar
            render_token_bar(
                used=opt_context.estimated_tokens,
                budget=token_budget,
                pruned_count=len(opt_context.dropped_candidates)
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Selected vs Dropped summary
            st.markdown("**Context Selection Breakdown:**")
            pill_cols = st.columns(2)
            with pill_cols[0]:
                st.caption("Included Charts:")
                for sc in opt_context.selected_candidates:
                    st.markdown(render_pill(f"✓ {sc.record.filename}", "success"), unsafe_allow_html=True)
            with pill_cols[1]:
                st.caption("Pruned Charts:")
                if opt_context.dropped_candidates:
                    for dc in opt_context.dropped_candidates:
                        st.markdown(render_pill(f"✗ {dc.record.filename}", "danger"), unsafe_allow_html=True)
                else:
                    st.caption("_None (All fit in budget)_")

            st.markdown("<br>", unsafe_allow_html=True)

            # Grounded LLM Response Box
            st.markdown("#### 🤖 Grounded Insight (Groq / Gemini)")

            latency_sec = llm_response.latency_ms / 1000.0 if llm_response.latency_ms else 0.0
            provider_badge = f"**Provider:** `{llm_response.provider.upper()}` | **Model:** `{llm_response.model}` | **Latency:** `{latency_sec:.2f}s`"
            st.caption(provider_badge)

            if llm_response.error_message:
                st.error(f"Inference Warning: {llm_response.error_message}")

            st.markdown(
                f"""
                <div style="
                    background: #151F30;
                    border: 1px solid rgba(105, 213, 203, 0.3);
                    border-radius: 8px;
                    padding: 18px;
                    font-size: 0.95rem;
                    line-height: 1.6;
                    color: #E2E8F0;
                    margin-top: 10px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                ">
                    {llm_response.answer}
                </div>
                """,
                unsafe_allow_html=True
            )
