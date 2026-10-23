import streamlit as st
from app.ui.theme import inject_custom_css, render_sidebar_header
from app.ui.components import render_token_bar
from app.core.fusion import run_fusion_pipeline
from app.core.token_optimizer import pack_context_within_budget

st.set_page_config(page_title="Token Optimizer | ContextPilot", layout="wide")
inject_custom_css()
render_sidebar_header()

st.title("⚙️ Token Optimizer Playground")
st.markdown(
    """
    <div style='color: #8B949E; margin-bottom: 2rem;'>
        Dynamically adjust the strict context window limit. ContextPilot will automatically prioritize and pack 
        the highest-scoring multimodal charts (visual descriptions + structured metadata) to fit the budget, dropping lower-ranked candidates to prevent LLM hallucination and context collapse.
    </div>
    """, 
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Configuration")
    query = st.text_input("Test Query", value="Which product category generates the most revenue?", key="opt_query")
    target_k = st.slider("Initial Candidates (Top-K)", min_value=5, max_value=20, value=10, step=1)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Token Constraints")
    token_budget = st.slider("Strict Token Budget", min_value=500, max_value=5000, value=1200, step=100)
    
    if st.button("Run Optimizer", type="primary", width="stretch"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Retrieving and packing context..."):
                # run_fusion_pipeline returns a plain list[FusedCandidate] directly
                candidates = run_fusion_pipeline(query, top_k=target_k, w_text=0.5, w_vision=0.5)
                optimized_ctx = pack_context_within_budget(query, candidates, token_budget)
                st.session_state.opt_result = optimized_ctx

with col2:
    st.markdown("### Packing Results")
    
    if "opt_result" in st.session_state:
        ctx = st.session_state.opt_result
        
        render_token_bar(ctx.estimated_tokens, ctx.token_budget)
        
        st.markdown(f"**Included Candidates ({len(ctx.selected_candidates)}):**")
        if ctx.selected_candidates:
            for c in ctx.selected_candidates:
                st.markdown(
                    f"<div style='padding: 8px 12px; background: rgba(105, 213, 203, 0.1); border-left: 3px solid #69D5CB; margin-bottom: 8px; border-radius: 4px;'>"
                    f"<span style='color: #69D5CB; font-weight: bold;'>Included</span> &nbsp;|&nbsp; "
                    f"Score: {c.fused_score:.3f} &nbsp;|&nbsp; {c.record.filename}"
                    f"</div>", 
                    unsafe_allow_html=True
                )
        else:
            st.info("Budget too small to include any context.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**Pruned Candidates ({len(ctx.dropped_candidates)}):**")
        if ctx.dropped_candidates:
            for c in ctx.dropped_candidates:
                st.markdown(
                    f"<div style='padding: 8px 12px; background: rgba(248, 81, 73, 0.1); border-left: 3px solid #F85149; margin-bottom: 8px; border-radius: 4px;'>"
                    f"<span style='color: #F85149; font-weight: bold;'>Pruned</span> &nbsp;|&nbsp; "
                    f"Score: {c.fused_score:.3f} &nbsp;|&nbsp; {c.record.filename}"
                    f"</div>", 
                    unsafe_allow_html=True
                )
        else:
            st.success("All initial candidates successfully packed into the budget!")

        with st.expander("View Final Assembled Prompt (Passed to LLM)"):
            st.code(ctx.full_prompt_text, language="markdown")
            
    else:
        st.info("Adjust the parameters on the left and click 'Run Optimizer' to visualize token packing.")

