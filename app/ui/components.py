"""
Reusable UI components for ContextPilot.
Pure rendering functions — no backend state mutations.
"""
from pathlib import Path
from typing import List, Optional, Union
import streamlit as st

from app.core.fusion import FusedCandidate


def render_stat_card(label: str, value: Union[str, int], sublabel: Optional[str] = None, icon: str = "📊"):
    """Renders a KPI metric card with dark navy surface styling."""
    sub_html = f'<div style="font-size: 0.8rem; color: #8A99AD; margin-top: 0.25rem;">{sublabel}</div>' if sublabel else ''
    html = f"""
    <div class="custom-card" style="text-align: center; padding: 1.2rem 1rem;">
        <div style="font-size: 1.4rem; margin-bottom: 0.2rem;">{icon}</div>
        <div style="font-size: 0.82rem; font-weight: 600; color: #CBD5E1; text-transform: uppercase; letter-spacing: 0.05em;">{label}</div>
        <div class="mono" style="font-size: 1.8rem; font-weight: 700; color: #69D5CB; margin-top: 0.3rem;">{value}</div>
        {sub_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_pill(text: str, variant: str = "primary"):
    """Renders a pill tag (variant: 'primary', 'neutral', 'success', 'danger')."""
    cls = "pill-primary"
    if variant == "neutral":
        cls = "pill-neutral"
    elif variant == "success":
        cls = "pill-success"
    elif variant == "danger":
        cls = "pill-danger"
    return f'<span class="pill {cls}">{text}</span>'


def render_status_dot(label: str = "", status: str = "green", is_healthy: bool = True, subtext: str = ""):
    """Renders a system status row with a glowing indicator dot."""
    if not is_healthy:
        status = "red"
    dot_cls = "dot-green" if status == "green" else ("dot-amber" if status == "amber" else "dot-red")
    
    label_html = label.replace('\n', '<br>')
    
    html = f"""
    <div class="custom-card" style="padding: 0.8rem 1rem; text-align: center; height: 100%;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 6px; margin-bottom: 4px;">
            <span class="status-dot {dot_cls}"></span>
            <span style="font-size: 0.75rem; font-weight: 600; color: #69D5CB; text-transform: uppercase;">ONLINE</span>
        </div>
        <div style="font-size: 0.82rem; color: #E2E8F0; font-weight: 500; line-height: 1.3;">
            {label_html}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_chart_card(candidate: FusedCandidate, is_packed: bool = True):
    """Renders a retrieved chart candidate card with thumbnail and scores."""
    rec = candidate.record
    score_pct = int(candidate.fused_score * 100)
    badge_variant = "success" if is_packed else "danger"
    badge_text = "✓ Packed in Context" if is_packed else "✗ Pruned (Budget Exceeded)"
    
    table_pills = " ".join([render_pill(tbl, "neutral") for tbl in rec.relevant_tables])
    
    st.markdown(f"""
    <div class="custom-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <div class="custom-card-title" style="margin: 0;">Rank #{candidate.rank} — {rec.filename}</div>
            {render_pill(badge_text, badge_variant)}
        </div>
        <div style="margin-bottom: 0.5rem;">
            {render_pill(rec.chart_type.upper(), 'primary')}
            {render_pill(rec.query_intent.capitalize(), 'neutral')}
        </div>
        <div style="font-size: 0.88rem; color: #CBD5E1; margin-bottom: 0.6rem; line-height: 1.4;">
            {rec.description}
        </div>
        <div style="display: flex; gap: 12px; margin-bottom: 0.6rem; font-size: 0.82rem;" class="mono">
            <span style="color: #69D5CB;">Text: {candidate.text_score:.3f}</span>
            <span style="color: #60A5FA;">Vision: {candidate.vision_score:.3f}</span>
            <span style="color: #F59E0B; font-weight: 700;">Fused: {candidate.fused_score:.3f} ({score_pct}%)</span>
        </div>
        <div style="font-size: 0.78rem; color: #8A99AD; margin-bottom: 0.5rem;">
            Tables: {table_pills}
        </div>
    </div>
    """, unsafe_allow_html=True)

    img_p = Path(rec.image_path)
    if img_p.exists():
        st.image(str(img_p), width="stretch")
    else:
        st.caption(f"⚠️ Chart image missing: {img_p}")


def render_token_bar(used: int, budget: int, pruned_count: int = 0):
    """Renders a horizontal token budget utilization progress bar."""
    pct = min(100, int((used / budget) * 100)) if budget > 0 else 0
    bar_color = "#22C55E" if pct <= 80 else ("#F59E0B" if pct <= 100 else "#EF4444")

    html = f"""
    <div style="margin: 1rem 0;" class="custom-card">
        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #CBD5E1; margin-bottom: 0.5rem;">
            <span>⚡ Token Budget Consumption</span>
            <span class="mono" style="font-weight: 600; color: {bar_color};">{used} / {budget} tokens ({pct}%)</span>
        </div>
        <div style="width: 100%; background-color: #1E2A44; height: 10px; border-radius: 9999px; overflow: hidden;">
            <div style="width: {pct}%; background-color: {bar_color}; height: 100%; transition: width 0.3s ease;"></div>
        </div>
        <div style="font-size: 0.78rem; color: #8A99AD; margin-top: 0.4rem;">
            Pruned charts due to limit: <strong style="color: #F8FAFC;">{pruned_count}</strong>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
