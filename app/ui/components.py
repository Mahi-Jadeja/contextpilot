"""
Reusable UI components for ContextPilot.
Pure rendering functions — no pipeline logic lives here.
"""
from pathlib import Path
from typing import List, Optional, Union
import streamlit as st

from app.core.data_loader import ChartRecord
from app.core.fusion import FusedCandidate


def render_stat_card(label: str, value: Union[str, int], sublabel: Optional[str] = None):
    """Renders a KPI metric card with dark navy surface styling."""
    sub_html = f'<div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.25rem;">{sublabel}</div>' if sublabel else ''
    html = f"""
    <div class="custom-card" style="text-align: center; padding: 1.2rem 1rem;">
        <div style="font-size: 0.85rem; font-weight: 500; color: #CBD5E1; text-transform: uppercase; letter-spacing: 0.05em;">{label}</div>
        <div class="mono" style="font-size: 1.8rem; font-weight: 700; color: #69D5CB; margin-top: 0.4rem;">{value}</div>
        {sub_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_pill(text: str, variant: str = "primary"):
    """Renders a pill tag (variant: 'primary' or 'neutral')."""
    cls = "pill-primary" if variant == "primary" else "pill-neutral"
    return f'<span class="pill {cls}">{text}</span>'


def render_status_dot(label: str, status: str = "green", subtext: str = ""):
    """Renders a system status row with a glowing indicator dot."""
    dot_cls = "dot-green" if status == "green" else ("dot-amber" if status == "amber" else "dot-red")
    sub = f' <span style="font-size: 0.8rem; color: #64748B;">({subtext})</span>' if subtext else ''
    html = f"""
    <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.4rem 0;">
        <span style="font-size: 0.9rem; color: #CBD5E1; display: flex; align-items: center;">
            <span class="status-dot {dot_cls}"></span>{label}
        </span>
        <span style="font-size: 0.85rem; font-weight: 600; color: #F8FAFC;">{sub}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_chart_card(candidate: FusedCandidate, rank: Optional[int] = None):
    """
    Renders a retrieved chart candidate card with image, relevance score,
    chart type, query intent, and DB table tags.
    """
    rec = candidate.record
    rank_prefix = f"#{rank} " if rank is not None else ""
    score_pct = int(candidate.fused_score * 100)
    
    # generate HTML table pills
    table_pills = "".join([render_pill(tbl, "neutral") for tbl in rec.relevant_tables])
    
    st.markdown(f"""
    <div class="custom-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <div class="custom-card-title">{rank_prefix}{rec.chart_id}</div>
            <div>
                <span class="pill pill-primary mono">Score: {candidate.fused_score:.2f} ({score_pct}%)</span>
            </div>
        </div>
        <div style="margin-bottom: 0.5rem;">
            {render_pill(rec.chart_type.upper(), 'primary')}
            {render_pill(rec.query_intent.capitalize(), 'neutral')}
        </div>
        <div style="font-size: 0.88rem; color: #CBD5E1; margin-bottom: 0.75rem;">
            {rec.description}
        </div>
        <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 0.5rem;">
            Relevant Tables: {table_pills}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # render image cleanly if present on disk
    if rec.image_path.exists():
        st.image(str(rec.image_path), use_container_width=True)
    else:
        st.caption(f"⚠️ Chart image missing: {rec.filename}")


def render_token_bar(used_tokens: int, budget_tokens: int):
    """Renders a horizontal token budget utilization progress bar."""
    pct = min(100, int((used_tokens / budget_tokens) * 100)) if budget_tokens > 0 else 0
    bar_color = "#22C55E" if pct <= 80 else ("#F59E0B" if pct <= 100 else "#EF4444")

    html = f"""
    <div style="margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #CBD5E1; margin-bottom: 0.3rem;">
            <span>Token Budget Consumption</span>
            <span class="mono" style="font-weight: 600; color: {bar_color};">{used_tokens} / {budget_tokens} tokens ({pct}%)</span>
        </div>
        <div style="width: 100%; background-color: #1E2A44; height: 10px; border-radius: 9999px; overflow: hidden;">
            <div style="width: {pct}%; background-color: {bar_color}; height: 100%; transition: width 0.3s ease;"></div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
