"""
Theme injector and sidebar branding for ContextPilot.
Permanently enforces dark navy enterprise SaaS theme.
"""
from pathlib import Path
import streamlit as st


def inject_custom_css(mode: str = "dark"):
    """Reads styles.css and injects it into the Streamlit DOM."""
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ styles.css not found. Default theme applied.")


def render_sidebar_header():
    """Renders the branded ContextPilot logo header in the sidebar."""
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; padding: 10px 0;">
            <div style="
                background: linear-gradient(135deg, #69D5CB 0%, #3B82F6 100%);
                width: 36px;
                height: 36px;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 800;
                color: #081220;
                font-size: 1.2rem;
            ">
                ⚡
            </div>
            <div>
                <div style="font-weight: 700; font-size: 1.15rem; color: #F8FAFC; letter-spacing: -0.3px; line-height: 1.1;">
                    ContextPilot
                </div>
                <div style="font-size: 0.72rem; color: #69D5CB; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
                    Multi-Modal Fusion
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
