"""
Theme injector for ContextPilot.
Reads custom CSS and applies it to the Streamlit app.
"""
from pathlib import Path
import streamlit as st

def inject_theme():
    """Reads styles.css and injects it via st.markdown."""
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ styles.css not found. Default theme applied.")

