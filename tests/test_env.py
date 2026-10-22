"""
Environment & Secrets Verification Script.
Runs a quick check to make sure our core dependencies (PyTorch, Streamlit, FAISS)
and LLM secrets are properly configured before building application logic.
"""
import sys
import torch
import streamlit as st
import faiss

def run_checks():
    print("=== Environment Verification ===")
    print(f"Python version:    {sys.version.split()[0]}")
    print(f"PyTorch version:   {torch.__version__}")
    print(f"Streamlit version: {st.__version__}")
    print(f"FAISS version:     {faiss.__version__}")
    print("--------------------------------")

    print("=== Checking Secrets ===")
    # Reads directly from .streamlit/secrets.toml
    groq_key = st.secrets.get("GROQ_API_KEY", "")
    gemini_key = st.secrets.get("GEMINI_API_KEY", "")

    if groq_key and groq_key.startswith("gsk_"):
        print("✅ Groq API Key:   Loaded successfully")
    else:
        print("❌ Groq API Key:   Missing or invalid format in secrets.toml")

    if gemini_key and gemini_key.startswith("AIza"):
        print("✅ Gemini API Key: Loaded successfully")
    else:
        print("❌ Gemini API Key: Missing or invalid format in secrets.toml")

if __name__ == "__main__":
    run_checks()
