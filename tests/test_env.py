"""
Phase 0 Environment & Integration Smoke Test.
Verifies Python version, PyTorch, FAISS, Streamlit, Groq API, and Gemini API.
"""
import sys
import torch
import streamlit as st
import faiss
from groq import Groq
from google import genai

def run_checks():
    print("=== Environment Verification ===")
    print(f"Python version:    {sys.version.split()[0]}")
    print(f"PyTorch version:   {torch.__version__}")
    print(f"Streamlit version: {st.__version__}")
    print(f"FAISS version:     {faiss.__version__}")
    print("--------------------------------")

    # 1. Check Groq API
    print("=== Checking Groq API ===")
    groq_key = st.secrets.get("GROQ_API_KEY", "")
    if not groq_key:
        print("❌ Groq API Key missing from secrets.toml")
    else:
        try:
            client = Groq(api_key=groq_key)
            models = client.models.list()
            print(f"✅ Groq API connected successfully (found {len(models.data)} models)")
        except Exception as e:
            print(f"❌ Groq API error: {e}")

    # 2. Check Gemini API
    print("=== Checking Gemini API ===")
    gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    if not gemini_key:
        print("❌ Gemini API Key missing from secrets.toml")
    else:
        try:
            client = genai.Client(api_key=gemini_key)
            models = list(client.models.list())
            print(f"✅ Gemini API connected successfully (accessible model: {models[0].name})")
        except Exception as e:
            print(f"❌ Gemini API error: {e}")

if __name__ == "__main__":
    run_checks()
