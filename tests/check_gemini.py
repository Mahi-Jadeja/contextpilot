"""
Checks whether our Gemini API key can authenticate.

A successful request is more reliable than checking the key prefix.
"""

import streamlit as st
from google import genai


def main():
    api_key = st.secrets.get("GEMINI_API_KEY", "")

    if not api_key:
        print("FAIL: GEMINI_API_KEY is missing.")
        return

    try:
        client = genai.Client(api_key=api_key)

        # Listing models checks authentication without generating content.
        models = client.models.list()

        first_model = next(iter(models), None)

        if first_model:
            print("PASS: Gemini authentication successful.")
            print(f"Accessible model: {first_model.name}")
        else:
            print("Connected, but no models were returned.")

    except Exception as error:
        # Avoid printing exception details that could contain credentials.
        print(f"FAIL: Gemini authentication failed ({type(error).__name__}).")


if __name__ == "__main__":
    main()
