"""
Unified LLM Client for ContextPilot with Multi-Model Auto-Retry.
Primary: Groq (auto-tries available models)
Fallback: Google Gemini (auto-tries available flash models)
Safety: Mock offline fallback
"""
import os
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from app.core.config import LLM_MAX_OUTPUT_TOKENS, LLM_TEMPERATURE, ROOT_DIR
from app.core.token_optimizer import OptimizedContext

# Ordered list of Groq models to try
GROQ_MODEL_CANDIDATES = [
    "qwen/qwen3.8-27b",
    "allam-2-7b",
    "openai/gpt-oss-20b",
]

# Ordered list of Gemini Flash models to try
GEMINI_MODEL_CANDIDATES = [
    "models/gemini-2.5-flash",
    "models/gemini-flash-latest",
    "models/gemini-3.5-flash",
]


@dataclass
class LLMResponse:
    answer: str
    provider: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    error_message: Optional[str] = None


def get_api_key(key_name: str) -> Optional[str]:
    """Retrieves API key from env, Streamlit secrets, or secrets.toml."""
    val = os.getenv(key_name)
    if val and val.strip():
        return val.strip()

    try:
        import streamlit as st
        if hasattr(st, "secrets") and key_name in st.secrets:
            s_val = str(st.secrets[key_name]).strip()
            if s_val:
                return s_val
    except Exception:
        pass

    secrets_file = ROOT_DIR / ".streamlit" / "secrets.toml"
    if secrets_file.is_file():
        try:
            with secrets_file.open("rb") as f:
                sec_dict = tomllib.load(f)
                if key_name in sec_dict and str(sec_dict[key_name]).strip():
                    return str(sec_dict[key_name]).strip()
        except Exception:
            pass

    return None


def call_groq(opt_context: OptimizedContext) -> LLMResponse:
    """Tries Groq models sequentially until one succeeds."""
    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not available.")

    from groq import Groq

    client = Groq(api_key=api_key)
    errors = []

    for model_id in GROQ_MODEL_CANDIDATES:
        try:
            start_time = time.perf_counter()
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": opt_context.system_prompt},
                    {"role": "user", "content": opt_context.user_prompt},
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_OUTPUT_TOKENS,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            content = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", opt_context.estimated_tokens) if usage else opt_context.estimated_tokens
            out_tok = getattr(usage, "completion_tokens", 0) if usage else 0

            return LLMResponse(
                answer=content.strip(),
                provider="groq",
                model=model_id,
                latency_ms=round(elapsed_ms, 2),
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
        except Exception as e:
            errors.append(f"{model_id}: {e}")

    raise RuntimeError(f"All Groq model candidates failed: {'; '.join(errors)}")


def call_gemini(opt_context: OptimizedContext) -> LLMResponse:
    """Tries Gemini models sequentially until one succeeds."""
    api_key = get_api_key("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not available.")

    from google import genai

    client = genai.Client(api_key=api_key)
    prompt = f"{opt_context.system_prompt}\n\n{opt_context.user_prompt}"
    errors = []

    for model_id in GEMINI_MODEL_CANDIDATES:
        try:
            start_time = time.perf_counter()
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            answer = getattr(response, "text", "") or ""
            if answer.strip():
                return LLMResponse(
                    answer=answer.strip(),
                    provider="gemini",
                    model=model_id,
                    latency_ms=round(elapsed_ms, 2),
                    input_tokens=opt_context.estimated_tokens,
                    output_tokens=len(answer.split()),
                )
        except Exception as e:
            errors.append(f"{model_id}: {e}")

    raise RuntimeError(f"All Gemini model candidates failed: {'; '.join(errors)}")


def call_mock(opt_context: OptimizedContext) -> LLMResponse:
    """Offline safety fallback."""
    start_time = time.perf_counter()

    top_chart = opt_context.selected_candidates[0].record if opt_context.selected_candidates else None
    chart_info = f"{top_chart.chart_id} ({top_chart.chart_type})" if top_chart else "chart context"

    mock_text = (
        f"[MOCK ANSWER - Offline Mode]\n\n"
        f"Based on retrieved visual context ({chart_info}), "
        f"the query '{opt_context.query}' matches relevant metrics across "
        f"{', '.join(top_chart.relevant_tables) if top_chart else 'database tables'}.\n\n"
        f"Chart Description: {top_chart.description if top_chart else 'N/A'}"
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return LLMResponse(
        answer=mock_text,
        provider="mock",
        model="offline-fallback",
        latency_ms=round(elapsed_ms, 2),
        input_tokens=opt_context.estimated_tokens,
        output_tokens=len(mock_text.split()),
    )


def generate_answer(opt_context: OptimizedContext) -> LLMResponse:
    """
    Main LLM entry point.
    Tries Groq (multi-model) -> Fallback Gemini (multi-model) -> Fallback Mock.
    """
    if get_api_key("GROQ_API_KEY"):
        try:
            return call_groq(opt_context)
        except Exception as e:
            print(f"[LLM Client] Groq candidates failed: {e}. Trying Gemini fallback...")

    if get_api_key("GEMINI_API_KEY"):
        try:
            return call_gemini(opt_context)
        except Exception as e:
            print(f"[LLM Client] Gemini candidates failed: {e}. Falling back to Mock...")

    return call_mock(opt_context)


if __name__ == "__main__":
    from app.core.fusion import run_fusion_pipeline
    from app.core.token_optimizer import pack_context_within_budget

    q = "What is the return rate for each product category?"
    cands = run_fusion_pipeline(q, top_k=3)
    opt = pack_context_within_budget(q, cands, token_budget=400)

    res = generate_answer(opt)
    print(f"\nProvider: {res.provider} | Model: {res.model} | Latency: {res.latency_ms}ms")
    print(f"Input Tokens: {res.input_tokens} | Output Tokens: {res.output_tokens}")
    print("\nAnswer:\n", res.answer)
