"""
Unified LLM Client for ContextPilot.
Primary: Groq (Llama 3.1 8B Instant)
Fallback: Google Gemini (google-genai / gemini-2.5-flash)
Safety: Mock offline fallback
"""
import os
import time
from dataclasses import dataclass
from typing import Optional

from app.core.config import GEMINI_MODEL, GROQ_MODEL, LLM_MAX_OUTPUT_TOKENS, LLM_TEMPERATURE
from app.core.token_optimizer import OptimizedContext


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
    """Retrieves API key from environment variable or Streamlit secrets."""
    val = os.getenv(key_name)
    if val:
        return val.strip()

    try:
        import streamlit as st
        if hasattr(st, "secrets") and key_name in st.secrets:
            return str(st.secrets[key_name]).strip()
    except Exception:
        pass

    return None


def call_groq(opt_context: OptimizedContext) -> LLMResponse:
    """Calls Groq API using Llama 3.1 8B Instant model."""
    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    from groq import Groq

    client = Groq(api_key=api_key)
    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
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
        model=GROQ_MODEL,
        latency_ms=round(elapsed_ms, 2),
        input_tokens=in_tok,
        output_tokens=out_tok,
    )


def call_gemini(opt_context: OptimizedContext) -> LLMResponse:
    """Fallback: Calls Gemini API using google-genai client."""
    api_key = get_api_key("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    from google import genai

    client = genai.Client(api_key=api_key)
    start_time = time.perf_counter()

    prompt = f"{opt_context.system_prompt}\n\n{opt_context.user_prompt}"
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    answer = getattr(response, "text", "") or ""

    return LLMResponse(
        answer=answer.strip(),
        provider="gemini",
        model=GEMINI_MODEL,
        latency_ms=round(elapsed_ms, 2),
        input_tokens=opt_context.estimated_tokens,
        output_tokens=len(answer.split()),
    )


def call_mock(opt_context: OptimizedContext) -> LLMResponse:
    """Offline safety fallback when no live API keys are provided."""
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
    Tries Groq -> Fallback to Gemini -> Fallback to Mock.
    """
    # 1. Try Groq Primary
    if get_api_key("GROQ_API_KEY"):
        try:
            return call_groq(opt_context)
        except Exception as e:
            print(f"[LLM Client] Groq failed: {e}. Trying Gemini fallback...")

    # 2. Try Gemini Fallback
    if get_api_key("GEMINI_API_KEY"):
        try:
            return call_gemini(opt_context)
        except Exception as e:
            print(f"[LLM Client] Gemini failed: {e}. Falling back to Mock...")

    # 3. Fallback to Mock
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
