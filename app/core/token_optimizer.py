"""
Token Optimizer module for ContextPilot.
Estimates token counts via tiktoken and selects chart context summaries to fit strict token budgets.
"""
from dataclasses import dataclass
from typing import List, Tuple

import tiktoken

from app.core.config import DEFAULT_TOKEN_BUDGET, TIKTOKEN_ENCODING
from app.core.fusion import FusedCandidate


@dataclass
class OptimizedContext:
    query: str
    selected_candidates: List[FusedCandidate]
    dropped_candidates: List[FusedCandidate]
    system_prompt: str
    user_prompt: str
    full_prompt_text: str
    estimated_tokens: int
    token_budget: int

    @property
    def utilization_pct(self) -> float:
        if self.token_budget <= 0:
            return 0.0
        return min(100.0, round((self.estimated_tokens / self.token_budget) * 100, 2))


def count_tokens(text: str, encoding_name: str = TIKTOKEN_ENCODING) -> int:
    """Calculates exact token count for a text string using tiktoken."""
    if not text:
        return 0
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def build_system_prompt() -> str:
    return (
        "You are ContextPilot, an expert AI context assistant for enterprise analytics.\n"
        "Your task is to answer the user's business query using ONLY the retrieved chart context summaries provided below.\n"
        "Guidelines:\n"
        "1. Reference specific chart IDs and chart types in your response.\n"
        "2. Keep your answer factual, precise, and directly grounded in the provided chart contexts.\n"
        "3. If context is insufficient, state what additional data table would be required."
    )


def _format_full_prompt(system_prompt: str, base_user_prompt: str, summaries_text: str) -> Tuple[str, str]:
    """Helper to format user_prompt and full_prompt_text identically."""
    user_prompt = (
        f"{base_user_prompt}{summaries_text}\n"
        f"Provide a clear, grounded answer to the query based on the chart context above."
    )
    full_prompt_text = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
    return user_prompt, full_prompt_text


def pack_context_within_budget(
    query: str,
    candidates: List[FusedCandidate],
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> OptimizedContext:
    """
    Greedily selects fused chart context summaries in rank order until token budget is reached.
    """
    system_prompt = build_system_prompt()
    base_user_prompt = f"User Query: {query}\n\nRetrieved Visual Context Summaries:\n"

    selected: List[FusedCandidate] = []
    dropped: List[FusedCandidate] = []

    current_summaries_text = ""

    for cand in candidates:
        candidate_summary = f"--- Chart #{cand.rank} ---\n{cand.build_summary()}\n\n"
        test_summaries = current_summaries_text + candidate_summary

        _, test_full_prompt = _format_full_prompt(system_prompt, base_user_prompt, test_summaries)
        tokens = count_tokens(test_full_prompt)

        if tokens <= token_budget or len(selected) == 0:
            selected.append(cand)
            current_summaries_text = test_summaries
        else:
            dropped.append(cand)

    final_user_prompt, final_full_prompt = _format_full_prompt(
        system_prompt, base_user_prompt, current_summaries_text
    )
    total_estimated_tokens = count_tokens(final_full_prompt)

    return OptimizedContext(
        query=query,
        selected_candidates=selected,
        dropped_candidates=dropped,
        system_prompt=system_prompt,
        user_prompt=final_user_prompt,
        full_prompt_text=final_full_prompt,
        estimated_tokens=total_estimated_tokens,
        token_budget=token_budget,
    )
