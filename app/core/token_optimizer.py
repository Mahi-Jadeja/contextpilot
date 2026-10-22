"""
Token Optimizer module for ContextPilot.
Estimates token counts via tiktoken and selects chart context summaries to fit strict token budgets.
"""
from dataclasses import dataclass, field
from typing import List

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
        summary_text = f"--- Chart #{cand.rank} ---\n{cand.build_summary()}\n\n"
        candidate_combined_prompt = (
            f"{system_prompt}\n\n{base_user_prompt}{current_summaries_text}{summary_text}\n"
            f"Ground your answer based on the context above."
        )

        tokens = count_tokens(candidate_combined_prompt)

        if tokens <= token_budget or len(selected) == 0:
            # Include candidate (always include at least top-1 even if tight)
            selected.append(cand)
            current_summaries_text += summary_text
        else:
            dropped.append(cand)

    final_user_prompt = (
        f"{base_user_prompt}{current_summaries_text}\n"
        f"Provide a clear, grounded answer to the query based on the chart context above."
    )

    full_prompt_text = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{final_user_prompt}"
    total_estimated_tokens = count_tokens(full_prompt_text)

    return OptimizedContext(
        query=query,
        selected_candidates=selected,
        dropped_candidates=dropped,
        system_prompt=system_prompt,
        user_prompt=final_user_prompt,
        full_prompt_text=full_prompt_text,
        estimated_tokens=total_estimated_tokens,
        token_budget=token_budget,
    )


if __name__ == "__main__":
    from app.core.fusion import run_fusion_pipeline

    cands = run_fusion_pipeline("What is the return rate for each product category?", top_k=5)
    opt = pack_context_within_budget("What is the return rate for each product category?", cands, token_budget=350)

    print(f"Token Budget: {opt.token_budget}")
    print(f"Estimated Tokens: {opt.estimated_tokens} ({opt.utilization_pct}% utilization)")
    print(f"Selected Chart Count: {len(opt.selected_candidates)}")
    print(f"Dropped Chart Count: {len(opt.dropped_candidates)}")
