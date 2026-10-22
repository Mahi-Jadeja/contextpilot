"""
Context Fusion module for ContextPilot.
Combines CLIP text and vision similarity scores via weighted re-ranking.
"""
from dataclasses import dataclass
from typing import List, Tuple

from app.core.config import FUSION_WEIGHT_TEXT, FUSION_WEIGHT_VISION
from app.core.data_loader import ChartRecord
from app.core.retrieval import RetrievalCandidate, get_retriever


@dataclass
class FusedCandidate:
    record: ChartRecord
    fused_score: float
    text_score: float
    vision_score: float
    rank: int

    def build_summary(self) -> str:
        """Generates compact, structured context summary of chart for LLM prompt."""
        tables_str = ", ".join(self.record.relevant_tables)
        return (
            f"[Chart ID: {self.record.chart_id}] ({self.record.chart_type.upper()} chart)\n"
            f"Description: {self.record.description}\n"
            f"Relevant DB Tables: {tables_str}\n"
            f"Query Intent: {self.record.query_intent.capitalize()}"
        )


def fuse_and_rerank(
    candidates: List[RetrievalCandidate],
    weight_text: float = FUSION_WEIGHT_TEXT,
    weight_vision: float = FUSION_WEIGHT_VISION,
) -> List[FusedCandidate]:
    """
    Applies weighted linear fusion across text and vision cosine similarity scores
    and returns sorted list of FusedCandidate objects.
    """
    # Normalize weights so they sum to 1.0
    total_w = weight_text + weight_vision
    w_t = weight_text / total_w if total_w > 0 else 0.55
    w_v = weight_vision / total_w if total_w > 0 else 0.45

    fused_list = []
    for cand in candidates:
        score = (w_t * cand.text_score) + (w_v * cand.vision_score)
        fused_list.append((score, cand))

    # Sort descending by fused score
    fused_list.sort(key=lambda x: x[0], reverse=True)

    results = []
    for rank, (fused_score, cand) in enumerate(fused_list, start=1):
        results.append(
            FusedCandidate(
                record=cand.record,
                fused_score=float(fused_score),
                text_score=cand.text_score,
                vision_score=cand.vision_score,
                rank=rank,
            )
        )

    return results


def run_fusion_pipeline(
    query: str,
    top_k: int = 5,
    w_text: float = FUSION_WEIGHT_TEXT,
    w_vision: float = FUSION_WEIGHT_VISION,
) -> List[FusedCandidate]:
    """
    End-to-end fusion pipeline: Retrieve -> Fuse & Re-rank.
    """
    retriever = get_retriever()
    candidates = retriever.retrieve_candidates(query=query, top_k=top_k)
    return fuse_and_rerank(candidates, weight_text=w_text, weight_vision=w_vision)


if __name__ == "__main__":
    results = run_fusion_pipeline("What is the return rate for each product category?", top_k=3)
    print(f"Fused {len(results)} results:")
    for res in results:
        print(f"\nRank {res.rank}: {res.record.filename}")
        print(f"  Fused Score: {res.fused_score:.4f} (Text: {res.text_score:.4f}, Vision: {res.vision_score:.4f})")
        print("  Summary:")
        print("  " + res.build_summary().replace("\n", "\n  "))
