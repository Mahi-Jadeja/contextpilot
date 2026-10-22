"""
Retrieval module for ContextPilot.
Performs FAISS similarity lookups for input queries across text and vision embeddings.
"""
from dataclasses import dataclass
from typing import List

import numpy as np

from app.core.config import DEFAULT_TOP_K
from app.core.data_loader import ChartRecord, load_chart_records
from app.core.embeddings import encode_text
from app.core.precompute import load_precomputed_data, run_precomputation


@dataclass
class RetrievalCandidate:
    record: ChartRecord
    text_score: float      # Cosine similarity (query text vs chart description)
    vision_score: float    # Cosine similarity (query text vs chart image)
    index: int             # Position in precomputed database array


class Retriever:
    """Manages FAISS index search and candidate retrieval for incoming queries."""

    def __init__(self):
        try:
            self.data = load_precomputed_data()
        except FileNotFoundError:
            print("[Retriever] Precomputed cache not found. Auto-generating...")
            self.data = run_precomputation(force_rebuild=False)

        self.records = load_chart_records()
        self.filename_to_idx = {rec.filename: i for i, rec in enumerate(self.records)}

        assert len(self.records) == len(self.data["filenames"]), "Record count mismatch with precomputed cache"

    def retrieve_candidates(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[RetrievalCandidate]:
        """
        Encodes query string into CLIP space, searches FAISS text & vision indices,
        and returns top_k RetrievalCandidate objects.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty")

        # Encode query text -> 512-d L2 normalized vector
        q_vec = encode_text(query.strip())  # shape (1, 512)

        # FAISS search against text index
        text_scores, text_indices = self.data["text_index"].search(q_vec, top_k)

        # FAISS search against vision index
        vision_scores, vision_indices = self.data["vision_index"].search(q_vec, top_k)

        t_scores_flat = text_scores[0]
        t_indices_flat = text_indices[0]

        v_scores_dict = {idx: score for score, idx in zip(vision_scores[0], vision_indices[0])}

        candidates = []
        for rank, idx in enumerate(t_indices_flat):
            idx = int(idx)
            rec = self.records[idx]
            t_score = float(t_scores_flat[rank])
            # If candidate index also appeared in vision search, use that score, else compute dot product
            if idx in v_scores_dict:
                v_score = float(v_scores_dict[idx])
            else:
                v_emb = self.data["vision_embeddings"][idx]
                v_score = float(np.dot(q_vec[0], v_emb))

            candidates.append(
                RetrievalCandidate(
                    record=rec,
                    text_score=t_score,
                    vision_score=v_score,
                    index=idx,
                )
            )

        return candidates


# Global singleton instance for app reuse
_RETRIEVER_INSTANCE = None


def get_retriever() -> Retriever:
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        _RETRIEVER_INSTANCE = Retriever()
    return _RETRIEVER_INSTANCE


if __name__ == "__main__":
    retriever = get_retriever()
    results = retriever.retrieve_candidates("How has total order revenue trended over time?", top_k=3)
    print(f"Retrieved {len(results)} candidates:")
    for c in results:
        print(f"  [{c.record.chart_type}] {c.record.filename} | Text Sim: {c.text_score:.4f} | Vision Sim: {c.vision_score:.4f}")
