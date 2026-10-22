"""
End-to-End Backend Verification Suite for ContextPilot.
Tests all Phase 1 core modules.
"""
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from app.core import config as c
from app.core.data_loader import load_chart_records, get_sample_queries
from app.core.embeddings import encode_text, encode_image
from app.core.precompute import load_precomputed_data
from app.core.retrieval import get_retriever
from app.core.fusion import run_fusion_pipeline
from app.core.token_optimizer import pack_context_within_budget
from app.core.llm_client import generate_answer


def test_01_config():
    print("Testing Config paths...")
    assert c.CHARTS_DIR.is_dir(), f"Missing CHARTS_DIR at {c.CHARTS_DIR}"
    assert c.CHART_METADATA_CSV.is_file(), f"Missing CSV at {c.CHART_METADATA_CSV}"
    assert c.QUERY_TRIPLETS_CSV.is_file(), f"Missing Triplets CSV at {c.QUERY_TRIPLETS_CSV}"
    print("  [PASS] Config paths verified.")


def test_02_data_loader():
    print("Testing Data Loader...")
    records = load_chart_records()
    assert len(records) == 174, f"Expected 174 records, got {len(records)}"
    for r in records:
        assert r.image_path.is_file(), f"Missing PNG file: {r.image_path}"
    samples = get_sample_queries(5)
    assert len(samples) == 5
    print(f"  [PASS] Data Loader verified ({len(records)} records with disk PNGs).")


def test_03_embeddings():
    print("Testing CLIP Embeddings...")
    t_emb = encode_text("Sample analytical query")
    assert t_emb.shape == (1, 512)
    assert np.isclose(np.linalg.norm(t_emb[0]), 1.0, atol=1e-3)

    records = load_chart_records()
    v_emb = encode_image(records[0].image_path)
    assert v_emb.shape == (1, 512)
    assert np.isclose(np.linalg.norm(v_emb[0]), 1.0, atol=1e-3)
    print("  [PASS] CLIP Text and Vision 512-d feature extraction verified.")


def test_04_precompute_cache():
    print("Testing Precomputed Cache & FAISS Indices...")
    data = load_precomputed_data()
    assert data["text_embeddings"].shape == (174, 512)
    assert data["vision_embeddings"].shape == (174, 512)
    assert data["text_index"].ntotal == 174
    assert data["vision_index"].ntotal == 174
    print("  [PASS] Precomputed cache & FAISS indices verified.")


def test_05_retrieval_and_fusion():
    print("Testing Retrieval and Fusion Pipeline...")
    query = "What is the return rate for each product category?"
    fused_results = run_fusion_pipeline(query, top_k=5)
    assert len(fused_results) == 5
    assert fused_results[0].rank == 1
    assert fused_results[0].fused_score >= fused_results[1].fused_score
    summary = fused_results[0].build_summary()
    assert "Chart ID:" in summary
    print(f"  [PASS] Retrieval & Fusion verified. Top match: {fused_results[0].record.filename}")


def test_06_token_optimizer():
    print("Testing Token Budget Optimizer...")
    query = "How has total order revenue trended over time?"
    candidates = run_fusion_pipeline(query, top_k=5)
    
    opt_tight = pack_context_within_budget(query, candidates, token_budget=250)
    opt_loose = pack_context_within_budget(query, candidates, token_budget=800)

    assert opt_tight.estimated_tokens <= 250 or len(opt_tight.selected_candidates) == 1
    assert len(opt_loose.selected_candidates) >= len(opt_tight.selected_candidates)
    print(f"  [PASS] Token Optimizer verified (Tight: {len(opt_tight.selected_candidates)} charts, Loose: {len(opt_loose.selected_candidates)} charts).")


def test_07_llm_client():
    print("Testing LLM Client Exec...")
    query = "Which seller has generated the highest revenue in North?"
    candidates = run_fusion_pipeline(query, top_k=3)
    opt_ctx = pack_context_within_budget(query, candidates, token_budget=400)

    res = generate_answer(opt_ctx)
    assert res.provider in ["groq", "gemini", "mock"]
    assert len(res.answer) > 10
    print(f"  [PASS] LLM Client verified (Provider: '{res.provider}', Latency: {res.latency_ms}ms).")


def run_all_tests():
    print("==================================================")
    print("  CONTEXTPILOT BACKEND END-TO-END TEST SUITE")
    print("==================================================")
    test_01_config()
    test_02_data_loader()
    test_03_embeddings()
    test_04_precompute_cache()
    test_05_retrieval_and_fusion()
    test_06_token_optimizer()
    test_07_llm_client()
    print("==================================================")
    print("  ALL PHASE 1 BACKEND TESTS PASSED SUCCESSFULLY!  ")
    print("==================================================")


if __name__ == "__main__":
    run_all_tests()
