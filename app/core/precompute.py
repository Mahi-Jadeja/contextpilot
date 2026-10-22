"""
Precomputation module for ContextPilot.
Computes and caches text/vision CLIP embeddings and builds FAISS vector indices for all charts.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
from pathlib import Path

import faiss
import numpy as np

from app.core.config import (
    CLIP_EMBED_DIM,
    EMBEDDINGS_DIR,
    FAISS_DIR,
)
from app.core.data_loader import load_chart_records
from app.core.embeddings import encode_image, encode_text

TEXT_EMB_PATH = EMBEDDINGS_DIR / "chart_text_embeddings.npy"
VISION_EMB_PATH = EMBEDDINGS_DIR / "chart_vision_embeddings.npy"
FILENAMES_PATH = EMBEDDINGS_DIR / "chart_filenames.json"

TEXT_FAISS_PATH = FAISS_DIR / "text_index.faiss"
VISION_FAISS_PATH = FAISS_DIR / "vision_index.faiss"


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Creates a FAISS Inner Product index for L2-normalized vectors (cosine similarity)."""
    dim = embeddings.shape[1]
    assert dim == CLIP_EMBED_DIM, f"Expected dimension {CLIP_EMBED_DIM}, got {dim}"
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def run_precomputation(force_rebuild: bool = False) -> dict:
    """
    Computes text and image embeddings for all chart records,
    saves numpy arrays and FAISS indices to cache/.
    """
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    files_exist = (
        TEXT_EMB_PATH.is_file()
        and VISION_EMB_PATH.is_file()
        and FILENAMES_PATH.is_file()
        and TEXT_FAISS_PATH.is_file()
        and VISION_FAISS_PATH.is_file()
    )

    if files_exist and not force_rebuild:
        print("[Precompute] Cached embeddings and FAISS indices already exist. Skipping rebuild.")
        return load_precomputed_data()

    print("[Precompute] Loading chart records...")
    records = load_chart_records()
    n_charts = len(records)
    print(f"[Precompute] Generating embeddings for {n_charts} chart records...")

    filenames = [rec.filename for rec in records]
    descriptions = [rec.description for rec in records]
    image_paths = [rec.image_path for rec in records]

    # Compute text embeddings
    print("[Precompute] Encoding text descriptions...")
    text_embeddings = encode_text(descriptions)

    # Compute image embeddings (batch_size=8 for low memory and thread safety)
    print("[Precompute] Encoding chart images...")
    vision_list = []
    batch_size = 8
    for i in range(0, n_charts, batch_size):
        batch_paths = image_paths[i : i + batch_size]
        v_batch = encode_image(batch_paths)
        vision_list.append(v_batch)
    vision_embeddings = np.vstack(vision_list)

    assert text_embeddings.shape == (n_charts, CLIP_EMBED_DIM)
    assert vision_embeddings.shape == (n_charts, CLIP_EMBED_DIM)

    # Save raw arrays and filenames JSON
    print("[Precompute] Saving embeddings to cache...")
    np.save(TEXT_EMB_PATH, text_embeddings)
    np.save(VISION_EMB_PATH, vision_embeddings)
    with open(FILENAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(filenames, f, indent=2)

    # Build and save FAISS indices
    print("[Precompute] Building FAISS indices...")
    text_index = build_faiss_index(text_embeddings)
    vision_index = build_faiss_index(vision_embeddings)

    faiss.write_index(text_index, str(TEXT_FAISS_PATH))
    faiss.write_index(vision_index, str(VISION_FAISS_PATH))

    print("[Precompute] Build complete and saved to cache/.")

    return {
        "text_embeddings": text_embeddings,
        "vision_embeddings": vision_embeddings,
        "filenames": filenames,
        "text_index": text_index,
        "vision_index": vision_index,
    }


def load_precomputed_data() -> dict:
    """Loads cached embeddings, filenames map, and FAISS indices into memory."""
    if not (TEXT_EMB_PATH.is_file() and VISION_EMB_PATH.is_file() and FILENAMES_PATH.is_file()):
        raise FileNotFoundError("Cached embeddings not found. Run run_precomputation() first.")

    text_emb = np.load(TEXT_EMB_PATH)
    vision_emb = np.load(VISION_EMB_PATH)
    with open(FILENAMES_PATH, "r", encoding="utf-8") as f:
        filenames = json.load(f)

    text_index = faiss.read_index(str(TEXT_FAISS_PATH))
    vision_index = faiss.read_index(str(VISION_FAISS_PATH))

    return {
        "text_embeddings": text_emb,
        "vision_embeddings": vision_emb,
        "filenames": filenames,
        "text_index": text_index,
        "vision_index": vision_index,
    }


if __name__ == "__main__":
    run_precomputation(force_rebuild=True)
