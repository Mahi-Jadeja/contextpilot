"""
CLIP Embedding Module for ContextPilot.
Provides dual text and vision embedding generation mapped into a shared 512-d space.
"""
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.core.config import CLIP_EMBED_DIM, CLIP_MODEL_ID

# Module-level cache for singleton model loading
_MODEL = None
_PROCESSOR = None
_DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def get_clip_model_and_processor():
    """Lazy loads CLIP model and processor into global memory cache."""
    global _MODEL, _PROCESSOR
    if _MODEL is None or _PROCESSOR is None:
        _MODEL = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(_DEVICE)
        _MODEL.eval()
        _PROCESSOR = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    return _MODEL, _PROCESSOR, _DEVICE


def _extract_tensor(output, attr_name: str) -> torch.Tensor:
    """Safely extracts tensor from transformers model output."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, attr_name):
        return getattr(output, attr_name)
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :]
    return output[0]


def encode_text(text: Union[str, List[str]]) -> np.ndarray:
    """
    Encodes single text prompt or list of prompts into L2-normalized 512-d embeddings.
    Returns np.ndarray of shape (N, 512) and dtype float32.
    """
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    model, processor, device = get_clip_model_and_processor()

    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
        text_features = _extract_tensor(out, "text_embeds")
        # L2 Normalize vectors
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    arr = text_features.cpu().numpy().astype(np.float32)
    return arr


def encode_image(image_input: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]]) -> np.ndarray:
    """
    Encodes image path(s) or PIL Image(s) into L2-normalized 512-d embeddings.
    Returns np.ndarray of shape (N, 512) and dtype float32.
    """
    if not isinstance(image_input, list):
        inputs_list = [image_input]
    else:
        inputs_list = image_input

    pil_images = []
    for item in inputs_list:
        if isinstance(item, (str, Path)):
            img_path = Path(item)
            if not img_path.is_file():
                raise FileNotFoundError(f"Chart image not found for encoding: {img_path}")
            img = Image.open(img_path).convert("RGB")
        elif isinstance(item, Image.Image):
            img = item.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(item)}")
        pil_images.append(img)

    model, processor, device = get_clip_model_and_processor()

    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.get_image_features(**inputs)
        image_features = _extract_tensor(out, "image_embeds")
        # L2 Normalize vectors
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    arr = image_features.cpu().numpy().astype(np.float32)
    return arr


if __name__ == "__main__":
    t_emb = encode_text("What is the return rate for each category?")
    print("Text embedding shape:", t_emb.shape, "dtype:", t_emb.dtype)
    print("Text vector L2 norm:", np.linalg.norm(t_emb[0]))
