"""
Pre-download torchvision + Hugging Face weights into repo-local .model_caches/
(same paths backend_app.py uses via env vars).

Run during deploy build, e.g. Render:
  pip install -r requirements.txt && python scripts/warm_model_caches.py
"""
from __future__ import annotations

import os
import sys

# Match backend_app.py cache layout (must run before importing torch/transformers).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_ROOT = os.path.join(_ROOT, ".model_caches")
_HF_CACHE = os.path.join(_CACHE_ROOT, "huggingface")
_TORCH_CACHE = os.path.join(_CACHE_ROOT, "torch")
for _d in (_CACHE_ROOT, _HF_CACHE, _TORCH_CACHE):
    os.makedirs(_d, exist_ok=True)
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", _HF_CACHE)
os.environ.setdefault("TORCH_HOME", _TORCH_CACHE)

_NUM_CLASSES = 38


def main() -> None:
    print("Warming torchvision ImageNet weights…")
    import torch
    from torchvision import models

    models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    del torch
    print("Warming Hugging Face ViT checkpoints (same IDs as backend_app)…")
    from transformers import ViTFeatureExtractor, ViTForImageClassification

    ViTForImageClassification.from_pretrained(
        "wambugu1738/crop_leaf_diseases_vit",
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=True,
    )
    ViTFeatureExtractor.from_pretrained("wambugu71/crop_leaf_diseases_vit")
    ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=_NUM_CLASSES,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=True,
    )
    ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    print("Cache warm finished. Caches are under:", _CACHE_ROOT)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("warm_model_caches failed:", e, file=sys.stderr)
        sys.exit(1)
