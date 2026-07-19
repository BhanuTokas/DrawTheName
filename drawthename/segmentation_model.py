"""Loads the black-box segmentation model under investigation.

Standard CV Mode's default checkpoint is a SegFormer fine-tuned on
Cityscapes (Hugging Face, NVIDIA org) -- pure PyTorch, no custom CUDA ops,
and its output classes already align with Cityscapes' 19 trainIds. Swap
`checkpoint` in configs/standard_cv.yaml for any other Hugging Face
semantic-segmentation checkpoint; the pipeline only ever sees predict()'s
output array.

FTW Mode uses PRUEModel instead, reusing ftw-baselines' own checkpoint
loader rather than reimplementing the UNet+efficientnet-b3 construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ftw_tools.inference.models import load_model_from_checkpoint
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

DEFAULT_CHECKPOINT = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

# Sentinel-2 reflectance -> ~[0, 1] scaling used by ftw-baselines' own
# inference.default_preprocess; PRUE was trained against inputs normalized
# this way, so any other scaling would silently degrade predictions.
FTW_REFLECTANCE_SCALE = 3000.0


class SegmentationModel:
    def __init__(
        self, checkpoint: str = DEFAULT_CHECKPOINT, device: str = "cuda"
    ) -> None:
        self.device = device
        self.processor = SegformerImageProcessor.from_pretrained(checkpoint)
        self.model = (
            SegformerForSemanticSegmentation.from_pretrained(checkpoint)
            .to(device)
            .eval()
        )

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """image: (H, W, 3) uint8 RGB. Returns (H, W) int64 predicted class ids,
        upsampled back to the input resolution."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        h, w = image.shape[:2]
        prediction = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(h, w)]
        )[0]
        return prediction.cpu().numpy().astype(np.int64)


class PRUEModel:
    """Wraps a PRUE field-boundary checkpoint loaded via ftw-baselines'
    load_model_from_checkpoint (handles the UNet+efficientnet-b3
    reconstruction and state-dict key stripping)."""

    def __init__(self, checkpoint: str | Path, device: str = "cuda") -> None:
        self.device = device
        self.model, self.model_type = load_model_from_checkpoint(str(checkpoint))
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """image: (H, W, 3) float32 Sentinel-2 reflectance, RGB (window A).
        Returns (H, W) int64 predicted class ids (0=background,
        1=field-interior, 2=field-boundary)."""
        tensor = (
            torch.from_numpy(image).permute(2, 0, 1).float() / FTW_REFLECTANCE_SCALE
        )
        tensor = tensor.unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)
