"""Loads the black-box segmentation model for Standard CV Mode.

Default checkpoint is a SegFormer fine-tuned on Cityscapes (Hugging Face,
NVIDIA org) -- pure PyTorch, no custom CUDA ops, and its output classes
already align with Cityscapes' 19 trainIds. Swap `checkpoint` in
configs/standard_cv.yaml for any other Hugging Face semantic-segmentation
checkpoint; the pipeline only ever sees predict()'s output array.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

DEFAULT_CHECKPOINT = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"


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
