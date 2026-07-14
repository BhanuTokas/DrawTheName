"""VLM backbone loading and region embedding extraction (spec section 3.1/3.2 step 3).

Primary backbone: CLIP ViT-L/14 (see [[project_eurosat_b2t_sanity]] for the
EuroSAT sanity-check justification). RemoteCLIP is not recommended.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import open_clip
import torch
from PIL import Image

from drawthename.regions import Region

# name in configs -> (open_clip architecture, pretrained tag).
# OpenAI's original CLIP checkpoints were trained with QuickGELU activation;
# the "-quickgelu" arch variant is required to reproduce them exactly
# (the plain variant silently uses the wrong activation function).
_BACKBONE_CONFIGS = {
    "ViT-L-14": ("ViT-L-14-quickgelu", "openai"),
    "ViT-B-32": ("ViT-B-32-quickgelu", "openai"),
    "ViT-B-16": ("ViT-B-16-quickgelu", "openai"),
}


class ClipLikeBackbone(Protocol):
    def encode_image(self, crops: list[np.ndarray]) -> np.ndarray: ...
    def encode_text(self, texts: list[str]) -> np.ndarray: ...


class OpenClipBackbone:
    def __init__(self, model, preprocess, tokenizer, device: str) -> None:
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def encode_image(self, crops: list[np.ndarray]) -> np.ndarray:
        batch = torch.stack(
            [self.preprocess(Image.fromarray(crop)) for crop in crops]
        ).to(self.device)
        features = self.model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(texts).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()


def load_backbone(name: str, device: str = "cuda") -> ClipLikeBackbone:
    """Loads a CLIP-family backbone by name, e.g. 'ViT-L-14'."""
    if name not in _BACKBONE_CONFIGS:
        raise ValueError(
            f"Unknown backbone '{name}'; available: {sorted(_BACKBONE_CONFIGS)}"
        )
    arch, pretrained = _BACKBONE_CONFIGS[name]
    model, _, preprocess = open_clip.create_model_and_transforms(
        arch, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(arch)
    model.eval()
    return OpenClipBackbone(model, preprocess, tokenizer, device)


def embed_regions(
    regions: list[Region], backbone: ClipLikeBackbone, batch_size: int = 64
) -> np.ndarray:
    """Encodes each region crop through the backbone's image encoder, batched."""
    crops = [region.crop for region in regions]
    if not crops:
        return np.zeros((0, 0), dtype=np.float32)
    batches = [
        backbone.encode_image(crops[i : i + batch_size])
        for i in range(0, len(crops), batch_size)
    ]
    return np.concatenate(batches, axis=0)
