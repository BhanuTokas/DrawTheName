"""VLM backbone loading and region embedding extraction (spec section 3.1/3.2 step 3).

Primary backbone: CLIP ViT-L/14 (see [[project_eurosat_b2t_sanity]] for the
EuroSAT sanity-check justification). RemoteCLIP is not recommended.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from drawthename.regions import Region


class ClipLikeBackbone(Protocol):
    def encode_image(self, crops: np.ndarray) -> np.ndarray: ...
    def encode_text(self, texts: list[str]) -> np.ndarray: ...


def load_backbone(name: str, device: str = "cuda") -> ClipLikeBackbone:
    """Loads a CLIP-family backbone by name, e.g. 'ViT-L-14', 'georsclip-ViT-B-32'."""
    raise NotImplementedError


def embed_regions(regions: list[Region], backbone: ClipLikeBackbone) -> np.ndarray:
    """Encodes each region crop through the backbone's image encoder."""
    raise NotImplementedError
