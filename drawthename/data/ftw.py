"""FTW (Fields of the World) dataloader for FTW Mode (Phase 2).

Adapts PRUE's existing tile loader (https://github.com/fieldsoftheworld/ftw-prue)
and adds NIR-band removal for VLM-facing crops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from torch.utils.data import Dataset

TileClassification = Literal["all-correct", "all-error", "mixed"]


@dataclass
class FTWTile:
    tile_id: str
    image_rgbnir: np.ndarray  # (4, H, W): B4, B3, B2, NIR
    ground_truth: np.ndarray
    geography: str
    acquisition_date: str


class FTWDataset(Dataset):
    """Loads FTW Sentinel-2 tiles and boundary masks for a given split."""

    def __init__(self, root: Path, split: str = "val") -> None:
        self.root = root
        self.split = split
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> FTWTile:
        raise NotImplementedError


def to_rgb(image_rgbnir: np.ndarray) -> np.ndarray:
    """Drops the NIR channel; VLM backbones are RGB-only (see spec section 7)."""
    return image_rgbnir[:3]


def classify_tile(error_mask: np.ndarray, error_threshold: float = 0.5) -> TileClassification:
    """Labels a tile as all-correct, all-error, or mixed from its per-pixel error mask."""
    raise NotImplementedError
