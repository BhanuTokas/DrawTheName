"""Error-mask computation and connected-component region extraction.

Shared by Standard CV Mode and FTW Mode (spec sections 3.1 step 1-2, 3.2 step 1-2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

RegionLabel = Literal["error", "correct"]


@dataclass
class Region:
    image_id: str
    region_id: int
    crop: np.ndarray
    label: RegionLabel
    class_id: int
    pixel_error_rate: float
    tile_id: str | None = None


def compute_error_mask(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """Per-pixel error mask: 1 where prediction != ground_truth, 0 elsewhere."""
    raise NotImplementedError


def extract_regions(
    image: np.ndarray,
    error_mask: np.ndarray,
    class_id: int,
    image_id: str,
    min_area_px: int = 64,
    pad_px_min: int = 16,
    pad_frac: float = 0.1,
    error_rate_threshold: float = 0.5,
    tile_id: str | None = None,
) -> list[Region]:
    """Connected-components on error_mask, crops padded bounding boxes, labels
    each region error/correct by pixel_error_rate_threshold, and discards
    components smaller than min_area_px."""
    raise NotImplementedError
