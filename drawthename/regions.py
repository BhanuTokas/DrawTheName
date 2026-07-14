"""Error-mask computation and connected-component region extraction.

Shared by Standard CV Mode and FTW Mode (spec sections 3.1 step 1-2, 3.2 step 1-2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from skimage.measure import label as connected_components
from skimage.measure import regionprops

RegionLabel = Literal["error", "correct"]

IGNORE_CLASS = 255


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
    return (prediction != ground_truth).astype(np.uint8)


def extract_regions(
    image: np.ndarray,
    error_mask: np.ndarray,
    ground_truth: np.ndarray,
    image_id: str,
    min_area_px: int = 64,
    pad_px_min: int = 16,
    pad_frac: float = 0.1,
    error_rate_threshold: float = 0.5,
    ignore_class: int = IGNORE_CLASS,
    tile_id: str | None = None,
) -> list[Region]:
    """Connected components per ground-truth class, crops padded bounding boxes,
    labels each region error/correct by its local pixel_error_rate, and discards
    components smaller than min_area_px."""
    h, w = ground_truth.shape
    regions: list[Region] = []
    region_id = 0

    for class_id in np.unique(ground_truth):
        if class_id == ignore_class:
            continue
        class_mask = ground_truth == class_id
        labeled = connected_components(class_mask, connectivity=2)

        for component in regionprops(labeled):
            if component.area < min_area_px:
                continue

            y0, x0, y1, x1 = component.bbox
            pad = max(pad_px_min, int(pad_frac * max(y1 - y0, x1 - x0)))
            y0p, y1p = max(0, y0 - pad), min(h, y1 + pad)
            x0p, x1p = max(0, x0 - pad), min(w, x1 + pad)

            component_mask = labeled == component.label
            pixel_error_rate = float(error_mask[component_mask].mean())
            region_label: RegionLabel = (
                "error" if pixel_error_rate > error_rate_threshold else "correct"
            )

            regions.append(
                Region(
                    image_id=image_id,
                    region_id=region_id,
                    crop=image[y0p:y1p, x0p:x1p],
                    label=region_label,
                    class_id=int(class_id),
                    pixel_error_rate=pixel_error_rate,
                    tile_id=tile_id,
                )
            )
            region_id += 1

    return regions
