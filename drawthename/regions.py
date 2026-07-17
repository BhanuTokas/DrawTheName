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
    bbox: tuple[int, int, int, int] = (
        0,
        0,
        0,
        0,
    )  # (y0, x0, y1, x1), padded, in source-image coords
    tile_id: str | None = None


def compute_error_mask(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """Per-pixel error mask: 1 where prediction != ground_truth, 0 elsewhere."""
    return (prediction != ground_truth).astype(np.uint8)


def _grid_bboxes(
    y0: int, x0: int, y1: int, x1: int, size: int
) -> list[tuple[int, int, int, int]]:
    """Non-overlapping size x size grid cells covering [y0:y1, x0:x1] (edge
    cells clipped, so they may be smaller than size)."""
    return [
        (ty, tx, min(ty + size, y1), min(tx + size, x1))
        for ty in range(y0, y1, size)
        for tx in range(x0, x1, size)
    ]


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
    subdivision_size: int | None = None,
) -> list[Region]:
    """Connected components per ground-truth class, crops padded bounding boxes,
    labels each region error/correct by its local pixel_error_rate, and discards
    components smaller than min_area_px.

    Components larger than subdivision_size^2 are split into a grid of
    subdivision_size x subdivision_size sub-regions instead of kept as one
    region. Error regions are typically already small (boundary slivers), so
    this mostly affects large correct blobs -- growing n_correct, which
    measurably tightens bootstrap stability for clusters that have a real
    signal to converge toward (confirmed empirically: holding an error
    cluster fixed and only increasing n_correct converges stability toward
    1.0, though it can't rescue a cluster whose error side is itself too
    scattered to have a stable direction)."""
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

            component_mask = labeled == component.label
            y0, x0, y1, x1 = component.bbox

            if subdivision_size and component.area > subdivision_size**2:
                sub_bboxes = _grid_bboxes(y0, x0, y1, x1, subdivision_size)
            else:
                sub_bboxes = [(y0, x0, y1, x1)]

            for sy0, sx0, sy1, sx1 in sub_bboxes:
                sub_mask = component_mask[sy0:sy1, sx0:sx1]
                if int(sub_mask.sum()) < min_area_px:
                    continue

                pad = max(pad_px_min, int(pad_frac * max(sy1 - sy0, sx1 - sx0)))
                y0p, y1p = max(0, sy0 - pad), min(h, sy1 + pad)
                x0p, x1p = max(0, sx0 - pad), min(w, sx1 + pad)

                pixel_error_rate = float(error_mask[sy0:sy1, sx0:sx1][sub_mask].mean())
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
                        bbox=(y0p, x0p, y1p, x1p),
                        tile_id=tile_id,
                    )
                )
                region_id += 1

    return regions
