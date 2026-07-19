"""Intra-tile vs. inter-tile comparison, FTW Mode only (spec section 3.2 step 5).

Intra-tile comparisons control for tile-level confounds (geography, acquisition
date, sensor conditions); divergence between intra- and inter-tile bias
directions flags likely confounds rather than genuine model bias.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from drawthename.naming import bias_direction
from drawthename.regions import Region


def intra_tile_direction(
    error_regions: list[Region],
    error_embeddings: np.ndarray,
    correct_regions: list[Region],
    correct_embeddings: np.ndarray,
) -> np.ndarray | None:
    """Averages, over tiles containing both error and correct regions of this
    class/cluster, that tile's (error mean - correct mean) -- differencing
    within the same tile cancels out any tile-level confound shared by both
    sides. Returns None if no tile has both error and correct regions."""
    error_by_tile: dict[str | None, list[np.ndarray]] = defaultdict(list)
    for region, embedding in zip(error_regions, error_embeddings, strict=True):
        error_by_tile[region.tile_id].append(embedding)
    correct_by_tile: dict[str | None, list[np.ndarray]] = defaultdict(list)
    for region, embedding in zip(correct_regions, correct_embeddings, strict=True):
        correct_by_tile[region.tile_id].append(embedding)

    shared_tiles = set(error_by_tile) & set(correct_by_tile)
    if not shared_tiles:
        return None

    per_tile_directions = [
        np.mean(error_by_tile[tile_id], axis=0)
        - np.mean(correct_by_tile[tile_id], axis=0)
        for tile_id in shared_tiles
    ]
    return np.mean(per_tile_directions, axis=0)


def inter_tile_direction(
    error_regions: list[Region],
    error_embeddings: np.ndarray,
    correct_regions: list[Region],
    correct_embeddings: np.ndarray,
) -> np.ndarray:
    """Plain mean(error) - mean(correct), pooling across all tiles regardless
    of which tile each region came from -- the naive baseline that
    intra_tile_direction is compared against to detect tile-level confounds.
    (error_regions/correct_regions are unused here -- only their embeddings
    matter -- but kept in the signature to mirror intra_tile_direction so
    both functions can be called interchangeably.)"""
    del error_regions, correct_regions
    return bias_direction(error_embeddings, correct_embeddings)


def flag_confound(
    intra_direction: np.ndarray, inter_direction: np.ndarray, threshold: float = 0.5
) -> bool:
    """True if cosine similarity between intra- and inter-tile directions is below threshold."""
    intra_norm = np.linalg.norm(intra_direction) + 1e-12
    inter_norm = np.linalg.norm(inter_direction) + 1e-12
    cos_sim = np.dot(intra_direction, inter_direction) / (intra_norm * inter_norm)
    return bool(cos_sim < threshold)
