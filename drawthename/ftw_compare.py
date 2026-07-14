"""Intra-tile vs. inter-tile comparison, FTW Mode only (spec section 3.2 step 5).

Intra-tile comparisons control for tile-level confounds (geography, acquisition
date, sensor conditions); divergence between intra- and inter-tile bias
directions flags likely confounds rather than genuine model bias.
"""

from __future__ import annotations

import numpy as np

from drawthename.regions import Region


def intra_tile_direction(regions: list[Region]) -> np.ndarray:
    """Bias direction from error vs. correct sub-regions within the same tile."""
    raise NotImplementedError


def inter_tile_direction(regions: list[Region]) -> np.ndarray:
    """Bias direction from error vs. correct sub-regions across different tiles."""
    raise NotImplementedError


def flag_confound(intra_direction: np.ndarray, inter_direction: np.ndarray, threshold: float = 0.5) -> bool:
    """True if cosine similarity between intra- and inter-tile directions is below threshold."""
    raise NotImplementedError
