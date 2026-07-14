"""Per-class clustering of error region embeddings (spec section 3.1/3.2 step 4).

Clustering errors before naming them is the core methodological fix over
vanilla B2T, which assumes unimodal error structure.
"""

from __future__ import annotations

import numpy as np


def select_k_by_silhouette(embeddings: np.ndarray, k_min: int = 2, k_max: int = 8) -> int:
    """Sweeps k in [k_min, k_max] and returns the k with the best silhouette score."""
    raise NotImplementedError


def cluster_embeddings(embeddings: np.ndarray, k: int) -> np.ndarray:
    """K-means cluster assignments for the given embeddings."""
    raise NotImplementedError
