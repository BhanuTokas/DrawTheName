"""Per-class clustering of error region embeddings (spec section 3.1/3.2 step 4).

Clustering errors before naming them is the core methodological fix over
vanilla B2T, which assumes unimodal error structure.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def select_k_by_silhouette(
    embeddings: np.ndarray, k_min: int = 2, k_max: int = 8
) -> int:
    """Sweeps k in [k_min, k_max] and returns the k with the best silhouette score.
    Falls back to k=1 (no meaningful split) when there aren't enough samples."""
    n_samples = embeddings.shape[0]
    if n_samples <= k_min:
        return 1

    best_k, best_score = 1, -1.0
    for k in range(k_min, min(k_max, n_samples - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def cluster_embeddings(embeddings: np.ndarray, k: int) -> np.ndarray:
    """K-means cluster assignments for the given embeddings."""
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(embeddings)
