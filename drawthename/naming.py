"""Bias direction computation, stability validation, and concept naming
(spec section 3.1/3.2 step 5-6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NamedDirection:
    class_id: int
    cluster_id: int
    bias_vector: np.ndarray
    concepts: list[str]
    stability: float
    intra_inter_flag: str | None = None  # FTW mode only


def bias_direction(error_embeddings: np.ndarray, correct_embeddings: np.ndarray) -> np.ndarray:
    """mean(error_embeddings) - mean(correct_embeddings)."""
    raise NotImplementedError


def bootstrap_sign_stability(
    error_embeddings: np.ndarray,
    correct_embeddings: np.ndarray,
    n_resamples: int = 500,
) -> float:
    """Fraction of resamples whose bias direction sign matches the full-sample direction."""
    raise NotImplementedError


def retrieve_concepts(
    bias_vector: np.ndarray,
    concept_texts: list[str],
    concept_embeddings: np.ndarray,
    top_k: int = 5,
) -> list[str]:
    """Top-k concept descriptors by cosine similarity to the bias vector."""
    raise NotImplementedError
