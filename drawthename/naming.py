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


def bias_direction(
    error_embeddings: np.ndarray, correct_embeddings: np.ndarray
) -> np.ndarray:
    """mean(error_embeddings) - mean(correct_embeddings)."""
    return error_embeddings.mean(axis=0) - correct_embeddings.mean(axis=0)


def bootstrap_sign_stability(
    error_embeddings: np.ndarray,
    correct_embeddings: np.ndarray,
    n_resamples: int = 500,
    seed: int = 0,
) -> float:
    """Fraction of bootstrap resamples whose bias direction has positive cosine
    similarity with the full-sample direction (i.e. points the same way)."""
    rng = np.random.default_rng(seed)
    full_direction = bias_direction(error_embeddings, correct_embeddings)
    full_norm = full_direction / (np.linalg.norm(full_direction) + 1e-12)

    n_error, n_correct = len(error_embeddings), len(correct_embeddings)
    agreements = 0
    for _ in range(n_resamples):
        error_sample = error_embeddings[rng.integers(0, n_error, n_error)]
        correct_sample = correct_embeddings[rng.integers(0, n_correct, n_correct)]
        direction = bias_direction(error_sample, correct_sample)
        cos_sim = np.dot(direction, full_norm) / (np.linalg.norm(direction) + 1e-12)
        if cos_sim > 0:
            agreements += 1
    return agreements / n_resamples


def retrieve_concepts(
    bias_vector: np.ndarray,
    concept_texts: list[str],
    concept_embeddings: np.ndarray,
    top_k: int = 5,
) -> list[str]:
    """Top-k concept descriptors by cosine similarity to the bias vector."""
    bias_norm = bias_vector / (np.linalg.norm(bias_vector) + 1e-12)
    concept_norms = concept_embeddings / (
        np.linalg.norm(concept_embeddings, axis=1, keepdims=True) + 1e-12
    )
    similarities = concept_norms @ bias_norm
    top_indices = np.argsort(-similarities)[:top_k]
    return [concept_texts[i] for i in top_indices]
