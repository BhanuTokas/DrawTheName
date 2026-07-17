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
    residual_ratio: float = 1.0  # ||deconfounded|| / ||bias_vector||; near 0 means concepts are ~arbitrary noise, not signal
    intra_inter_flag: str | None = None  # FTW mode only


@dataclass
class GlobalErrorMode:
    """The class-agnostic direction shared across (nearly) every class's bias
    vector -- e.g. "errors tend to be small/blurry/oddly-cropped" -- named the
    same way as any other direction. Reported on its own, and also projected
    out of each class's bias_vector before that direction's retrieve_concepts
    call, so a class's concepts reflect what's specific to it rather than
    being drowned out by this shared confound."""

    bias_vector: np.ndarray
    concepts: list[str]
    stability: float


def bias_direction(
    error_embeddings: np.ndarray, correct_embeddings: np.ndarray
) -> np.ndarray:
    """mean(error_embeddings) - mean(correct_embeddings)."""
    return error_embeddings.mean(axis=0) - correct_embeddings.mean(axis=0)


def bootstrap_sign_stability(
    error_embeddings: np.ndarray,
    correct_embeddings: np.ndarray,
    n_resamples: int = 500,
    cosine_threshold: float = 0.9,
    seed: int = 0,
) -> float:
    """Fraction of bootstrap resamples whose bias direction has cosine
    similarity above cosine_threshold with the full-sample direction (i.e.
    closely realigns with it, not just points loosely the same way -- a plain
    >0 check saturates near 1.0 whenever clusters have more than a few dozen
    points, since bootstrap resampling of a mean converges tightly by then).

    Resamples are drawn as multinomial per-point counts rather than looping
    n_resamples times over an explicit with-replacement gather: "draw n items
    with replacement from n items" and "draw a Multinomial(n, uniform) count
    per item" are the same distribution, but expressing it as counts lets all
    n_resamples resampled means come from one matrix multiply against the
    embeddings instead of n_resamples sequential Python-loop gathers -- the
    same computation, dramatically faster once pools reach the thousands."""
    rng = np.random.default_rng(seed)
    full_direction = bias_direction(error_embeddings, correct_embeddings)
    full_norm = full_direction / (np.linalg.norm(full_direction) + 1e-12)

    n_error, n_correct = len(error_embeddings), len(correct_embeddings)
    error_counts = rng.multinomial(
        n_error, np.full(n_error, 1 / n_error), size=n_resamples
    )
    correct_counts = rng.multinomial(
        n_correct, np.full(n_correct, 1 / n_correct), size=n_resamples
    )

    error_means = (error_counts @ error_embeddings) / n_error
    correct_means = (correct_counts @ correct_embeddings) / n_correct
    directions = error_means - correct_means  # (n_resamples, dim)

    norms = np.linalg.norm(directions, axis=1) + 1e-12
    cos_sims = (directions @ full_norm) / norms
    return float(np.mean(cos_sims > cosine_threshold))


def deconfound(bias_vector: np.ndarray, global_direction: np.ndarray) -> np.ndarray:
    """Projects global_direction out of bias_vector (standard vector
    rejection), isolating whatever's specific to this direction from the
    shared, class-agnostic component."""
    global_unit = global_direction / (np.linalg.norm(global_direction) + 1e-12)
    return bias_vector - np.dot(bias_vector, global_unit) * global_unit


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
