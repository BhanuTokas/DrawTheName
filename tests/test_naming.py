import numpy as np

from drawthename.naming import (
    bias_direction,
    bootstrap_sign_stability,
    retrieve_concepts,
)


def test_bias_direction_is_mean_difference():
    error_embeddings = np.array([[2.0, 2.0], [4.0, 4.0]])
    correct_embeddings = np.array([[0.0, 0.0], [0.0, 0.0]])
    direction = bias_direction(error_embeddings, correct_embeddings)
    np.testing.assert_array_almost_equal(direction, [3.0, 3.0])


def test_bootstrap_sign_stability_high_for_clear_separation():
    rng = np.random.default_rng(0)
    error_embeddings = rng.normal(loc=[5, 5], scale=0.1, size=(30, 2))
    correct_embeddings = rng.normal(loc=[0, 0], scale=0.1, size=(30, 2))
    stability = bootstrap_sign_stability(
        error_embeddings, correct_embeddings, n_resamples=200
    )
    assert stability > 0.95


def test_bootstrap_sign_stability_low_for_overlapping_noise():
    rng = np.random.default_rng(0)
    error_embeddings = rng.normal(loc=[0, 0], scale=1.0, size=(30, 2))
    correct_embeddings = rng.normal(loc=[0, 0], scale=1.0, size=(30, 2))
    stability = bootstrap_sign_stability(
        error_embeddings, correct_embeddings, n_resamples=200
    )
    assert stability < 0.95


def test_retrieve_concepts_picks_closest_by_cosine_similarity():
    bias_vector = np.array([1.0, 0.0])
    concept_texts = ["aligned", "orthogonal", "opposite"]
    concept_embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    top = retrieve_concepts(bias_vector, concept_texts, concept_embeddings, top_k=1)
    assert top == ["aligned"]
