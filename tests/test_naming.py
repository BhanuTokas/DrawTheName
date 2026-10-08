import numpy as np
import pytest

from drawthename.naming import (
    bias_direction,
    bootstrap_sign_stability,
    deconfound,
    grouped_bias_direction,
    grouped_bootstrap_sign_stability,
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


def test_deconfound_removes_component_along_global_direction():
    global_direction = np.array([1.0, 0.0, 0.0])
    bias_vector = np.array([3.0, 4.0, 0.0])
    residual = deconfound(bias_vector, global_direction)
    assert np.dot(residual, global_direction) == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_array_almost_equal(residual, [0.0, 4.0, 0.0])


def test_deconfound_leaves_orthogonal_vector_unchanged():
    global_direction = np.array([1.0, 0.0])
    bias_vector = np.array([0.0, 5.0])
    residual = deconfound(bias_vector, global_direction)
    np.testing.assert_array_almost_equal(residual, bias_vector)


def test_retrieve_concepts_picks_closest_by_cosine_similarity():
    bias_vector = np.array([1.0, 0.0])
    concept_texts = ["aligned", "orthogonal", "opposite"]
    concept_embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    top = retrieve_concepts(bias_vector, concept_texts, concept_embeddings, top_k=1)
    assert top == ["aligned"]


def test_grouped_bias_direction_weights_groups_equally():
    # group a: direction (1, 0) from 1 vs 1 points; group b: (0, 1) from 9 vs 1
    error_groups = [np.array([[1.0, 0.0]]), np.zeros((9, 2)) + [0.0, 1.0]]
    correct_groups = [np.zeros((1, 2)), np.zeros((1, 2))]
    direction = grouped_bias_direction(error_groups, correct_groups)
    np.testing.assert_allclose(direction, [0.5, 0.5])


def test_grouped_bootstrap_sign_stability_high_for_clear_separation():
    rng = np.random.default_rng(0)
    error_groups = [rng.normal(size=(50, 8)) + 3 for _ in range(2)]
    correct_groups = [rng.normal(size=(50, 8)) for _ in range(2)]
    assert (
        grouped_bootstrap_sign_stability(error_groups, correct_groups, n_resamples=100)
        > 0.95
    )
