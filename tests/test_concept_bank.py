import numpy as np

from drawthename.concept_bank import center_embeddings


def test_center_embeddings_removes_the_mean():
    embeddings = np.array([[1.0, 1.0], [3.0, 1.0], [2.0, 4.0]])
    centered = center_embeddings(embeddings)
    np.testing.assert_array_almost_equal(centered.mean(axis=0), [0.0, 0.0])


def test_center_embeddings_preserves_relative_differences():
    embeddings = np.array([[1.0, 1.0], [3.0, 1.0], [2.0, 4.0]])
    centered = center_embeddings(embeddings)
    np.testing.assert_array_almost_equal(
        embeddings[1] - embeddings[0], centered[1] - centered[0]
    )
