import numpy as np

from drawthename.clustering import cluster_embeddings, select_k_by_silhouette


def _two_blobs(n_per_blob: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blob_a = rng.normal(loc=[0, 0], scale=0.1, size=(n_per_blob, 2))
    blob_b = rng.normal(loc=[10, 10], scale=0.1, size=(n_per_blob, 2))
    return np.concatenate([blob_a, blob_b], axis=0)


def test_select_k_by_silhouette_finds_two_well_separated_blobs():
    embeddings = _two_blobs()
    k = select_k_by_silhouette(embeddings, k_min=2, k_max=8)
    assert k == 2


def test_select_k_by_silhouette_falls_back_when_too_few_samples():
    embeddings = np.zeros((2, 4))
    k = select_k_by_silhouette(embeddings, k_min=2, k_max=8)
    assert k == 1


def test_cluster_embeddings_separates_blobs():
    embeddings = _two_blobs()
    labels = cluster_embeddings(embeddings, k=2)
    assert len(labels) == len(embeddings)
    # the two blobs should end up in different clusters
    assert len(set(labels[:20])) == 1
    assert len(set(labels[20:])) == 1
    assert labels[0] != labels[20]
