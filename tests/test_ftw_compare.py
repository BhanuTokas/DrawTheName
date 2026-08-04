import numpy as np

from drawthename.ftw_compare import (
    compare_concept_sets,
    flag_confound,
    inter_country_direction,
    inter_tile_direction,
    intra_country_direction,
    intra_tile_direction,
)
from drawthename.regions import Region


def _region(tile_id: str, label: str, country: str | None = None) -> Region:
    return Region(
        image_id=tile_id,
        region_id=0,
        crop=np.zeros((4, 4, 3), dtype=np.uint8),
        label=label,
        class_id=1,
        pixel_error_rate=1.0 if label == "error" else 0.0,
        tile_id=tile_id,
        country=country,
    )


def test_intra_tile_direction_none_when_no_shared_tile():
    error_regions = [_region("tile_a", "error")]
    correct_regions = [_region("tile_b", "correct")]
    result = intra_tile_direction(
        error_regions, np.array([[1.0, 0.0]]), correct_regions, np.array([[0.0, 1.0]])
    )
    assert result is None


def test_intra_tile_direction_averages_within_tile_diffs():
    # tile_a: error=[2, 0], correct=[0, 0] -> diff [2, 0]
    # tile_b: error=[0, 4], correct=[0, 0] -> diff [0, 4]
    error_regions = [_region("tile_a", "error"), _region("tile_b", "error")]
    correct_regions = [_region("tile_a", "correct"), _region("tile_b", "correct")]
    error_embeddings = np.array([[2.0, 0.0], [0.0, 4.0]])
    correct_embeddings = np.array([[0.0, 0.0], [0.0, 0.0]])

    result = intra_tile_direction(
        error_regions, error_embeddings, correct_regions, correct_embeddings
    )
    np.testing.assert_allclose(result, [1.0, 2.0])


def test_inter_tile_direction_matches_plain_bias_direction():
    error_regions = [_region("tile_a", "error")]
    correct_regions = [_region("tile_b", "correct")]
    error_embeddings = np.array([[3.0, 0.0]])
    correct_embeddings = np.array([[1.0, 0.0]])

    result = inter_tile_direction(
        error_regions, error_embeddings, correct_regions, correct_embeddings
    )
    np.testing.assert_allclose(result, [2.0, 0.0])


def test_flag_confound_true_for_orthogonal_directions():
    assert (
        flag_confound(np.array([1.0, 0.0]), np.array([0.0, 1.0]), threshold=0.5) is True
    )


def test_flag_confound_false_for_aligned_directions():
    assert (
        flag_confound(np.array([1.0, 0.0]), np.array([2.0, 0.0]), threshold=0.5)
        is False
    )


def test_intra_country_direction_none_when_no_shared_country():
    error_regions = [_region("t1", "error", country="austria")]
    correct_regions = [_region("t2", "correct", country="france")]
    result = intra_country_direction(
        error_regions, np.array([[1.0, 0.0]]), correct_regions, np.array([[0.0, 1.0]])
    )
    assert result is None


def test_intra_country_direction_averages_within_country_diffs():
    # austria: error tiles average [2, 0], correct tiles average [0, 0] -> diff [2, 0]
    # france: error tiles average [0, 4], correct tiles average [0, 0] -> diff [0, 4]
    error_regions = [
        _region("a1", "error", country="austria"),
        _region("a2", "error", country="austria"),
        _region("f1", "error", country="france"),
    ]
    correct_regions = [
        _region("a3", "correct", country="austria"),
        _region("f2", "correct", country="france"),
    ]
    error_embeddings = np.array([[3.0, 0.0], [1.0, 0.0], [0.0, 4.0]])
    correct_embeddings = np.array([[0.0, 0.0], [0.0, 0.0]])

    result = intra_country_direction(
        error_regions, error_embeddings, correct_regions, correct_embeddings
    )
    np.testing.assert_allclose(result, [1.0, 2.0])


def test_inter_country_direction_none_when_single_country():
    error_regions = [_region("t1", "error", country="austria")]
    correct_regions = [_region("t2", "correct", country="austria")]
    result = inter_country_direction(
        error_regions, np.array([[1.0, 0.0]]), correct_regions, np.array([[0.0, 1.0]])
    )
    assert result is None


def test_inter_country_direction_excludes_same_country_pairs():
    # austria error=[4, 0], austria correct=[0, 0] -- same-country pair, must be excluded
    # france correct=[1, 1]
    # only cross-country pair possible: austria error [4,0] vs france correct [1,1] -> [3,-1]
    error_regions = [_region("a1", "error", country="austria")]
    correct_regions = [
        _region("a2", "correct", country="austria"),
        _region("f1", "correct", country="france"),
    ]
    error_embeddings = np.array([[4.0, 0.0]])
    correct_embeddings = np.array([[0.0, 0.0], [1.0, 1.0]])

    result = inter_country_direction(
        error_regions, error_embeddings, correct_regions, correct_embeddings
    )
    np.testing.assert_allclose(result, [3.0, -1.0])


def test_compare_concept_sets_buckets_correctly():
    result = compare_concept_sets(
        intra_tile_concepts=["a", "b", "c"],
        intra_country_concepts=["b", "c", "d"],
        inter_country_concepts=["c", "d", "e"],
    )
    assert result["prevalent"] == ["c"]
    assert result["tile_sensitive"] == ["d"]
    assert result["domain_shift_candidates"] == ["e"]
