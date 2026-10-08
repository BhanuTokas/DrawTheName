import json

import numpy as np
import pytest

from drawthename.naming import GlobalErrorMode, NamedDirection
from drawthename.pipeline import (
    _compute_global_error_mode,
    _stratified_subsample,
    _write_clusters,
    _write_embeddings,
    _write_summary,
)
from drawthename.regions import Region

_NAMING_CFG = {"bootstrap_resamples": 10, "cosine_threshold": 0.9, "top_k_concepts": 1}


def _make_region(label: str) -> Region:
    return Region(
        image_id="x",
        region_id=0,
        crop=np.zeros((4, 4, 3), dtype=np.uint8),
        label=label,
        class_id=0,
        pixel_error_rate=1.0 if label == "error" else 0.0,
    )


def test_compute_global_error_mode_raises_when_no_error_regions():
    regions = [_make_region("correct"), _make_region("correct")]
    embeddings = np.zeros((2, 8))
    with pytest.raises(ValueError, match="0 error region"):
        _compute_global_error_mode(
            regions, embeddings, ["x"], np.zeros((1, 8)), _NAMING_CFG
        )


def test_compute_global_error_mode_raises_when_no_correct_regions():
    regions = [_make_region("error"), _make_region("error")]
    embeddings = np.zeros((2, 8))
    with pytest.raises(ValueError, match="0 correct region"):
        _compute_global_error_mode(
            regions, embeddings, ["x"], np.zeros((1, 8)), _NAMING_CFG
        )


def test_compute_global_error_mode_per_country_cancels_country_mix():
    # Within each country, errors sit +1 along axis 0 from corrects. Country
    # "b" also sits +5 along axis 1 and holds most of the errors, so pooling
    # tilts the direction toward axis 1 ("looks like b"); per-country doesn't.
    def region(label, country):
        r = _make_region(label)
        r.country = country
        return r

    regions = (
        [region("error", "a")]
        + [region("correct", "a")] * 9
        + [region("error", "b")] * 9
        + [region("correct", "b")]
    )
    embeddings = np.array(
        [[1.0, 0.0]] + [[0.0, 0.0]] * 9 + [[1.0, 5.0]] * 9 + [[0.0, 5.0]]
    )
    concepts, concept_embeddings = ["x"], np.ones((1, 2))

    pooled = _compute_global_error_mode(
        regions, embeddings, concepts, concept_embeddings, _NAMING_CFG
    )
    per_country = _compute_global_error_mode(
        regions,
        embeddings,
        concepts,
        concept_embeddings,
        {**_NAMING_CFG, "global_error_mode_per_country": True},
    )
    assert pooled.bias_vector[1] > 1.0
    np.testing.assert_allclose(per_country.bias_vector, [1.0, 0.0])


def test_compute_global_error_mode_per_country_requires_country():
    regions = [_make_region("error"), _make_region("correct")]
    with pytest.raises(ValueError, match="needs Region.country"):
        _compute_global_error_mode(
            regions,
            np.zeros((2, 8)),
            ["x"],
            np.zeros((1, 8)),
            {**_NAMING_CFG, "global_error_mode_per_country": True},
        )


def test_stratified_subsample_noop_when_under_budget():
    groups = [np.zeros((10, 4)), np.zeros((5, 4))]
    result = _stratified_subsample(groups, max_total=100, rng=np.random.default_rng(0))
    assert result is groups


def test_stratified_subsample_preserves_proportions_and_budget():
    rng = np.random.default_rng(0)
    groups = [np.zeros((1000, 4)), np.zeros((100, 4)), np.zeros((10, 4))]
    result = _stratified_subsample(groups, max_total=222, rng=rng)

    sizes = [len(g) for g in result]
    # every non-empty group keeps at least one point
    assert all(n >= 1 for n in sizes)
    # roughly matches the input proportions (1000:100:10 ~= 10:1:0.1)
    assert sizes[0] > sizes[1] > sizes[2]
    # total stays close to the budget (small overage allowed from the min-1 floor)
    assert sum(sizes) <= 222 + len(groups)


def test_stratified_subsample_never_exceeds_group_size():
    rng = np.random.default_rng(0)
    groups = [np.zeros((3, 4)), np.zeros((3, 4))]
    result = _stratified_subsample(groups, max_total=4, rng=rng)
    assert all(len(g) <= 3 for g in result)


def test_write_summary_flags_low_residual_ratio(tmp_path):
    global_error_mode = GlobalErrorMode(
        bias_vector=np.zeros(4), concepts=["x"], stability=1.0
    )
    directions = [
        NamedDirection(
            class_id=0,
            cluster_id=0,
            bias_vector=np.zeros(4),
            concepts=["a"],
            stability=1.0,
            residual_ratio=0.02,
        ),
        NamedDirection(
            class_id=1,
            cluster_id=0,
            bias_vector=np.zeros(4),
            concepts=["b"],
            stability=1.0,
            residual_ratio=0.8,
        ),
    ]
    out_path = tmp_path / "summary.md"
    _write_summary(
        directions,
        global_error_mode,
        stability_threshold=0.95,
        path=out_path,
        class_names={},
        residual_ratio_threshold=0.1,
    )

    text = out_path.read_text()
    low_signal_line = next(line for line in text.splitlines() if "class_id=0" in line)
    stable_line = next(line for line in text.splitlines() if "class_id=1" in line)
    assert "low residual signal" in low_signal_line
    assert "low residual signal" not in stable_line


def test_write_embeddings_includes_cluster_id(tmp_path):
    regions = [_make_region("error"), _make_region("correct"), _make_region("error")]
    embeddings = np.zeros((3, 4))
    # region index 0 -> cluster 1, index 2 has no assignment (e.g. a class
    # with no named direction), index 1 (correct) is never clustered either.
    cluster_id_by_region_idx = {0: 1}
    out_path = tmp_path / "embeddings.npz"

    _write_embeddings(regions, embeddings, cluster_id_by_region_idx, out_path)

    data = np.load(out_path)
    np.testing.assert_array_equal(data["cluster_id"], [1, -1, -1])


def test_write_embeddings_country_loadable_without_allow_pickle(tmp_path):
    # Standard CV Mode never sets Region.country -- np.array([None, ...]) is
    # object-dtype, which np.load() can't read back without allow_pickle=True.
    # _write_embeddings must write a real (empty-string) sentinel instead, so
    # the file stays loadable with plain np.load().
    regions = [_make_region("error"), _make_region("correct")]
    embeddings = np.zeros((2, 4))
    out_path = tmp_path / "embeddings.npz"

    _write_embeddings(regions, embeddings, {}, out_path)

    data = np.load(out_path)  # no allow_pickle=True -- must not raise
    np.testing.assert_array_equal(data["country"], ["", ""])


def test_write_clusters_includes_silhouette_scores(tmp_path):
    directions = [
        NamedDirection(
            class_id=0,
            cluster_id=0,
            bias_vector=np.zeros(4),
            concepts=["a"],
            stability=1.0,
        ),
        NamedDirection(
            class_id=0,
            cluster_id=1,
            bias_vector=np.zeros(4),
            concepts=["b"],
            stability=1.0,
        ),
    ]
    silhouette_by_class = {0: 0.42}
    out_path = tmp_path / "clusters.json"

    _write_clusters(directions, silhouette_by_class, out_path)

    payload = json.loads(out_path.read_text())
    assert payload["k_selected"] == {"0": 2}
    assert payload["silhouette_scores"] == {"0": 0.42}
    assert payload["clusters"] == {"0": [0, 1]}
