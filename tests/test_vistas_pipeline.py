import json
from dataclasses import dataclass

import numpy as np

import drawthename.pipeline as pipeline
from drawthename.pipeline import (
    VISTAS_CLASS_NAMES,
    _downscale,
    _extract_and_embed_vistas_regions,
    run_vistas_pipeline,
)
from drawthename.regions import IGNORE_CLASS

ROAD, CAR, TRUCK = 0, 13, 14
_REGIONS_CFG = {
    "min_area_px": 64,
    "pad_px_min": 2,
    "pad_frac": 0.1,
    "error_rate_threshold": 0.5,
    "subdivision_size": None,
}


@dataclass
class _FakeSample:
    image_id: str
    image: np.ndarray
    ground_truth: np.ndarray
    continent: str


def _make_sample(image_id: str, continent: str) -> _FakeSample:
    """64x64 image: road everywhere, plus four 10x10 cars -- the two on the
    left are dark (predicted correctly by _FakeModel), the two on the right
    bright (mispredicted as truck)."""
    ground_truth = np.full((64, 64), ROAD, dtype=np.uint8)
    image = np.full((64, 64, 3), 100, dtype=np.uint8)
    for y0, x0 in [(5, 5), (40, 5), (5, 40), (40, 40)]:
        ground_truth[y0 : y0 + 10, x0 : x0 + 10] = CAR
        image[y0 : y0 + 10, x0 : x0 + 10] = 30 if x0 < 32 else 220
    return _FakeSample(image_id, image, ground_truth, continent)


class _FakeDataset:
    def __init__(self, samples: list[_FakeSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> _FakeSample:
        return self._samples[index]


class _FakeModel:
    """Predicts road everywhere except cars, and predicts right-half cars as
    truck."""

    def predict(self, image: np.ndarray) -> np.ndarray:
        prediction = np.full(image.shape[:2], ROAD, dtype=np.int64)
        car_mask = image[..., 0] != 100
        prediction[car_mask] = CAR
        prediction[:, 32:][car_mask[:, 32:]] = TRUCK
        return prediction


class _FakeBackbone:
    """Embeds a crop as its mean color plus a little noise, so error (bright)
    and correct (dark) crops land in separable directions."""

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def encode_image(self, crops: list[np.ndarray]) -> np.ndarray:
        features = np.array(
            [
                np.concatenate([crop.reshape(-1, 3).mean(axis=0) / 255, [1.0]])
                + self._rng.normal(0, 0.01, 4)
                for crop in crops
            ]
        )
        return features / np.linalg.norm(features, axis=1, keepdims=True)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        features = self._rng.normal(size=(len(texts), 4))
        return features / np.linalg.norm(features, axis=1, keepdims=True)


def test_downscale_shrinks_longer_side_and_keeps_class_ids():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    ground_truth = np.zeros((100, 200), dtype=np.uint8)
    ground_truth[:, 100:] = CAR
    small_image, small_gt = _downscale(image, ground_truth, max_side=50)
    assert small_image.shape == (25, 50, 3)
    assert small_gt.shape == (25, 50)
    assert set(np.unique(small_gt)) == {0, CAR}  # nearest: no blended ids


def test_downscale_noop_when_within_max_side_or_none():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    ground_truth = np.zeros((10, 20), dtype=np.uint8)
    assert _downscale(image, ground_truth, max_side=50)[1] is ground_truth
    assert _downscale(image, ground_truth, max_side=None)[1] is ground_truth


def test_extract_and_embed_restricts_to_shared_classes_and_drops_crops():
    dataset = _FakeDataset([_make_sample("a", "Europe"), _make_sample("b", "Asia")])
    regions, embeddings, pixel_counts = _extract_and_embed_vistas_regions(
        dataset, _FakeModel(), _FakeBackbone(), _REGIONS_CFG
    )

    assert {r.class_id for r in regions} == {CAR}  # road excluded
    assert len(regions) == len(embeddings) == 8
    assert sorted(r.label for r in regions) == ["correct"] * 4 + ["error"] * 4
    assert all(r.crop.size == 0 for r in regions)
    assert {(r.tile_id, r.country) for r in regions} == {
        ("a", "Europe"),
        ("b", "Asia"),
    }
    assert ROAD not in pixel_counts and IGNORE_CLASS not in pixel_counts
    assert pixel_counts[CAR] == (400, 800)  # 2 of 4 cars wrong, per image


def test_vistas_class_names_are_the_seven_shared_classes():
    assert sorted(VISTAS_CLASS_NAMES.values()) == sorted(
        ["person", "rider", "car", "truck", "bus", "motorcycle", "bicycle"]
    )


def test_run_vistas_pipeline_end_to_end(tmp_path, monkeypatch):
    samples = [
        _make_sample(f"{continent}{i}", continent)
        for continent in ("Europe", "Asia")
        for i in range(3)
    ]
    monkeypatch.setattr(
        pipeline, "MapillaryVistasDataset", lambda **_: _FakeDataset(samples)
    )
    monkeypatch.setattr(pipeline, "SegmentationModel", lambda **_: _FakeModel())
    monkeypatch.setattr(pipeline, "load_backbone", lambda *_, **__: _FakeBackbone())
    concept_bank = tmp_path / "concepts.txt"
    concept_bank.write_text("bright\ndark\ntruck\ncar\n")
    config = {
        "data": {"root": "unused", "limit": None, "max_side": None},
        "segmentation_model": {"checkpoint": "unused"},
        "backbone": {"name": "unused", "device": "cpu"},
        "regions": _REGIONS_CFG,
        "clustering": {"k_min": 2, "k_max": 3},
        "naming": {
            "bootstrap_resamples": 10,
            "cosine_threshold": 0.9,
            "stability_threshold": 0.95,
            "top_k_concepts": 2,
        },
        "geo_compare": {
            "intra_inter_divergence_threshold": 0.5,
            "concept_comparison_top_k": 2,
            "min_pair_region_count": 1,
        },
        "concept_bank": str(concept_bank),
        "plots": {"max_points_per_class": None},
    }
    output_dir = tmp_path / "out"

    run_vistas_pipeline(config, output_dir)

    payload = json.loads((output_dir / "bias_directions.json").read_text())
    assert payload["directions"]
    assert {d["class_id"] for d in payload["directions"]} == {CAR}
    assert all(d["country_confound_flag"] is not None for d in payload["directions"])
    embeddings = np.load(output_dir / "embeddings.npz")
    assert set(embeddings["country"]) == {"Europe", "Asia"}
    assert "car (class_id=13)" in (output_dir / "summary.md").read_text()
