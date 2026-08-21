import json
from dataclasses import dataclass

import numpy as np
import pytest

from drawthename.geo_bias import IGNORE_INDEX, ConfusionAccumulator
from drawthename.geo_bias_pipeline import _compute_results, _run_inference


@dataclass
class _FakeSample:
    image: np.ndarray
    ground_truth: np.ndarray
    continent: str


class _FakeDataset:
    def __init__(self, samples: list[_FakeSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> _FakeSample:
        return self._samples[index]


class _FakeModel:
    """Always predicts class 13 (car) everywhere -- exercises correct vs.
    confused predictions deterministically without any real inference."""

    def predict(self, image: np.ndarray) -> np.ndarray:
        return np.full(image.shape[:2], 13, dtype=np.int64)


def _sample(continent: str, ground_truth_class: int) -> _FakeSample:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    ground_truth = np.full((2, 2), ground_truth_class, dtype=np.uint8)
    return _FakeSample(image=image, ground_truth=ground_truth, continent=continent)


def test_run_inference_accumulates_per_continent():
    dataset = _FakeDataset(
        [
            _sample("Europe", ground_truth_class=13),  # car -- model gets it right
            _sample("Africa", ground_truth_class=14),  # truck -- model predicts car instead
        ]
    )
    accumulator, n_images_by_continent = _run_inference(dataset, _FakeModel())

    assert n_images_by_continent == {"Europe": 1, "Africa": 1}
    assert accumulator.matrices["Europe"][13, 13] == 4  # 2x2 image, all correct
    assert accumulator.matrices["Africa"][14, 13] == 4  # truck mistaken for car


def test_run_inference_respects_limit():
    dataset = _FakeDataset([_sample("Europe", 13) for _ in range(5)])
    accumulator, n_images_by_continent = _run_inference(dataset, _FakeModel(), limit=2)
    assert n_images_by_continent == {"Europe": 2}
    assert accumulator.matrices["Europe"].sum() == 8  # 2 images x 4 pixels


def test_run_inference_ignores_ignore_index_pixels():
    sample = _FakeSample(
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        ground_truth=np.array([[13, IGNORE_INDEX], [13, 13]], dtype=np.uint8),
        continent="Europe",
    )
    accumulator, _ = _run_inference(_FakeDataset([sample]), _FakeModel())
    assert accumulator.matrices["Europe"].sum() == 3  # the ignored pixel is dropped


def test_compute_results_merges_car_and_truck_group():
    accumulator = ConfusionAccumulator()
    # Both continents start with identical pre-merge truck IoU (1 correct,
    # 1 misclassified truck pixel each) -- so Disp before merging is exactly
    # 0 (no geo-disparity yet). Europe's mistake is truck->car (within the
    # 4-wheeler merge group, so merging recovers it to IoU 1.0); Africa's is
    # truck->person (outside the group, so merging can't recover it and its
    # IoU stays 0.5). Merging therefore *creates* a disparity here rather
    # than closing one -- illustrating that the merge only recovers
    # within-group confusion, it isn't a metric that's guaranteed to shrink
    # Disp in every case (the paper's reductions are an empirical finding on
    # real data, not a mathematical property of the merge itself).
    accumulator.update("Europe", np.array([14, 14]), np.array([13, 14]))
    accumulator.update("Africa", np.array([14, 14]), np.array([11, 14]))

    results = _compute_results(accumulator, {"Europe": 1, "Africa": 1})

    assert results["continents"] == ["Africa", "Europe"]
    truck = results["classes"]["truck"]
    assert truck["merge_group"] == "4-wheeler"
    assert truck["iou_by_continent"]["Europe"] == pytest.approx(0.5)
    assert truck["iou_by_continent"]["Africa"] == pytest.approx(0.5)
    assert truck["disp_before_merge"] == pytest.approx(0.0)

    assert truck["merged_iou_by_continent"]["Europe"] == pytest.approx(1.0)
    assert truck["merged_iou_by_continent"]["Africa"] == pytest.approx(0.5)
    assert truck["disp_after_merge"] == pytest.approx(1 / 3)
    # disp_before_merge is exactly 0 here, so "% reduction" is undefined
    # (matches pct_reduction's own NaN-on-zero-baseline behavior) even
    # though Disp measurably grew in absolute terms.
    assert np.isnan(truck["disp_pct_reduction"])


def test_compute_results_json_serializable():
    accumulator = ConfusionAccumulator()
    accumulator.update("Europe", np.array([13]), np.array([13]))
    results = _compute_results(accumulator, {"Europe": 1})
    json.dumps(results)  # raises if anything is a bare numpy type
