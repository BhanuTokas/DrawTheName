import numpy as np
import pytest

from drawthename.geo_bias import (
    IGNORE_INDEX,
    NUM_CLASSES,
    ConfusionAccumulator,
    disp,
    iou_per_class,
    merged_iou,
    pct_reduction,
)


def test_confusion_accumulator_counts_correctly():
    acc = ConfusionAccumulator()
    gt = np.array([1, 1, 2, 2, IGNORE_INDEX])
    pred = np.array([1, 2, 2, 2, 1])  # last pair ignored (gt is IGNORE_INDEX)
    acc.update("europe", gt, pred)

    matrix = acc.matrices["europe"]
    assert matrix.shape == (NUM_CLASSES, NUM_CLASSES)
    assert matrix[1, 1] == 1  # correct class-1 prediction
    assert matrix[1, 2] == 1  # class-1 mistaken for class-2
    assert matrix[2, 2] == 2  # both class-2 pixels correct
    assert matrix.sum() == 4  # the ignored pixel isn't counted anywhere


def test_confusion_accumulator_accumulates_across_updates():
    acc = ConfusionAccumulator()
    acc.update("europe", np.array([1]), np.array([1]))
    acc.update("europe", np.array([1]), np.array([1]))
    assert acc.matrices["europe"][1, 1] == 2


def test_confusion_accumulator_also_drops_ignored_predictions():
    acc = ConfusionAccumulator()
    acc.update("europe", np.array([1]), np.array([IGNORE_INDEX]))
    assert acc.matrices["europe"].sum() == 0


def test_iou_per_class_perfect_prediction():
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    confusion[3, 3] = 10
    assert iou_per_class(confusion, 3) == pytest.approx(1.0)


def test_iou_per_class_with_confusion():
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    confusion[3, 3] = 5  # true positives
    confusion[3, 4] = 5  # class 3 mistaken for class 4 (false negatives for 3)
    confusion[4, 3] = 5  # class 4 mistaken for class 3 (false positives for 3)
    # TP=5, FP=5, FN=5 -> IoU = 5 / 15
    assert iou_per_class(confusion, 3) == pytest.approx(5 / 15)


def test_iou_per_class_undefined_is_nan():
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    assert np.isnan(iou_per_class(confusion, 3))


def test_merged_iou_recovers_within_group_confusion():
    """car(13) predicted as truck(14) or vice versa should count as a true
    positive once merged, recovering the IoU that per-class scoring lost."""
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    confusion[13, 13] = 5
    confusion[13, 14] = 5  # car mistaken for truck
    confusion[14, 14] = 5
    confusion[14, 13] = 5  # truck mistaken for car

    car_iou = iou_per_class(confusion, 13)
    truck_iou = iou_per_class(confusion, 14)
    merged = merged_iou(confusion, [13, 14])

    assert car_iou == pytest.approx(5 / 15)
    assert truck_iou == pytest.approx(5 / 15)
    assert merged == pytest.approx(1.0)  # all 20 pixels land in-group


def test_merged_iou_does_not_recover_out_of_group_confusion():
    """car(13) mistaken for a class outside the merge group should still
    count as an error after merging -- merging only forgives *within-group*
    misclassification."""
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    confusion[13, 13] = 5
    confusion[13, 11] = 5  # car mistaken for person (not in the vehicle group)

    merged = merged_iou(confusion, [13, 14])  # car+truck group; person not included
    # TP=5 (car->car), FP=0, FN=5 (car->person, still an error) -> 5/10
    assert merged == pytest.approx(5 / 10)


def test_disp_zero_for_identical_values():
    assert disp([0.5, 0.5, 0.5]) == pytest.approx(0.0)


def test_disp_positive_for_varied_values():
    assert disp([0.2, 0.8]) > 0


def test_disp_drops_nan_values():
    # a class missing from one continent (NaN) shouldn't be treated as 0,
    # which would inflate Disp spuriously for every other class too.
    with_nan = disp([0.5, 0.5, float("nan")])
    without_nan = disp([0.5, 0.5])
    assert with_nan == pytest.approx(without_nan)


def test_disp_nan_when_all_values_missing():
    assert np.isnan(disp([float("nan"), float("nan")]))


def test_pct_reduction_positive_when_disp_shrinks():
    assert pct_reduction(0.5, 0.1) == pytest.approx(80.0)


def test_pct_reduction_negative_when_disp_grows():
    assert pct_reduction(0.1, 0.5) == pytest.approx(-400.0)


def test_pct_reduction_nan_when_before_is_zero():
    assert np.isnan(pct_reduction(0.0, 0.1))
