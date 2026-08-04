import numpy as np

from drawthename.data.ftw import classify_tile, remap_classes, to_rgb


def test_classify_tile_all_correct():
    error_mask = np.zeros((4, 4), dtype=bool)
    assert classify_tile(error_mask) == "all-correct"


def test_classify_tile_all_error():
    error_mask = np.ones((4, 4), dtype=bool)
    assert classify_tile(error_mask) == "all-error"


def test_classify_tile_mixed():
    error_mask = np.zeros((4, 4), dtype=bool)
    error_mask[0, 0] = True
    assert classify_tile(error_mask) == "mixed"


def test_to_rgb_passthrough_for_three_channel():
    image = np.zeros((8, 8, 3), dtype=np.float32)
    result = to_rgb(image)
    assert result.shape == (8, 8, 3)


def test_to_rgb_drops_extra_channels():
    image = np.zeros((8, 8, 4), dtype=np.float32)
    result = to_rgb(image)
    assert result.shape == (8, 8, 3)


def test_remap_classes_shifts_labeled_values():
    ground_truth = np.array([[1, 2], [3, 1]], dtype=np.uint8)
    result = remap_classes(ground_truth, {1: 0, 2: 1, 3: 2})
    np.testing.assert_array_equal(result, [[0, 1], [2, 0]])


def test_remap_classes_leaves_unmapped_values_unchanged():
    ground_truth = np.array([0, 1, 5], dtype=np.uint8)
    result = remap_classes(ground_truth, {1: 2})
    np.testing.assert_array_equal(result, [0, 2, 5])
