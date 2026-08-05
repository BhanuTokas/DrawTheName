import numpy as np

from drawthename.data.ftw import FTW_UNKNOWN_CLASS, IGNORE_CLASS, remap_classes, to_rgb


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


def test_ftw_unknown_class_matches_dataset_convention():
    # FTW's own training code labels its 3-class scheme
    # ["background", "field", "boundary", "unknown"] and every 3-class
    # checkpoint (including the PRUE one used here) trains with
    # ignore_index=3 -- this is a dataset-wide standard, not a per-country
    # quirk, so FTWDataset.__getitem__ remaps it unconditionally for every
    # country before any user-supplied class_remap runs.
    assert FTW_UNKNOWN_CLASS == 3


def test_unknown_class_remap_composes_with_a_per_country_remap():
    # Mirrors FTWDataset.__getitem__: the unconditional unknown->ignore
    # remap runs first, then any user-supplied per-country remap on top.
    ground_truth = np.array([0, 1, 2, 3], dtype=np.uint8)
    after_unknown_remap = remap_classes(ground_truth, {FTW_UNKNOWN_CLASS: IGNORE_CLASS})
    np.testing.assert_array_equal(after_unknown_remap, [0, 1, 2, IGNORE_CLASS])

    after_country_remap = remap_classes(after_unknown_remap, {0: 9})
    np.testing.assert_array_equal(after_country_remap, [9, 1, 2, IGNORE_CLASS])
