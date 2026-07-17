import numpy as np

from drawthename.regions import compute_error_mask, extract_regions


def test_compute_error_mask_marks_mismatches():
    prediction = np.array([[0, 1], [1, 1]])
    ground_truth = np.array([[0, 0], [1, 1]])
    mask = compute_error_mask(prediction, ground_truth)
    np.testing.assert_array_equal(mask, [[0, 1], [0, 0]])


def _make_scene():
    # 20x20 image: left half class 0 (all-error), right half class 1 (all-correct).
    ground_truth = np.zeros((20, 20), dtype=np.uint8)
    ground_truth[:, 10:] = 1
    error_mask = np.zeros((20, 20), dtype=np.uint8)
    error_mask[:, :10] = 1  # class 0 fully wrong
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    return image, error_mask, ground_truth


def test_extract_regions_labels_by_error_rate():
    image, error_mask, ground_truth = _make_scene()
    regions = extract_regions(
        image, error_mask, ground_truth, image_id="scene0", min_area_px=10
    )

    by_class = {r.class_id: r for r in regions}
    assert by_class[0].label == "error"
    assert by_class[0].pixel_error_rate == 1.0
    assert by_class[1].label == "correct"
    assert by_class[1].pixel_error_rate == 0.0


def test_extract_regions_discards_small_components():
    image, error_mask, ground_truth = _make_scene()
    regions = extract_regions(
        image, error_mask, ground_truth, image_id="scene0", min_area_px=1000
    )
    assert regions == []


def test_extract_regions_ignores_ignore_class():
    image, error_mask, ground_truth = _make_scene()
    ground_truth[:, :] = 255
    regions = extract_regions(
        image, error_mask, ground_truth, image_id="scene0", min_area_px=10
    )
    assert regions == []


def test_extract_regions_subdivides_large_components():
    # 40x40, single class, all correct -- one component, area 1600.
    ground_truth = np.zeros((40, 40), dtype=np.uint8)
    error_mask = np.zeros((40, 40), dtype=np.uint8)
    image = np.zeros((40, 40, 3), dtype=np.uint8)

    whole = extract_regions(
        image, error_mask, ground_truth, image_id="scene0", min_area_px=10
    )
    assert len(whole) == 1

    tiled = extract_regions(
        image,
        error_mask,
        ground_truth,
        image_id="scene0",
        min_area_px=10,
        subdivision_size=10,
    )
    assert len(tiled) == 16  # 40x40 split into 10x10 tiles
    assert all(r.label == "correct" for r in tiled)


def test_extract_regions_subdivision_leaves_small_components_whole():
    image, error_mask, ground_truth = _make_scene()
    regions = extract_regions(
        image,
        error_mask,
        ground_truth,
        image_id="scene0",
        min_area_px=10,
        subdivision_size=192,
    )
    assert len(regions) == 2  # each half (10x20=200px) is well under 192^2, stays whole
