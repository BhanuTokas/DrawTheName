"""Placeholder for connected-component region extraction tests (Phase 1)."""

import numpy as np

from drawthename.regions import compute_error_mask


def test_compute_error_mask_matches_shape():
    prediction = np.zeros((4, 4), dtype=np.uint8)
    ground_truth = np.zeros((4, 4), dtype=np.uint8)
    try:
        compute_error_mask(prediction, ground_truth)
    except NotImplementedError:
        pass
