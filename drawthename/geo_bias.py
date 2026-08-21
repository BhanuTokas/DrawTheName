"""Replicates the classification-vs-localization error decomposition from
"Classification Drives Geographic Bias in Street Scene Segmentation" (Nair,
Tseng, Rolf, Tokas, Kerner; CVPRW 2025, arXiv:2412.11061): per-continent,
per-class pixel IoU, the paper's Disp = sigma/mu geo-disparity metric, and
the class-merging technique that isolates how much of that disparity comes
from classification error (confusing two visually-similar classes) rather
than localization error (drawing the right class in the wrong place).

Independent of drawthename's own CLIP-based bias-naming pipeline -- this
reimplements the paper's own pixel-confusion-matrix methodology so its
result can be compared against the paper's published numbers directly,
rather than assuming our differently-designed naming pipeline would
reproduce the same metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

IGNORE_INDEX = 255
NUM_CLASSES = 19  # Cityscapes trainId space (0-18)


@dataclass
class ConfusionAccumulator:
    """Accumulates a (NUM_CLASSES, NUM_CLASSES) confusion matrix (rows=ground
    truth trainId, cols=predicted trainId) per geographic group (continent)."""

    matrices: dict[str, np.ndarray] = field(default_factory=dict)

    def update(
        self, group: str, ground_truth: np.ndarray, prediction: np.ndarray
    ) -> None:
        if group not in self.matrices:
            self.matrices[group] = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        valid = (ground_truth != IGNORE_INDEX) & (prediction != IGNORE_INDEX)
        gt, pred = (
            ground_truth[valid].astype(np.int64),
            prediction[valid].astype(np.int64),
        )
        flat_index = gt * NUM_CLASSES + pred
        counts = np.bincount(flat_index, minlength=NUM_CLASSES * NUM_CLASSES)
        self.matrices[group] += counts.reshape(NUM_CLASSES, NUM_CLASSES)


def iou_per_class(confusion: np.ndarray, class_id: int) -> float:
    """Standard IoU for one class from a (NUM_CLASSES, NUM_CLASSES) confusion
    matrix: TP / (TP + FP + FN). NaN if the class never appears as ground
    truth or prediction (undefined, not zero)."""
    tp = confusion[class_id, class_id]
    fp = confusion[:, class_id].sum() - tp
    fn = confusion[class_id, :].sum() - tp
    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else float("nan")


def merged_iou(confusion: np.ndarray, class_ids: list[int]) -> float:
    """IoU treating class_ids as a single merged class: a prediction that
    lands on any member of the group when ground truth is any member of the
    group counts as a true positive, recovering "right group, wrong
    fine-grained class" predictions that iou_per_class would count as both a
    false positive (for the predicted class) and a false negative (for the
    true class). This is the paper's class-merging technique: the percentage
    change in Disp before vs. after this recovery quantifies how much of the
    geo-disparity was caused by classification error rather than
    localization error."""
    ids = np.array(class_ids)
    tp = confusion[np.ix_(ids, ids)].sum()
    fp = confusion[:, ids].sum() - tp
    fn = confusion[ids, :].sum() - tp
    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else float("nan")


def disp(values: list[float]) -> float:
    """The paper's geo-disparity metric: population std / mean across
    geographic groups (e.g. continents) for one class. NaN values (a class
    absent from some continent) are dropped rather than propagated, since a
    class simply not appearing somewhere isn't evidence of bias there."""
    clean = np.array([v for v in values if not np.isnan(v)])
    if len(clean) == 0 or clean.mean() == 0:
        return float("nan")
    return float(clean.std() / clean.mean())


def pct_reduction(disp_before: float, disp_after: float) -> float:
    """Percent reduction in Disp after class-merging -- positive means
    merging shrank the geo-disparity (i.e. classification error was
    contributing to it), matching the paper's reported "reduced X%" framing.
    (The paper's own formula, Disp_corrected - Disp / Disp * 100, is the
    negative of this -- this returns the positive "how much better" number
    instead, which is what's actually reported in their tables/text.)"""
    if np.isnan(disp_before) or np.isnan(disp_after) or disp_before == 0:
        return float("nan")
    return float((disp_before - disp_after) / disp_before * 100)
