"""Cityscapes dataloader for Standard CV Mode (Phase 1).

Wraps torchvision.datasets.Cityscapes (which already knows the official
leftImg8bit/gtFine layout and label metadata) and remaps labelIds to the 19
trainIds used for evaluation. Pixels outside those 19 classes are set to
IGNORE_INDEX (255) and excluded from region extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes as _TVCityscapes

IGNORE_INDEX = 255

# id (0-33, as stored in *_gtFine_labelIds.png) -> trainId (0-18, or 255 to ignore).
_ID_TO_TRAINID = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _cls in _TVCityscapes.classes:
    if _cls.id >= 0:
        _ID_TO_TRAINID[_cls.id] = _cls.train_id if not _cls.ignore_in_eval else IGNORE_INDEX

TRAIN_ID_NAMES = {
    cls.train_id: cls.name
    for cls in _TVCityscapes.classes
    if not cls.ignore_in_eval and cls.train_id != IGNORE_INDEX
}


@dataclass
class CityscapesSample:
    image_id: str
    image: np.ndarray  # (H, W, 3) uint8 RGB
    ground_truth: np.ndarray  # (H, W) uint8, values in 0-18 or IGNORE_INDEX


class CityscapesDataset(Dataset):
    """Loads Cityscapes images and fine-annotation semantic masks for a split."""

    def __init__(self, root: Path, split: str = "val") -> None:
        self._dataset = _TVCityscapes(
            str(root), split=split, mode="fine", target_type="semantic"
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> CityscapesSample:
        image, label_ids = self._dataset[index]
        image_path = Path(self._dataset.images[index])
        label_ids = np.array(label_ids, dtype=np.uint8)
        ground_truth = _ID_TO_TRAINID[label_ids]
        return CityscapesSample(
            image_id=image_path.stem,
            image=np.array(image.convert("RGB"), dtype=np.uint8),
            ground_truth=ground_truth,
        )
