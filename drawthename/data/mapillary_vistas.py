"""Mapillary Vistas v1.2 dataloader, built specifically to replicate
"Classification Drives Geographic Bias in Street Scene Segmentation"
(Nair, Tseng, Rolf, Tokas, Kerner; CVPRW 2025, arXiv:2412.11061) against
DrawTheName's own tooling, as an external reliability check: does an
independently-built bias-discovery method reproduce a peer-reviewed finding
on the same data?

Vistas ships no per-image geographic metadata (GPS is stripped for privacy
in the public research release) -- continent labels come from a companion
lat/lon file the paper's own authors published separately
(https://zenodo.org/records/11459554, ASU). Of its 20,000 rows only 11,300
have coordinates; that number matches the paper's own reported post-
preprocessing image count exactly, so this is very likely the exact metadata
source the paper itself used. See scripts/build_mapillary_continent_labels.py
for how mapillary_vistas_continents.csv (bundled alongside this module) was
derived from it: offline reverse-geocoding (no network calls), 6-continent
scheme (Europe/North America/South America/Africa/Asia/Oceania), 11,300 of
11,300 coordinate-bearing rows successfully labeled.

Directory layout assumed (the standard public v1.2 research-dataset layout;
not yet verified against a local copy since Vistas isn't downloaded on this
machine -- confirm against the actual HPC download before a real run):
    <root>/training/images/<key>.jpg
    <root>/training/labels/<key>.png     (single-channel, pixel value = raw
                                           Vistas v1.2 class id 0-64)
    <root>/validation/images/<key>.jpg
    <root>/validation/labels/<key>.png
Only training+validation are used (matching the paper: Vistas' testing split
ships images with no public ground truth), combined into one pool the same
way the continent CSV combines them.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = 255  # matches drawthename.data.cityscapes' convention

DEFAULT_CONTINENT_LABELS_PATH = (
    Path(__file__).parent / "mapillary_vistas_continents.csv"
)

# Raw Mapillary Vistas v1.2 label id -> Cityscapes trainId (0-18), matching
# the id2label ordering published at
# huggingface.co/datasets/huggingface/label-files/blob/main/mapillary-vistas-id2label.json
# (the standard 65-class v1.2 semantic taxonomy). This is the widely-used
# published Vistas<->Cityscapes correspondence table (e.g. as used to
# jointly train on both datasets); not the paper's own table verbatim, since
# that wasn't recoverable from the paper text -- only the 7 "thing" classes
# below (person/rider/car/truck/bus/motorcycle/bicycle) matter for this
# replication, so the "stuff" side of the table (road/sidewalk/etc.) just
# needs to be *a* reasonable mapping, not an exact match to the paper's,
# since it only affects classes this experiment doesn't score.
_MAPILLARY_ID_TO_TRAINID: dict[int, int] = {
    13: 0,  # Road -> road
    15: 1,  # Sidewalk -> sidewalk
    17: 2,  # Building -> building
    6: 3,  # Wall -> wall
    3: 4,  # Fence -> fence
    45: 5,  # Pole -> pole
    47: 5,  # Utility Pole -> pole
    48: 6,  # Traffic Light -> traffic light
    50: 7,  # Traffic Sign (Front) -> traffic sign
    30: 8,  # Vegetation -> vegetation
    29: 9,  # Terrain -> terrain
    27: 10,  # Sky -> sky
    19: 11,  # Person -> person
    20: 12,  # Bicyclist -> rider
    21: 12,  # Motorcyclist -> rider
    22: 12,  # Other Rider -> rider
    55: 13,  # Car -> car
    61: 14,  # Truck -> truck
    54: 15,  # Bus -> bus
    58: 16,  # On Rails -> train
    57: 17,  # Motorcycle -> motorcycle
    52: 18,  # Bicycle -> bicycle
}
# everything else (Bird, Curb, Barrier, Bike Lane, Crosswalk, Parking,
# Pedestrian Area, Rail Track, Service Lane, Bridge, Tunnel, lane markings,
# Mountain, Sand, Snow, Water, street furniture, Caravan, Other Vehicle,
# Trailer, Wheeled Slow, Boat, Car Mount, Ego Vehicle, ...) -> IGNORE_INDEX.
MAPILLARY_ID_TO_TRAINID = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _mapillary_id, _train_id in _MAPILLARY_ID_TO_TRAINID.items():
    MAPILLARY_ID_TO_TRAINID[_mapillary_id] = _train_id

# The paper's 7 classes shared between Cityscapes and Vistas' instance
# taxonomy, in Cityscapes trainId space.
SHARED_CLASSES: dict[str, int] = {
    "person": 11,
    "rider": 12,
    "car": 13,
    "truck": 14,
    "bus": 15,
    "motorcycle": 17,
    "bicycle": 18,
}

# The paper's class-merging groups for isolating classification-error
# contribution (car/bus/truck confused for each other, motorcycle/bicycle
# confused for each other, person/rider confused for each other -- merging
# recovers IoU for a "right group, wrong fine-grained class" prediction).
MERGE_GROUPS: dict[str, list[str]] = {
    "4-wheeler": ["car", "truck", "bus"],
    "2-wheeler": ["motorcycle", "bicycle"],
    "human": ["person", "rider"],
}


def load_continent_labels(path: Path = DEFAULT_CONTINENT_LABELS_PATH) -> dict[str, str]:
    """Mapillary_ID -> continent name, from the bundled CSV."""
    labels: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["Mapillary_ID"]] = row["continent"]
    return labels


@dataclass
class MapillaryVistasSample:
    image_id: str  # Mapillary_ID (matches the continent CSV's key)
    image: np.ndarray  # (H, W, 3) uint8 RGB
    ground_truth: np.ndarray  # (H, W) uint8, Cityscapes trainIds 0-18 or IGNORE_INDEX
    continent: str


class MapillaryVistasDataset(Dataset):
    """Combined training+validation split, restricted to images with a known
    continent (drops the ~8,700/20,000 images the companion geolocation file
    has no coordinates for -- matches the paper's own preprocessing)."""

    def __init__(
        self,
        root: Path,
        continent_labels_path: Path = DEFAULT_CONTINENT_LABELS_PATH,
        splits: tuple[str, ...] = ("training", "validation"),
    ) -> None:
        self.root = Path(root)
        continent_by_id = load_continent_labels(continent_labels_path)

        self._samples: list[tuple[Path, Path, str, str]] = []
        for split in splits:
            images_dir = self.root / split / "images"
            labels_dir = self.root / split / "labels"
            for image_path in sorted(images_dir.glob("*.jpg")):
                image_id = image_path.stem
                continent = continent_by_id.get(image_id)
                if continent is None:
                    continue
                label_path = labels_dir / f"{image_id}.png"
                self._samples.append((image_path, label_path, image_id, continent))

        if not self._samples:
            raise ValueError(
                f"No Vistas images with a known continent found under {self.root} "
                f"(splits={splits}). Check that root points at the extracted "
                "dataset and that continent_labels_path matches its Mapillary_IDs."
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> MapillaryVistasSample:
        image_path, label_path, image_id, continent = self._samples[index]
        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        raw_labels = np.array(Image.open(label_path), dtype=np.uint8)
        ground_truth = MAPILLARY_ID_TO_TRAINID[raw_labels]
        return MapillaryVistasSample(
            image_id=image_id,
            image=image,
            ground_truth=ground_truth,
            continent=continent,
        )
