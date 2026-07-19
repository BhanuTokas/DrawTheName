"""FTW (Fields of the World) dataloader for FTW Mode (Phase 2).

Wraps ftw_tools.training.datasets.FTW (which already knows FTW's tile/
label-mask file layout, country/split structure, and checksum validation)
the same way CityscapesDataset wraps torchvision.datasets.Cityscapes.

Note on NIR: the spec assumed PRUE consumes 4-band RGB+NIR input and built
an NIR-dropping step around that (see to_rgb below). The actual local PRUE
checkpoint (prue-unet-logcoshdice-augs-efficientnetb3-winargb) was trained
with temporal_options="window_a_rgb" -- a single time window, already
RGB-only, in_channels=3. ftw_tools's own dataset loader hands back RGB
directly for that option, so to_rgb is a no-op for this checkpoint; it's
kept for any future PRUE variant that does take 4+ bands.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
from ftw_tools.training.datasets import FTW as _FTWTools
from torch.utils.data import Dataset

TileClassification = Literal["all-correct", "all-error", "mixed"]

# matches the checkpoint's num_classes=3 (label_masks/semantic_3class)
CLASS_NAMES = {0: "background", 1: "field-interior", 2: "field-boundary"}

IGNORE_CLASS = 255  # FTW's 3-class masks have no ignore label; kept for interface parity with Cityscapes

# Matches PRUEModel.FTW_REFLECTANCE_SCALE (the value PRUE was trained
# against); reused here as a simple linear clip+stretch to uint8 for
# CLIP-facing crops. Not a radiometric/atmospheric color correction -- good
# enough for bias-direction embedding, not for photometric accuracy.
DISPLAY_REFLECTANCE_CLIP = 3000.0


@dataclass
class FTWTile:
    tile_id: str  # aoi_id, e.g. "g77_00002_10"
    image: np.ndarray  # (H, W, 3) float32 Sentinel-2 reflectance, RGB (window A)
    ground_truth: (
        np.ndarray
    )  # (H, W) uint8, 0=background, 1=field-interior, 2=field-boundary
    geography: str  # tile centroid, "lat,lon"
    acquisition_date: str | None = (
        None  # not present in this dataset's metadata (chips parquet has no date column)
    )


class FTWDataset(Dataset):
    """Loads FTW Sentinel-2 tiles and field-boundary masks for a country/split."""

    def __init__(self, root: Path, countries: list[str], split: str = "val") -> None:
        self.root = Path(root)
        self.countries = [c.lower() for c in countries]
        self._dataset = _FTWTools(
            root=str(root),
            countries=self.countries,
            split=split,
            temporal_options="window_a_rgb",
            load_boundaries=True,
        )
        self._centroid_by_aoi_id = self._load_centroids()

    def _load_centroids(self) -> dict[str, tuple[float, float]]:
        # Tiles are small (a few hundred meters), so the midpoint of the
        # lon/lat bounding box is an adequate stand-in for a true centroid
        # and sidesteps geopandas' geographic-CRS centroid warning.
        centroids: dict[str, tuple[float, float]] = {}
        for country in self.countries:
            chips_path = self.root / country / f"chips_{country}.parquet"
            chips_df = gpd.read_parquet(chips_path)
            bounds = chips_df.geometry.bounds
            lat = (bounds["miny"] + bounds["maxy"]) / 2
            lon = (bounds["minx"] + bounds["maxx"]) / 2
            for aoi_id, lat_val, lon_val in zip(
                chips_df["aoi_id"], lat, lon, strict=True
            ):
                centroids[aoi_id] = (lat_val, lon_val)
        return centroids

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> FTWTile:
        sample = self._dataset[index]
        tile_id = Path(self._dataset.filenames[index]["window_a"]).stem

        image = sample["image"].numpy().transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
        ground_truth = sample["mask"].numpy().astype(np.uint8)

        lat, lon = self._centroid_by_aoi_id.get(tile_id, (float("nan"), float("nan")))
        return FTWTile(
            tile_id=tile_id,
            image=image,
            ground_truth=ground_truth,
            geography=f"{lat:.4f},{lon:.4f}",
        )


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Drops any channel beyond the first 3 from a (H, W, C) array; VLM
    backbones are RGB-only (spec section 7). No-op for the current
    checkpoint, which is already RGB."""
    return image[..., :3]


def to_display_rgb(
    image: np.ndarray, reflectance_clip: float = DISPLAY_REFLECTANCE_CLIP
) -> np.ndarray:
    """Converts (H, W, 3) float32 Sentinel-2 reflectance to uint8 RGB for
    CLIP-facing region crops -- encode_image's PIL conversion expects a
    standard 0-255 image, not raw reflectance. PRUEModel.predict takes the
    raw reflectance tile directly and does its own normalization; this is
    only for the embedding/visualization path."""
    stretched = np.clip(image, 0, reflectance_clip) / reflectance_clip
    return (stretched * 255).astype(np.uint8)


def classify_tile(error_mask: np.ndarray) -> TileClassification:
    """Labels a tile all-correct (zero error pixels), all-error (every pixel
    wrong), or mixed (anything in between -- these are the tiles where
    intra-tile comparison is meaningful, since they have both error and
    correct sub-regions to compare)."""
    error_rate = float(error_mask.mean())
    if error_rate == 0.0:
        return "all-correct"
    if error_rate == 1.0:
        return "all-error"
    return "mixed"
