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

import geopandas as gpd
import numpy as np
from ftw_tools.training.datasets import FTW as _FTWTools
from torch.utils.data import Dataset

# matches the checkpoint's num_classes=3 (label_masks/semantic_3class)
CLASS_NAMES = {0: "background", 1: "field-interior", 2: "field-boundary"}

IGNORE_CLASS = 255  # our internal sentinel (matches Cityscapes' convention)

# FTW's own reserved "unknown" class -- ftw_tools' own training code labels
# its 3-class scheme ["background", "field", "boundary", "unknown"]
# (ftw_tools/training/trainers.py), and every 3-class FTW checkpoint we've
# found -- including the PRUE checkpoint used here -- is trained with
# ignore_index: 3 (see prue-unet-...-winargb_config.yaml). This is a
# dataset-wide standard, not a per-country quirk: unconfirmed/no-data area
# is labeled 3 rather than confidently labeled background (0). Remapped to
# IGNORE_CLASS for every country unconditionally, below.
FTW_UNKNOWN_CLASS = 3

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
    country: str  # e.g. "austria" -- the country subfolder this tile was loaded from
    acquisition_date: str | None = (
        None  # not present in this dataset's metadata (chips parquet has no date column)
    )


class FTWDataset(Dataset):
    """Loads FTW Sentinel-2 tiles and field-boundary masks for a country/split."""

    def __init__(
        self,
        root: Path,
        countries: list[str],
        split: str = "val",
        class_remap: dict[str, dict[int, int]] | None = None,
    ) -> None:
        """class_remap, if given, maps country -> {old_class_id: new_class_id},
        applied to that country's ground_truth on load, on top of the
        unconditional FTW_UNKNOWN_CLASS -> IGNORE_CLASS remap every country
        already gets (see FTW_UNKNOWN_CLASS above). Use this for a genuine
        per-country label-convention mismatch, not for the standard "unknown"
        class -- that's handled automatically."""
        self.root = Path(root)
        self.countries = [c.lower() for c in countries]
        self.class_remap = {k.lower(): v for k, v in (class_remap or {}).items()}
        self._dataset = _FTWTools(
            root=str(root),
            countries=self.countries,
            split=split,
            temporal_options="window_a_rgb",
            load_boundaries=True,
        )
        self._centroid_by_country_aoi_id = self._load_centroids()

    def _load_centroids(self) -> dict[tuple[str, str], tuple[float, float]]:
        # Tiles are small (a few hundred meters), so the midpoint of the
        # lon/lat bounding box is an adequate stand-in for a true centroid
        # and sidesteps geopandas' geographic-CRS centroid warning.
        # Keyed by (country, aoi_id), not aoi_id alone -- aoi_id is only
        # guaranteed unique within a single country's chips file, and with
        # multiple countries loaded a collision would silently overwrite.
        centroids: dict[tuple[str, str], tuple[float, float]] = {}
        for country in self.countries:
            chips_path = self.root / country / f"chips_{country}.parquet"
            chips_df = gpd.read_parquet(chips_path)
            bounds = chips_df.geometry.bounds
            lat = (bounds["miny"] + bounds["maxy"]) / 2
            lon = (bounds["minx"] + bounds["maxx"]) / 2
            for aoi_id, lat_val, lon_val in zip(
                chips_df["aoi_id"], lat, lon, strict=True
            ):
                centroids[(country, aoi_id)] = (lat_val, lon_val)
        return centroids

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> FTWTile:
        sample = self._dataset[index]
        window_a_path = Path(self._dataset.filenames[index]["window_a"])
        tile_id = window_a_path.stem
        # ftw_tools lays out files as {root}/{country}/s2_images/window_a/{aoi_id}.tif
        # -- the first path component after root is the country.
        country = window_a_path.relative_to(self._dataset.root).parts[0]

        image = sample["image"].numpy().transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
        ground_truth = sample["mask"].numpy().astype(np.uint8)
        ground_truth = remap_classes(ground_truth, {FTW_UNKNOWN_CLASS: IGNORE_CLASS})

        remap = self.class_remap.get(country)
        if remap:
            ground_truth = remap_classes(ground_truth, remap)

        lat, lon = self._centroid_by_country_aoi_id.get(
            (country, tile_id), (float("nan"), float("nan"))
        )
        return FTWTile(
            tile_id=tile_id,
            image=image,
            ground_truth=ground_truth,
            geography=f"{lat:.4f},{lon:.4f}",
            country=country,
        )


def remap_classes(ground_truth: np.ndarray, remap: dict[int, int]) -> np.ndarray:
    """Applies a {old_class_id: new_class_id} remap to a ground_truth array.
    Values not present in remap pass through unchanged."""
    lookup = np.arange(256, dtype=np.uint8)
    for old_id, new_id in remap.items():
        lookup[old_id] = new_id
    return lookup[ground_truth]


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
