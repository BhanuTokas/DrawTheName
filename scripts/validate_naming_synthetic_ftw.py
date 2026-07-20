"""FTW counterpart to validate_naming_synthetic.py: sanity check for the
naming methodology against ftw_concepts.txt specifically, since a concept
bank is only as good as its vocabulary -- this doesn't validate PRUE's real
errors, it validates that the *bank* has the right phrases for the pipeline
to find when a known signal is actually present.

Takes real FTW tile crops (for realistic CLIP embeddings of actual Sentinel-2
content), creates a matched "error" variant of each by applying a *known*
transformation associated with a specific FTW concept, and checks whether the
naming pipeline recovers that concept from the resulting bias direction.

Doesn't need PRUE or ground truth -- just real tile imagery, so it runs in
under a minute (no segmentation inference).
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

from drawthename.concept_bank import center_embeddings, load_concept_bank
from drawthename.data.ftw import FTWDataset, to_display_rgb
from drawthename.embeddings import embed_regions, load_backbone
from drawthename.naming import bias_direction, retrieve_concepts
from drawthename.regions import Region


@dataclass
class SyntheticTest:
    name: str
    transform: object  # Callable[[Image.Image], Image.Image]
    expected_concepts: list[str]


def cloud_overlay(img: Image.Image, alpha: float = 0.6) -> Image.Image:
    """Blends toward a light gray-white -- what cloud cover looks like from
    above, as opposed to darkening (which is what its *shadow* looks like)."""
    arr = np.array(img).astype(np.float32)
    cloud_color = np.array([230.0, 230.0, 235.0])
    blended = arr * (1 - alpha) + cloud_color * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def cloud_shadow(img: Image.Image, factor: float = 0.35) -> Image.Image:
    """Darkens only the top half -- a shadow *falling across part of* the
    tile, distinct from uniform low light."""
    arr = np.array(img).astype(np.float32)
    h = arr.shape[0]
    arr[: h // 2, :] *= factor
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def standing_water(img: Image.Image, coverage: float = 0.4) -> Image.Image:
    """Replaces the bottom band with a dark, blue-shifted patch."""
    arr = np.array(img).astype(np.float32)
    h = arr.shape[0]
    patch_h = int(h * coverage)
    water_color = np.array([40.0, 60.0, 90.0])
    arr[h - patch_h :, :] = arr[h - patch_h :, :] * 0.3 + water_color * 0.7
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def tillage_stripes(
    img: Image.Image, band_width: int = 6, contrast: float = 50
) -> Image.Image:
    """Shifts toward bare-soil brown and adds a sharp-edged (square-wave, not
    sinusoidal) horizontal banding -- ridge/furrow lines in freshly plowed
    soil. Two earlier versions of this test failed: a luminance ripple alone
    over the crop's original (likely still green) color read to CLIP as
    'image striping artifact' (a sensor defect), and even after adding the
    soil color shift, a *smooth sine* ripple still didn't read as furrows --
    apparently CLIP wants the sharper ridge-like edge a real furrow casts,
    not a soft gradient."""
    arr = np.array(img).astype(np.float32)
    soil_color = np.array([120.0, 90.0, 60.0])
    arr = arr * 0.4 + soil_color * 0.6
    h = arr.shape[0]
    band_idx = np.arange(h) // band_width
    banding = np.where(band_idx % 2 == 0, contrast, -contrast).reshape(-1, 1, 1)
    return Image.fromarray(np.clip(arr + banding, 0, 255).astype(np.uint8))


def golden_stubble(img: Image.Image) -> Image.Image:
    """Shifts color toward the golden-brown of cut, dry crop residue: boosts
    red, mutes blue, leaves green roughly alone."""
    arr = np.array(img).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    out = np.stack(
        [
            np.clip(r * 1.3 + 20, 0, 255),
            np.clip(g * 1.05, 0, 255),
            np.clip(b * 0.5, 0, 255),
        ],
        axis=-1,
    )
    return Image.fromarray(out.astype(np.uint8))


def green_boost(img: Image.Image) -> Image.Image:
    r, g, b = img.split()
    g = ImageOps.autocontrast(g).point(lambda p: min(255, int(p * 1.5)))
    return Image.merge("RGB", (r, g, b))


TESTS = [
    SyntheticTest(
        "cloud_overlay",
        cloud_overlay,
        ["cloud cover", "thick cloud cover", "haze over field", "atmospheric haze"],
    ),
    SyntheticTest(
        "cloud_shadow",
        cloud_shadow,
        [
            "cloud shadow",
            "cloud shadow across field",
            "dark shadow area",
            "patchy cloud shadow pattern",
        ],
    ),
    SyntheticTest(
        "standing_water",
        standing_water,
        [
            "waterlogged field",
            "standing water in field",
            "flooded field section",
            "open water",
        ],
    ),
    SyntheticTest(
        "tillage_stripes",
        tillage_stripes,
        [
            "tilled soil pattern",
            "plowed field furrows",
            "tillage furrow lines",
            "cultivated soil texture",
        ],
    ),
    SyntheticTest(
        "golden_stubble",
        golden_stubble,
        [
            "post-harvest stubble",
            "grain stubble field",
            "golden wheat",
            "straw residue field",
        ],
    ),
    SyntheticTest(
        "green_boost",
        green_boost,
        [
            "green vegetation",
            "green-up period",
            "dense vegetation",
            "summer crop canopy",
        ],
    ),
]


def load_source_crops(
    dataset: FTWDataset, n: int, crop_size: int, seed: int
) -> list[Image.Image]:
    """Random crop_size x crop_size windows from a handful of real FTW tiles
    (converted to display RGB, same as the pipeline's CLIP-facing crops) --
    content doesn't need ground truth/class labels, this just needs to be
    realistic Sentinel-2 imagery for CLIP to embed meaningfully."""
    rng = random.Random(seed)
    tile_indices = rng.sample(range(len(dataset)), min(8, len(dataset)))
    tile_images = [to_display_rgb(dataset[i].image) for i in tile_indices]

    crops = []
    for _ in range(n):
        tile = rng.choice(tile_images)
        h, w = tile.shape[:2]
        x0 = rng.randint(0, w - crop_size)
        y0 = rng.randint(0, h - crop_size)
        crops.append(Image.fromarray(tile[y0 : y0 + crop_size, x0 : x0 + crop_size]))
    return crops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/ftw.yaml"))
    parser.add_argument("--n-crops", type=int, default=40)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    backbone = load_backbone(
        config["backbone"]["name"], device=config["backbone"]["device"]
    )
    concept_texts = load_concept_bank(Path(config["concept_bank"]))
    concept_embeddings = center_embeddings(backbone.encode_text(concept_texts))

    dataset = FTWDataset(
        Path(config["data"]["root"]),
        config["data"]["countries"],
        config["data"]["split"],
    )
    source_crops = load_source_crops(dataset, args.n_crops, args.crop_size, args.seed)
    correct_regions = [
        Region(
            image_id="synthetic",
            region_id=i,
            crop=np.array(c),
            label="correct",
            class_id=0,
            pixel_error_rate=0.0,
        )
        for i, c in enumerate(source_crops)
    ]
    correct_embeddings = embed_regions(correct_regions, backbone)

    top_k = config["naming"]["top_k_concepts"]
    n_passed = 0
    print(f"{len(TESTS)} synthetic tests, {args.n_crops} crops each, top_k={top_k}\n")
    for test in TESTS:
        error_crops = [test.transform(c) for c in source_crops]
        error_regions = [
            Region(
                image_id="synthetic",
                region_id=i,
                crop=np.array(c),
                label="error",
                class_id=0,
                pixel_error_rate=1.0,
            )
            for i, c in enumerate(error_crops)
        ]
        error_embeddings = embed_regions(error_regions, backbone)

        direction = bias_direction(error_embeddings, correct_embeddings)
        top_concepts = retrieve_concepts(
            direction, concept_texts, concept_embeddings, top_k=top_k
        )

        hit = next((c for c in test.expected_concepts if c in top_concepts), None)
        n_passed += hit is not None
        status = f"PASS (found {hit!r})" if hit else "FAIL"
        print(f"[{status}] {test.name}: expected one of {test.expected_concepts}")
        print(f"    top-{top_k}: {top_concepts}\n")

    print(f"recovered {n_passed}/{len(TESTS)} known injected concepts")


if __name__ == "__main__":
    main()
