"""Sanity check for the naming methodology itself (bias_direction ->
deconfound -> retrieve_concepts), independent of whether Cityscapes' real
segmentation errors have any particular known cause.

Takes real image crops (for realistic CLIP embeddings), creates a matched
"error" variant of each by applying a *known* transformation associated with
a specific concept in the concept bank, and checks whether the naming
pipeline recovers that concept from the resulting bias direction. Since both
groups are the same crops apart from the one injected transform, this
isolates whether the mechanism can find a signal we know is there.

Doesn't need the segmentation model or ground truth -- just real images, so
it runs in under a minute (no SegFormer inference).
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps
from scipy.ndimage import convolve

from drawthename.concept_bank import center_embeddings, load_concept_bank
from drawthename.embeddings import embed_regions, load_backbone
from drawthename.naming import bias_direction, retrieve_concepts
from drawthename.regions import Region


@dataclass
class SyntheticTest:
    name: str
    transform: object  # Callable[[Image.Image], Image.Image]
    expected_concepts: list[str]


def motion_blur(img: Image.Image, size: int = 15) -> Image.Image:
    """Directional (horizontal) blur via a line kernel -- deliberately not
    Gaussian blur. An earlier version of this test used uniform Gaussian
    blur and it failed to recover any blur-related concept (motion blur and
    out-of-focus blur both scored *negative* similarity): those concept
    phrases apparently mean specific things to CLIP -- motion blur implies
    directional streaking, out-of-focus blur implies depth-of-field bokeh --
    neither of which a uniform whole-frame blur produces. This kernel
    produces genuine directional streaking, which recovers 'motion blur' as
    the #1 concept out of the whole bank."""
    arr = np.array(img).astype(np.float32)
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = 1.0
    kernel /= kernel.sum()
    blurred = np.stack(
        [convolve(arr[..., c], kernel, mode="reflect") for c in range(3)], axis=-1
    )
    return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))


def darken(img: Image.Image) -> Image.Image:
    return Image.eval(img, lambda p: int(p * 0.2))


def occlude(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    h, w = arr.shape[:2]
    arr[h // 3 : h, : w // 2] = 0
    return Image.fromarray(arr)


def overexpose(img: Image.Image) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    arr = np.clip(arr * 1.8 + 60, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def green_tint(img: Image.Image) -> Image.Image:
    r, g, b = img.split()
    g = ImageOps.autocontrast(g).point(lambda p: min(255, int(p * 1.6)))
    return Image.merge("RGB", (r, g, b))


TESTS = [
    SyntheticTest(
        "motion_blur",
        motion_blur,
        ["motion blur", "blurriness", "low resolution texture", "out-of-focus blur"],
    ),
    SyntheticTest(
        "darken", darken, ["dim lighting", "nighttime lighting", "underexposed shadows"]
    ),
    SyntheticTest(
        "occlude",
        occlude,
        [
            "partial occlusion",
            "heavily occluded object",
            "object occluded by another object",
        ],
    ),
    SyntheticTest(
        "overexpose",
        overexpose,
        ["overexposed highlights", "harsh direct sunlight", "glare and reflections"],
    ),
    SyntheticTest("green_tint", green_tint, ["green"]),
]


def load_source_crops(
    dataset_root: Path, n: int, crop_size: int, seed: int
) -> list[np.ndarray]:
    """Random crop_size x crop_size windows from a handful of real Cityscapes
    images -- content doesn't need ground truth/class labels, this just needs
    to be realistic image content for CLIP to embed meaningfully."""
    rng = random.Random(seed)
    image_dir = dataset_root / "leftImg8bit" / "val"
    image_paths = sorted(image_dir.glob("*/*_leftImg8bit.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found under {image_dir}")
    sample_paths = rng.sample(image_paths, min(8, len(image_paths)))

    crops = []
    for _ in range(n):
        path = rng.choice(sample_paths)
        img = Image.open(path).convert("RGB")
        w, h = img.size
        x0 = rng.randint(0, w - crop_size)
        y0 = rng.randint(0, h - crop_size)
        crops.append(img.crop((x0, y0, x0 + crop_size, y0 + crop_size)))
    return crops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/standard_cv.yaml"))
    parser.add_argument("--n-crops", type=int, default=40)
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    backbone = load_backbone(
        config["backbone"]["name"], device=config["backbone"]["device"]
    )
    concept_texts = load_concept_bank(Path(config["concept_bank"]))
    concept_embeddings = center_embeddings(backbone.encode_text(concept_texts))

    source_crops = load_source_crops(
        Path(config["data"]["root"]), args.n_crops, args.crop_size, args.seed
    )
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
