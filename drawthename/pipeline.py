"""End-to-end orchestration for both pipeline modes (spec section 3)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap
from PIL import Image
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from drawthename.clustering import cluster_embeddings, select_k_by_silhouette
from drawthename.concept_bank import (
    center_embeddings,
    embed_concept_bank,
    load_concept_bank,
)
from drawthename.data.cityscapes import TRAIN_ID_NAMES, CityscapesDataset
from drawthename.data.ftw import CLASS_NAMES as FTW_CLASS_NAMES
from drawthename.data.ftw import FTWDataset, to_display_rgb
from drawthename.data.mapillary_vistas import (
    DEFAULT_CONTINENT_LABELS_PATH,
    SHARED_CLASSES,
    MapillaryVistasDataset,
)
from drawthename.embeddings import ClipLikeBackbone, embed_regions, load_backbone
from drawthename.ftw_compare import (
    compare_concept_sets,
    correct_country_pair_directions,
    domain_shift_candidates_per_pair,
    flag_confound,
    inter_country_direction,
    inter_country_pair_directions,
    inter_tile_direction,
    intra_country_direction,
    intra_tile_direction,
)
from drawthename.naming import (
    GlobalErrorMode,
    NamedDirection,
    bias_direction,
    bootstrap_sign_stability,
    deconfound,
    grouped_bias_direction,
    grouped_bootstrap_sign_stability,
    retrieve_concepts,
)
from drawthename.regions import (
    IGNORE_CLASS,
    Region,
    compute_error_mask,
    extract_regions,
)
from drawthename.segmentation_model import PRUEModel, SegmentationModel

# Vistas Mode analyzes only the geo-bias paper's 7 classes shared between
# Cityscapes and Vistas (person/rider/car/truck/bus/motorcycle/bicycle).
VISTAS_CLASS_NAMES = {class_id: name for name, class_id in SHARED_CLASSES.items()}

# Placeholder crop for regions whose crop was dropped after embedding.
_DROPPED_CROP = np.zeros((0, 0, 3), dtype=np.uint8)


def run_standard_cv_pipeline(
    config: dict[str, Any], output_dir: Path, image_indices: list[int] | None = None
) -> None:
    """Phase 1: Cityscapes inference -> region extraction -> embeddings ->
    clustering -> bias naming. Writes embeddings.npz, clusters.json,
    bias_directions.json, pixel_accuracy.json, summary.md, plots/ to
    output_dir.

    image_indices, if given, restricts the run to those dataset indices
    (e.g. for splitting the val set into two halves to check reproducibility)
    instead of the first config["data"]["limit"] images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CityscapesDataset(Path(config["data"]["root"]), config["data"]["split"])
    segmentation_model = SegmentationModel(
        checkpoint=config["segmentation_model"]["checkpoint"],
        device=config["backbone"]["device"],
    )
    backbone = load_backbone(
        config["backbone"]["name"], device=config["backbone"]["device"]
    )

    concept_texts = load_concept_bank(Path(config["concept_bank"]))
    concept_embeddings = center_embeddings(embed_concept_bank(concept_texts, backbone))

    regions, pixel_counts = _extract_all_regions(
        dataset,
        segmentation_model,
        config["regions"],
        limit=config["data"].get("limit"),
        indices=image_indices,
    )
    embeddings = embed_regions(regions, backbone)

    global_error_mode = _compute_global_error_mode(
        regions, embeddings, concept_texts, concept_embeddings, config["naming"]
    )

    named_directions, cluster_id_by_region_idx, silhouette_by_class = (
        _name_bias_directions(
            regions,
            embeddings,
            concept_texts,
            concept_embeddings,
            global_error_mode,
            clustering_cfg=config["clustering"],
            naming_cfg=config["naming"],
            output_dir=output_dir,
            class_names=TRAIN_ID_NAMES,
            plot_max_points=config.get("plots", {}).get("max_points_per_class", 3000),
        )
    )

    _write_embeddings(
        regions, embeddings, cluster_id_by_region_idx, output_dir / "embeddings.npz"
    )
    _write_clusters(named_directions, silhouette_by_class, output_dir / "clusters.json")
    _write_bias_directions(
        named_directions, global_error_mode, output_dir / "bias_directions.json"
    )
    _write_pixel_accuracy(pixel_counts, output_dir / "pixel_accuracy.json")
    _write_summary(
        named_directions,
        global_error_mode,
        config["naming"]["stability_threshold"],
        output_dir / "summary.md",
        class_names=TRAIN_ID_NAMES,
        residual_ratio_threshold=config["naming"].get("residual_ratio_threshold", 0.1),
    )


def run_ftw_pipeline(config: dict[str, Any], output_dir: Path) -> None:
    """Phase 2: PRUE inference on FTW tiles -> tile classification ->
    sub-region extraction -> embeddings -> clustering -> intra/inter-tile
    comparison -> bias naming. Writes the same output set as Standard CV Mode,
    plus intra/inter-tile flags in bias_directions.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = FTWDataset(
        Path(config["data"]["root"]),
        config["data"]["countries"],
        config["data"]["split"],
        class_remap=config["data"].get("class_remap"),
    )
    segmentation_model = PRUEModel(
        checkpoint=config["prue"]["checkpoint"], device=config["backbone"]["device"]
    )
    backbone = load_backbone(
        config["backbone"]["name"], device=config["backbone"]["device"]
    )

    concept_texts = load_concept_bank(Path(config["concept_bank"]))
    concept_embeddings = center_embeddings(embed_concept_bank(concept_texts, backbone))

    regions, pixel_counts = _extract_all_ftw_regions(
        dataset,
        segmentation_model,
        config["regions"],
        limit=config["data"].get("limit"),
    )
    embeddings = embed_regions(regions, backbone)

    global_error_mode = _compute_global_error_mode(
        regions, embeddings, concept_texts, concept_embeddings, config["naming"]
    )

    named_directions, cluster_id_by_region_idx, silhouette_by_class = (
        _name_bias_directions(
            regions,
            embeddings,
            concept_texts,
            concept_embeddings,
            global_error_mode,
            clustering_cfg=config["clustering"],
            naming_cfg=config["naming"],
            output_dir=output_dir,
            class_names=FTW_CLASS_NAMES,
            plot_max_points=config.get("plots", {}).get("max_points_per_class", 3000),
            check_tile_confounds=True,
            intra_inter_cos_threshold=config["ftw_compare"][
                "intra_inter_divergence_threshold"
            ],
            concept_comparison_top_k=config["ftw_compare"].get(
                "concept_comparison_top_k", 20
            ),
            min_pair_region_count=config["ftw_compare"].get(
                "min_pair_region_count", 10
            ),
        )
    )

    _write_embeddings(
        regions, embeddings, cluster_id_by_region_idx, output_dir / "embeddings.npz"
    )
    _write_clusters(named_directions, silhouette_by_class, output_dir / "clusters.json")
    _write_bias_directions(
        named_directions, global_error_mode, output_dir / "bias_directions.json"
    )
    _write_pixel_accuracy(pixel_counts, output_dir / "pixel_accuracy.json")
    _write_summary(
        named_directions,
        global_error_mode,
        config["naming"]["stability_threshold"],
        output_dir / "summary.md",
        class_names=FTW_CLASS_NAMES,
        residual_ratio_threshold=config["naming"].get("residual_ratio_threshold", 0.1),
    )


def run_vistas_pipeline(config: dict[str, Any], output_dir: Path) -> None:
    """Vistas Mode: Standard CV Mode's SegFormer inference + bias naming on
    Mapillary Vistas, restricted to the geo-bias paper's 7 shared classes,
    with FTW Mode's confound checks re-scoped: each region's continent goes
    in Region.country (intra/inter-country -> intra/inter-continent), and
    each image plays an FTW tile's role, since its regions share camera,
    weather and time of day (intra/inter-tile -> intra/inter-image). Tests
    whether the pipeline, unprompted, names the class confusions the
    geo-bias paper (arXiv:2412.11061) and our own geo_bias_pipeline
    replication measure. Writes the same output set as FTW Mode."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MapillaryVistasDataset(
        root=Path(config["data"]["root"]),
        continent_labels_path=Path(
            config["data"].get("continent_labels", DEFAULT_CONTINENT_LABELS_PATH)
        ),
    )
    segmentation_model = SegmentationModel(
        checkpoint=config["segmentation_model"]["checkpoint"],
        device=config["backbone"]["device"],
    )
    backbone = load_backbone(
        config["backbone"]["name"], device=config["backbone"]["device"]
    )

    concept_texts = load_concept_bank(Path(config["concept_bank"]))
    concept_embeddings = center_embeddings(embed_concept_bank(concept_texts, backbone))

    regions, embeddings, pixel_counts = _extract_and_embed_vistas_regions(
        dataset,
        segmentation_model,
        backbone,
        config["regions"],
        limit=config["data"].get("limit"),
        max_side=config["data"].get("max_side"),
    )

    global_error_mode = _compute_global_error_mode(
        regions, embeddings, concept_texts, concept_embeddings, config["naming"]
    )

    named_directions, cluster_id_by_region_idx, silhouette_by_class = (
        _name_bias_directions(
            regions,
            embeddings,
            concept_texts,
            concept_embeddings,
            global_error_mode,
            clustering_cfg=config["clustering"],
            naming_cfg=config["naming"],
            output_dir=output_dir,
            class_names=VISTAS_CLASS_NAMES,
            plot_max_points=config.get("plots", {}).get("max_points_per_class", 3000),
            check_tile_confounds=True,
            intra_inter_cos_threshold=config["geo_compare"][
                "intra_inter_divergence_threshold"
            ],
            concept_comparison_top_k=config["geo_compare"].get(
                "concept_comparison_top_k", 20
            ),
            min_pair_region_count=config["geo_compare"].get(
                "min_pair_region_count", 10
            ),
        )
    )

    _write_embeddings(
        regions, embeddings, cluster_id_by_region_idx, output_dir / "embeddings.npz"
    )
    _write_clusters(named_directions, silhouette_by_class, output_dir / "clusters.json")
    _write_bias_directions(
        named_directions, global_error_mode, output_dir / "bias_directions.json"
    )
    _write_pixel_accuracy(pixel_counts, output_dir / "pixel_accuracy.json")
    _write_summary(
        named_directions,
        global_error_mode,
        config["naming"]["stability_threshold"],
        output_dir / "summary.md",
        class_names=VISTAS_CLASS_NAMES,
        residual_ratio_threshold=config["naming"].get("residual_ratio_threshold", 0.1),
    )


def _downscale(
    image: np.ndarray, ground_truth: np.ndarray, max_side: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Shrinks image (bilinear) and ground_truth (nearest, so class ids stay
    valid) so their longer side is at most max_side; no-op if already within
    it or max_side is None."""
    h, w = ground_truth.shape
    if max_side is None or max(h, w) <= max_side:
        return image, ground_truth
    scale = max_side / max(h, w)
    size = (round(w * scale), round(h * scale))
    return (
        np.array(Image.fromarray(image).resize(size, Image.BILINEAR)),
        np.array(Image.fromarray(ground_truth).resize(size, Image.NEAREST)),
    )


def _extract_and_embed_vistas_regions(
    dataset: MapillaryVistasDataset,
    segmentation_model: SegmentationModel,
    backbone: ClipLikeBackbone,
    regions_cfg: dict[str, Any],
    limit: int | None = None,
    max_side: int | None = None,
) -> tuple[list[Region], np.ndarray, dict[int, tuple[int, int]]]:
    """Vistas-mode counterpart to _extract_all_regions, returning embeddings
    too. Ground truth outside VISTAS_CLASS_NAMES is set to IGNORE_CLASS
    before extraction (predictions are untouched, so a car predicted as
    "road" still counts as a car error). Each image's regions are embedded
    straight away and their crops dropped: crops are views into the full
    image, so holding every crop until one embed_regions call at the end
    (as the other modes do) would keep all ~11,300 images in memory."""
    shared_ids = np.array(sorted(VISTAS_CLASS_NAMES))
    all_regions: list[Region] = []
    all_embeddings: list[np.ndarray] = []
    pixel_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    num_samples = min(len(dataset), limit) if limit else len(dataset)
    for index in tqdm(
        range(num_samples), desc="Running inference + extracting regions"
    ):
        sample = dataset[index]
        image, ground_truth = _downscale(sample.image, sample.ground_truth, max_side)
        ground_truth = np.where(
            np.isin(ground_truth, shared_ids), ground_truth, IGNORE_CLASS
        ).astype(np.uint8)
        prediction = segmentation_model.predict(image)
        error_mask = compute_error_mask(prediction, ground_truth)

        for class_id in np.unique(ground_truth):
            if class_id == IGNORE_CLASS:
                continue
            class_pixel_mask = ground_truth == class_id
            counts = pixel_counts[int(class_id)]
            counts[0] += int(error_mask[class_pixel_mask].sum())
            counts[1] += int(class_pixel_mask.sum())

        regions = extract_regions(
            image=image,
            error_mask=error_mask,
            ground_truth=ground_truth,
            image_id=sample.image_id,
            min_area_px=regions_cfg["min_area_px"],
            pad_px_min=regions_cfg["pad_px_min"],
            pad_frac=regions_cfg["pad_frac"],
            error_rate_threshold=regions_cfg["error_rate_threshold"],
            tile_id=sample.image_id,
            country=sample.continent,
            subdivision_size=regions_cfg.get("subdivision_size"),
        )
        if not regions:
            continue
        all_embeddings.append(embed_regions(regions, backbone))
        for region in regions:
            region.crop = _DROPPED_CROP
        all_regions.extend(regions)

    embeddings = (
        np.concatenate(all_embeddings, axis=0)
        if all_embeddings
        else np.zeros((0, 0), dtype=np.float32)
    )
    return (
        all_regions,
        embeddings,
        {class_id: tuple(counts) for class_id, counts in pixel_counts.items()},
    )


def _extract_all_ftw_regions(
    dataset: FTWDataset,
    segmentation_model: PRUEModel,
    regions_cfg: dict[str, Any],
    limit: int | None = None,
) -> tuple[list[Region], dict[int, tuple[int, int]]]:
    """FTW-mode counterpart to _extract_all_regions: same per-class pixel
    counting and region extraction, but against FTWTile's raw-reflectance
    image (fed to PRUEModel as-is) and a separately stretched uint8 RGB
    version (fed to CLIP via extract_regions' crops -- encode_image's PIL
    conversion needs a standard 0-255 image, not raw reflectance). Each tile
    is its own image_id/tile_id: FTW tiles don't share a "same source image"
    grouping the way Cityscapes region subdivisions do."""
    all_regions: list[Region] = []
    pixel_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    num_samples = min(len(dataset), limit) if limit else len(dataset)
    for index in tqdm(
        range(num_samples), desc="Running inference + extracting regions"
    ):
        tile = dataset[index]
        prediction = segmentation_model.predict(tile.image)
        error_mask = compute_error_mask(prediction, tile.ground_truth)

        for class_id in np.unique(tile.ground_truth):
            if class_id == IGNORE_CLASS:
                continue
            class_pixel_mask = tile.ground_truth == class_id
            counts = pixel_counts[int(class_id)]
            counts[0] += int(error_mask[class_pixel_mask].sum())
            counts[1] += int(class_pixel_mask.sum())

        display_image = to_display_rgb(tile.image)
        all_regions.extend(
            extract_regions(
                image=display_image,
                error_mask=error_mask,
                ground_truth=tile.ground_truth,
                image_id=tile.tile_id,
                min_area_px=regions_cfg["min_area_px"],
                pad_px_min=regions_cfg["pad_px_min"],
                pad_frac=regions_cfg["pad_frac"],
                error_rate_threshold=regions_cfg["error_rate_threshold"],
                tile_id=tile.tile_id,
                country=tile.country,
                subdivision_size=regions_cfg.get("subdivision_size"),
            )
        )
    return all_regions, {
        class_id: tuple(counts) for class_id, counts in pixel_counts.items()
    }


def _extract_all_regions(
    dataset: CityscapesDataset,
    segmentation_model: SegmentationModel,
    regions_cfg: dict[str, Any],
    limit: int | None = None,
    indices: list[int] | None = None,
) -> tuple[list[Region], dict[int, tuple[int, int]]]:
    """Returns the extracted regions, plus per-class (error_pixels,
    total_pixels) pixel-level counts -- a class's region-level "error rate"
    (fraction of its regions labeled error) and its pixel-level error rate
    can differ hugely, since region-counting weighs a 64px^2 boundary sliver
    the same as a 40,000px^2 well-segmented blob.

    indices, if given, takes priority over limit and restricts the run to
    exactly those dataset indices."""
    all_regions: list[Region] = []
    pixel_counts: dict[int, list[int]] = defaultdict(
        lambda: [0, 0]
    )  # class_id -> [error_px, total_px]
    if indices is None:
        num_samples = min(len(dataset), limit) if limit else len(dataset)
        indices = list(range(num_samples))
    for index in tqdm(indices, desc="Running inference + extracting regions"):
        sample = dataset[index]
        prediction = segmentation_model.predict(sample.image)
        error_mask = compute_error_mask(prediction, sample.ground_truth)

        for class_id in np.unique(sample.ground_truth):
            if class_id == IGNORE_CLASS:
                continue
            class_pixel_mask = sample.ground_truth == class_id
            counts = pixel_counts[int(class_id)]
            counts[0] += int(error_mask[class_pixel_mask].sum())
            counts[1] += int(class_pixel_mask.sum())

        all_regions.extend(
            extract_regions(
                image=sample.image,
                error_mask=error_mask,
                ground_truth=sample.ground_truth,
                image_id=sample.image_id,
                min_area_px=regions_cfg["min_area_px"],
                pad_px_min=regions_cfg["pad_px_min"],
                pad_frac=regions_cfg["pad_frac"],
                error_rate_threshold=regions_cfg["error_rate_threshold"],
                subdivision_size=regions_cfg.get("subdivision_size"),
            )
        )
    return all_regions, {
        class_id: tuple(counts) for class_id, counts in pixel_counts.items()
    }


def _compute_global_error_mode(
    regions: list[Region],
    embeddings: np.ndarray,
    concept_texts: list[str],
    concept_embeddings: np.ndarray,
    naming_cfg: dict[str, Any],
) -> GlobalErrorMode:
    """The direction shared across nearly every class's bias vector (e.g.
    "errors tend to be small/blurry/oddly-cropped regardless of class") --
    pooling every class's error/correct regions together, rather than
    averaging the per-cluster vectors, so it isn't skewed by classes that
    happened to get more clusters.

    With naming_cfg["global_error_mode_per_country"], it is instead computed
    within each Region.country and averaged with equal weight per country:
    when some countries have a higher error share (Vistas: ~50% in Africa vs.
    ~40% in Europe), the pooled direction partly encodes "looks like those
    countries", and projecting it out of every cluster would then strip real
    geographic signal along with the shared confound."""
    error_idx = [i for i, r in enumerate(regions) if r.label == "error"]
    correct_idx = [i for i, r in enumerate(regions) if r.label == "correct"]
    if not error_idx or not correct_idx:
        raise ValueError(
            f"Can't compute a global error mode: {len(error_idx)} error region(s) and "
            f"{len(correct_idx)} correct region(s) across the whole run. Need at least "
            "one of each, or bias_direction silently returns NaN. Check error_rate_threshold "
            "and the dataset/limit being used (this is most likely on a tiny sanity-check run)."
        )
    if naming_cfg.get("global_error_mode_per_country"):
        error_groups, correct_groups = _group_error_correct_by_country(
            regions, embeddings
        )
        direction = grouped_bias_direction(error_groups, correct_groups)
        stability = grouped_bootstrap_sign_stability(
            error_groups,
            correct_groups,
            n_resamples=naming_cfg["bootstrap_resamples"],
            cosine_threshold=naming_cfg["cosine_threshold"],
        )
    else:
        direction = bias_direction(embeddings[error_idx], embeddings[correct_idx])
        stability = bootstrap_sign_stability(
            embeddings[error_idx],
            embeddings[correct_idx],
            n_resamples=naming_cfg["bootstrap_resamples"],
            cosine_threshold=naming_cfg["cosine_threshold"],
        )
    concepts = retrieve_concepts(
        direction, concept_texts, concept_embeddings, top_k=naming_cfg["top_k_concepts"]
    )
    return GlobalErrorMode(
        bias_vector=direction, concepts=concepts, stability=stability
    )


def _group_error_correct_by_country(
    regions: list[Region], embeddings: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Parallel per-country (error, correct) embedding lists, keeping only
    countries with at least one region of each label."""
    by_country: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"error": [], "correct": []}
    )
    for i, region in enumerate(regions):
        if region.country is None:
            raise ValueError(
                "global_error_mode_per_country needs Region.country on every "
                f"region, but region {i} ({region.image_id}) has none"
            )
        by_country[region.country][region.label].append(i)
    countries = sorted(
        c for c, idx in by_country.items() if idx["error"] and idx["correct"]
    )
    if not countries:
        raise ValueError(
            "global_error_mode_per_country: no country has both error and "
            "correct regions"
        )
    return (
        [embeddings[by_country[c]["error"]] for c in countries],
        [embeddings[by_country[c]["correct"]] for c in countries],
    )


def _name_bias_directions(
    regions: list[Region],
    embeddings: np.ndarray,
    concept_texts: list[str],
    concept_embeddings: np.ndarray,
    global_error_mode: GlobalErrorMode,
    clustering_cfg: dict[str, Any],
    naming_cfg: dict[str, Any],
    output_dir: Path,
    class_names: dict[int, str],
    plot_max_points: int = 3000,
    check_tile_confounds: bool = False,
    intra_inter_cos_threshold: float = 0.5,
    concept_comparison_top_k: int = 20,
    min_pair_region_count: int = 1,  # deliberately permissive here for direct/test callers; run_ftw_pipeline always passes the stricter config-driven default (10) documented in configs/ftw.yaml.example
) -> tuple[list[NamedDirection], dict[int, int], dict[int, float | None]]:
    """Returns (named_directions, cluster_id_by_region_idx, silhouette_by_class).
    cluster_id_by_region_idx maps a region's index in `regions`/`embeddings` to
    its cluster_id (only error regions in classes that got clustered are
    present); silhouette_by_class is the silhouette score for the k chosen
    per class (None when k=1, since silhouette isn't defined for one cluster)."""
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, region in enumerate(regions):
        by_class[region.class_id].append(i)

    named_directions: list[NamedDirection] = []
    cluster_id_by_region_idx: dict[int, int] = {}
    silhouette_by_class: dict[int, float | None] = {}
    for class_id, indices in by_class.items():
        class_regions = [regions[i] for i in indices]
        class_embeddings = embeddings[indices]

        error_idx = [i for i, r in enumerate(class_regions) if r.label == "error"]
        correct_idx = [i for i, r in enumerate(class_regions) if r.label == "correct"]
        if len(error_idx) < clustering_cfg["k_min"] or len(correct_idx) < 1:
            continue

        error_embeddings = class_embeddings[error_idx]
        correct_embeddings = class_embeddings[correct_idx]
        correct_regions = [class_regions[i] for i in correct_idx]

        k = select_k_by_silhouette(
            error_embeddings, clustering_cfg["k_min"], clustering_cfg["k_max"]
        )
        cluster_labels = (
            cluster_embeddings(error_embeddings, k)
            if k > 1
            else np.zeros(len(error_embeddings), dtype=int)
        )
        silhouette_by_class[class_id] = (
            float(silhouette_score(error_embeddings, cluster_labels)) if k > 1 else None
        )
        for local_idx, cluster_id in zip(error_idx, cluster_labels, strict=True):
            cluster_id_by_region_idx[indices[local_idx]] = int(cluster_id)

        _plot_class_embeddings(
            class_id,
            error_embeddings,
            correct_embeddings,
            cluster_labels,
            output_dir,
            class_names,
            plot_max_points,
        )

        for cluster_id in range(k):
            cluster_mask = cluster_labels == cluster_id
            cluster_embeds = error_embeddings[cluster_mask]
            if len(cluster_embeds) == 0:
                continue
            direction = bias_direction(cluster_embeds, correct_embeddings)
            stability = bootstrap_sign_stability(
                cluster_embeds,
                correct_embeddings,
                n_resamples=naming_cfg["bootstrap_resamples"],
                cosine_threshold=naming_cfg["cosine_threshold"],
            )
            deconfounded = deconfound(direction, global_error_mode.bias_vector)
            residual_ratio = float(
                np.linalg.norm(deconfounded) / (np.linalg.norm(direction) + 1e-12)
            )
            concepts = retrieve_concepts(
                deconfounded,
                concept_texts,
                concept_embeddings,
                top_k=naming_cfg["top_k_concepts"],
            )

            intra_inter_flag = None
            country_confound_flag = None
            concept_comparison = None
            country_pair_domain_shifts = None
            if check_tile_confounds:
                # cluster_mask and error_idx are parallel arrays, both
                # length len(error_embeddings) (cluster_labels was computed
                # directly on error_embeddings above) -- zip pairs each
                # cluster_mask entry with the error_idx it corresponds to,
                # so this recovers exactly the Region objects in this cluster.
                cluster_regions = [
                    class_regions[i]
                    for local_idx, i in zip(cluster_mask, error_idx, strict=True)
                    if local_idx
                ]
                intra_tile = intra_tile_direction(
                    cluster_regions, cluster_embeds, correct_regions, correct_embeddings
                )
                if intra_tile is None:
                    intra_inter_flag = (
                        "insufficient same-tile error+correct overlap to check"
                    )
                else:
                    inter_tile = inter_tile_direction(
                        cluster_regions,
                        cluster_embeds,
                        correct_regions,
                        correct_embeddings,
                    )
                    confound = flag_confound(
                        intra_tile, inter_tile, threshold=intra_inter_cos_threshold
                    )
                    intra_inter_flag = "confound flagged" if confound else "consistent"

                intra_country = intra_country_direction(
                    cluster_regions, cluster_embeds, correct_regions, correct_embeddings
                )
                inter_country = inter_country_direction(
                    cluster_regions, cluster_embeds, correct_regions, correct_embeddings
                )
                if intra_country is None or inter_country is None:
                    country_confound_flag = (
                        "insufficient multi-country coverage to check"
                    )
                else:
                    confound = flag_confound(
                        intra_country,
                        inter_country,
                        threshold=intra_inter_cos_threshold,
                    )
                    country_confound_flag = (
                        "confound flagged" if confound else "consistent"
                    )

                if intra_tile is not None and intra_country is not None:
                    intra_tile_concepts = retrieve_concepts(
                        deconfound(intra_tile, global_error_mode.bias_vector),
                        concept_texts,
                        concept_embeddings,
                        top_k=concept_comparison_top_k,
                    )
                    intra_country_concepts = retrieve_concepts(
                        deconfound(intra_country, global_error_mode.bias_vector),
                        concept_texts,
                        concept_embeddings,
                        top_k=concept_comparison_top_k,
                    )

                    if inter_country is not None:
                        concept_comparison = compare_concept_sets(
                            intra_tile_concepts,
                            intra_country_concepts,
                            retrieve_concepts(
                                deconfound(
                                    inter_country, global_error_mode.bias_vector
                                ),
                                concept_texts,
                                concept_embeddings,
                                top_k=concept_comparison_top_k,
                            ),
                        )

                    pair_directions = inter_country_pair_directions(
                        cluster_regions,
                        cluster_embeds,
                        correct_regions,
                        correct_embeddings,
                        min_region_count=min_pair_region_count,
                    )
                    pair_concepts = {
                        pair: retrieve_concepts(
                            deconfound(pair_direction, global_error_mode.bias_vector),
                            concept_texts,
                            concept_embeddings,
                            top_k=concept_comparison_top_k,
                        )
                        for pair, pair_direction in pair_directions.items()
                    }
                    pair_domain_shifts = domain_shift_candidates_per_pair(
                        intra_tile_concepts, intra_country_concepts, pair_concepts
                    )
                    baseline_directions = correct_country_pair_directions(
                        correct_regions,
                        correct_embeddings,
                        min_region_count=min_pair_region_count,
                    )
                    baseline_concepts = {
                        pair: retrieve_concepts(
                            deconfound(
                                baseline_directions[pair],
                                global_error_mode.bias_vector,
                            ),
                            concept_texts,
                            concept_embeddings,
                            top_k=concept_comparison_top_k,
                        )
                        for pair in pair_domain_shifts
                        if pair in baseline_directions
                    }
                    country_pair_domain_shifts = []
                    for pair, candidates in pair_domain_shifts.items():
                        # baseline is None when pair[0] has too few correct
                        # regions for a scenery baseline
                        baseline = baseline_concepts.get(pair)
                        country_pair_domain_shifts.append(
                            {
                                "error_country": pair[0],
                                "correct_country": pair[1],
                                "domain_shift_candidates": candidates,
                                "scenery_baseline_concepts": baseline,
                                "error_specific_candidates": (
                                    None
                                    if baseline is None
                                    else sorted(set(candidates) - set(baseline))
                                ),
                            }
                        )

            named_directions.append(
                NamedDirection(
                    class_id=class_id,
                    cluster_id=cluster_id,
                    bias_vector=direction,
                    concepts=concepts,
                    stability=stability,
                    residual_ratio=residual_ratio,
                    intra_inter_flag=intra_inter_flag,
                    country_confound_flag=country_confound_flag,
                    concept_comparison=concept_comparison,
                    country_pair_domain_shifts=country_pair_domain_shifts,
                )
            )
    return named_directions, cluster_id_by_region_idx, silhouette_by_class


def _stratified_subsample(
    groups: list[np.ndarray], max_total: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Subsamples each group proportionally to its size, so a plot's point
    budget doesn't let a large group (e.g. a common cluster, or the correct
    pool) drown out smaller ones. Every non-empty group keeps at least 1
    point so it stays visible."""
    sizes = np.array([len(g) for g in groups])
    total = int(sizes.sum())
    if total <= max_total:
        return groups

    target = np.minimum(
        np.maximum(1, np.round(max_total * sizes / total).astype(int)), sizes
    )
    return [
        group
        if len(group) <= n
        else group[rng.choice(len(group), size=n, replace=False)]
        for group, n in zip(groups, target)
    ]


def _plot_class_embeddings(
    class_id: int,
    error_embeddings: np.ndarray,
    correct_embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    output_dir: Path,
    class_names: dict[int, str],
    max_points: int | None = 3000,
) -> None:
    num_clusters = int(cluster_labels.max()) + 1 if len(cluster_labels) else 0
    groups = [error_embeddings[cluster_labels == c] for c in range(num_clusters)] + [
        correct_embeddings
    ]
    if max_points:
        groups = _stratified_subsample(groups, max_points, np.random.default_rng(0))
    error_groups, correct_sample = groups[:-1], groups[-1]

    error_sample = (
        np.concatenate(error_groups, axis=0)
        if error_groups
        else np.zeros((0, correct_embeddings.shape[1]))
    )
    cluster_sample_labels = (
        np.concatenate([np.full(len(g), c) for c, g in enumerate(error_groups)])
        if error_groups
        else np.zeros((0,), dtype=int)
    )

    combined = np.concatenate([error_sample, correct_sample], axis=0)
    if len(combined) < 4:
        return

    reducer = umap.UMAP(n_neighbors=min(15, len(combined) - 1), random_state=0)
    projected = reducer.fit_transform(combined)
    n_error = len(error_sample)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        projected[:n_error, 0],
        projected[:n_error, 1],
        c=cluster_sample_labels,
        cmap="tab10",
        label="error",
        marker="x",
    )
    ax.scatter(
        projected[n_error:, 0],
        projected[n_error:, 1],
        c="gray",
        label="correct",
        marker="o",
        alpha=0.4,
    )
    class_name = class_names.get(class_id, str(class_id))
    ax.set_title(f"{class_name} (class_id={class_id})")
    ax.legend()
    fig.savefig(plots_dir / f"class_{class_id}_{class_name}.png", dpi=150)
    plt.close(fig)


def _write_embeddings(
    regions: list[Region],
    embeddings: np.ndarray,
    cluster_id_by_region_idx: dict[int, int],
    path: Path,
) -> None:
    # -1 for regions with no cluster assignment: "correct" regions (never
    # clustered) and error regions in classes that didn't get named
    # directions (too few error/correct examples for clustering_cfg.k_min).
    cluster_id = np.array(
        [cluster_id_by_region_idx.get(i, -1) for i in range(len(regions))]
    )
    np.savez(
        path,
        embeddings=embeddings,
        image_id=np.array([r.image_id for r in regions]),
        region_id=np.array([r.region_id for r in regions]),
        label=np.array([r.label for r in regions]),
        class_id=np.array([r.class_id for r in regions]),
        pixel_error_rate=np.array([r.pixel_error_rate for r in regions]),
        cluster_id=cluster_id,
        # "" sentinel, not None: Standard CV Mode never sets Region.country,
        # and np.array([None, ...]) is object-dtype -- np.load() defaults to
        # allow_pickle=False since NumPy 1.16.3, so an all-None country
        # column would make embeddings.npz raise ValueError on read for
        # anyone who indexes into "country" without passing allow_pickle=True.
        country=np.array([r.country or "" for r in regions]),
    )


def _write_pixel_accuracy(pixel_counts: dict[int, tuple[int, int]], path: Path) -> None:
    payload = {
        str(class_id): {
            "error_pixels": error_px,
            "total_pixels": total_px,
            "pixel_error_rate": error_px / total_px if total_px else None,
        }
        for class_id, (error_px, total_px) in sorted(pixel_counts.items())
    }
    path.write_text(json.dumps(payload, indent=2))


def _write_clusters(
    named_directions: list[NamedDirection],
    silhouette_by_class: dict[int, float | None],
    path: Path,
) -> None:
    by_class: dict[int, list[int]] = defaultdict(list)
    for d in named_directions:
        by_class[d.class_id].append(d.cluster_id)
    k_by_class = {class_id: len(clusters) for class_id, clusters in by_class.items()}
    path.write_text(
        json.dumps(
            {
                "k_selected": k_by_class,
                "silhouette_scores": silhouette_by_class,
                "clusters": by_class,
                "note": "per-region cluster assignments live in embeddings.npz's cluster_id array (-1 = unclustered: a correct region, or an error region in a class with no named direction)",
            },
            indent=2,
        )
    )


def _write_bias_directions(
    named_directions: list[NamedDirection],
    global_error_mode: GlobalErrorMode,
    path: Path,
) -> None:
    payload = {
        "global_error_mode": {
            "bias_vector": global_error_mode.bias_vector.tolist(),
            "concepts": global_error_mode.concepts,
            "stability": global_error_mode.stability,
            "note": "shared, class-agnostic direction; projected out of each direction below before its concepts were retrieved",
        },
        "directions": [
            {
                "class_id": d.class_id,
                "cluster_id": d.cluster_id,
                "bias_vector": d.bias_vector.tolist(),
                "concepts": d.concepts,
                "stability": d.stability,
                "residual_ratio": d.residual_ratio,
                "intra_inter_flag": d.intra_inter_flag,
                "country_confound_flag": d.country_confound_flag,
                "concept_comparison": d.concept_comparison,
                "country_pair_domain_shifts": d.country_pair_domain_shifts,
            }
            for d in named_directions
        ],
    }
    path.write_text(json.dumps(payload, indent=2))


def _write_summary(
    named_directions: list[NamedDirection],
    global_error_mode: GlobalErrorMode,
    stability_threshold: float,
    path: Path,
    class_names: dict[int, str],
    residual_ratio_threshold: float = 0.1,
) -> None:
    lines = ["# Bias Naming Summary\n"]
    if any(d.concept_comparison is not None for d in named_directions):
        lines.append("## How to read the concept-comparison buckets below\n")
        lines.append(
            "Each cluster's named concepts are cross-checked against three "
            "versions of its bias direction, computed at increasingly loose "
            "comparison scopes:\n"
        )
        lines.append(
            "- **prevalent**: retrieved from all three scopes (intra-tile, "
            "intra-country, inter-country) -- the strongest evidence of a "
            "genuine, geography-independent model failure mode."
        )
        lines.append(
            "- **tile-sensitive**: retrieved once tiles within the same "
            "country are pooled (intra-country) and in inter-country, but "
            "not from a single tile alone -- plausibly real, just needed a "
            "less noisy pool to surface."
        )
        lines.append(
            "- **domain-shift candidates**: retrieved *only* once "
            "comparisons are allowed to cross a country boundary "
            "(inter-country), absent from both intra-tile and intra-country "
            "-- may reflect a genuine visual difference between countries' "
            "landscapes rather than a model failure mode as such."
        )
        lines.append(
            "- **domain-shift candidates by country pair**: the pooled "
            "domain-shift-candidates figure above averages every "
            "cross-country pair into one direction before retrieval, which "
            "can dilute or cancel a shift specific to just one country pair "
            "-- this breakdown instead retrieves concepts separately for "
            "each (error country, correct country) pair, so a single "
            "country's distinct shift isn't hidden by averaging with "
            "others. A pair is omitted here if either side has fewer than "
            "min_pair_region_count regions: retrieve_concepts always "
            "returns a full top-k list regardless of how many embeddings a "
            "direction was averaged from, so a country with only a handful "
            "of regions could otherwise produce a confident-looking "
            "concept list from essentially a single noisy sample."
        )
        lines.append(
            "- **error-specific vs. scenery**: each (error country, correct "
            "country) direction mixes the error country's failures with how "
            "the two countries simply look different. Each pair line lists "
            "first the candidates that are *not* also retrieved from the "
            "same class's correct-vs-correct direction between those two "
            "countries (error-specific), then in brackets the ones that are "
            "(scenery: present even where the model gets it right).\n"
        )
    lines.append("## Global Error Mode (shared across classes)")
    lines.append(f"- stability: {global_error_mode.stability:.3f}")
    lines.append(f"- concepts: {', '.join(global_error_mode.concepts)}")
    lines.append(
        "- this direction is projected out of every class/cluster direction below before its concepts are retrieved\n"
    )
    for d in sorted(named_directions, key=lambda d: (d.class_id, d.cluster_id)):
        class_name = class_names.get(d.class_id, str(d.class_id))
        flags = []
        if d.stability < stability_threshold:
            flags.append("below stability threshold")
        if d.residual_ratio < residual_ratio_threshold:
            flags.append(
                "low residual signal -- mostly shared confound, concepts may be near-arbitrary"
            )
        if d.intra_inter_flag == "confound flagged":
            flags.append("intra/inter-tile confound flagged")
        if d.country_confound_flag == "confound flagged":
            flags.append("intra/inter-country confound flagged")
        flag = f" ({'; '.join(flags)})" if flags else ""
        lines.append(
            f"## {class_name} (class_id={d.class_id}), cluster {d.cluster_id}{flag}"
        )
        lines.append(f"- stability: {d.stability:.3f}")
        lines.append(f"- residual_ratio: {d.residual_ratio:.3f}")
        if d.intra_inter_flag is not None:
            lines.append(f"- intra/inter-tile check: {d.intra_inter_flag}")
        if d.country_confound_flag is not None:
            lines.append(f"- intra/inter-country check: {d.country_confound_flag}")
        if d.concept_comparison is not None:
            cc = d.concept_comparison
            lines.append(
                f"- prevalent (survives all scopes): {', '.join(cc['prevalent']) or 'none'}"
            )
            lines.append(
                f"- tile-sensitive (needs cross-tile diversity): {', '.join(cc['tile_sensitive']) or 'none'}"
            )
            lines.append(
                f"- domain-shift candidates (only appear cross-country): {', '.join(cc['domain_shift_candidates']) or 'none'}"
            )
        if d.country_pair_domain_shifts:
            lines.append("- domain-shift candidates by country pair (not pooled):")
            for pair in d.country_pair_domain_shifts:
                label = f"{pair['error_country']} error vs. {pair['correct_country']} correct"
                specific = pair.get("error_specific_candidates")
                if specific is None:
                    candidates = ", ".join(pair["domain_shift_candidates"]) or "none"
                    lines.append(f"  - {label}: {candidates} (no scenery baseline)")
                    continue
                scenery = sorted(set(pair["domain_shift_candidates"]) - set(specific))
                lines.append(
                    f"  - {label}: {', '.join(specific) or 'none'}"
                    f" [scenery: {', '.join(scenery) or 'none'}]"
                )
        lines.append(f"- concepts: {', '.join(d.concepts)}\n")
    path.write_text("\n".join(lines))
