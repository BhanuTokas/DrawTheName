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
from tqdm import tqdm

from drawthename.clustering import cluster_embeddings, select_k_by_silhouette
from drawthename.concept_bank import embed_concept_bank, load_concept_bank
from drawthename.data.cityscapes import TRAIN_ID_NAMES, CityscapesDataset
from drawthename.embeddings import embed_regions, load_backbone
from drawthename.naming import (
    NamedDirection,
    bias_direction,
    bootstrap_sign_stability,
    retrieve_concepts,
)
from drawthename.regions import Region, compute_error_mask, extract_regions
from drawthename.segmentation_model import SegmentationModel


def run_standard_cv_pipeline(config: dict[str, Any], output_dir: Path) -> None:
    """Phase 1: Cityscapes inference -> region extraction -> embeddings ->
    clustering -> bias naming. Writes embeddings.npz, clusters.json,
    bias_directions.json, summary.md, plots/ to output_dir."""
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
    concept_embeddings = embed_concept_bank(concept_texts, backbone)

    regions = _extract_all_regions(
        dataset,
        segmentation_model,
        config["regions"],
        limit=config["data"].get("limit"),
    )
    embeddings = embed_regions(regions, backbone)

    named_directions = _name_bias_directions(
        regions,
        embeddings,
        concept_texts,
        concept_embeddings,
        clustering_cfg=config["clustering"],
        naming_cfg=config["naming"],
        output_dir=output_dir,
        plot_max_points=config.get("plots", {}).get("max_points_per_class", 3000),
    )

    _write_embeddings(regions, embeddings, output_dir / "embeddings.npz")
    _write_clusters(named_directions, output_dir / "clusters.json")
    _write_bias_directions(named_directions, output_dir / "bias_directions.json")
    _write_summary(
        named_directions,
        config["naming"]["stability_threshold"],
        output_dir / "summary.md",
    )


def run_ftw_pipeline(config: dict[str, Any], output_dir: Path) -> None:
    """Phase 2: PRUE inference on FTW tiles -> tile classification ->
    sub-region extraction -> embeddings -> clustering -> intra/inter-tile
    comparison -> bias naming. Writes the same output set as Standard CV Mode,
    plus intra/inter-tile flags in bias_directions.json."""
    raise NotImplementedError


def _extract_all_regions(
    dataset: CityscapesDataset,
    segmentation_model: SegmentationModel,
    regions_cfg: dict[str, Any],
    limit: int | None = None,
) -> list[Region]:
    all_regions: list[Region] = []
    num_samples = min(len(dataset), limit) if limit else len(dataset)
    for index in tqdm(
        range(num_samples), desc="Running inference + extracting regions"
    ):
        sample = dataset[index]
        prediction = segmentation_model.predict(sample.image)
        error_mask = compute_error_mask(prediction, sample.ground_truth)
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
            )
        )
    return all_regions


def _name_bias_directions(
    regions: list[Region],
    embeddings: np.ndarray,
    concept_texts: list[str],
    concept_embeddings: np.ndarray,
    clustering_cfg: dict[str, Any],
    naming_cfg: dict[str, Any],
    output_dir: Path,
    plot_max_points: int = 3000,
) -> list[NamedDirection]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, region in enumerate(regions):
        by_class[region.class_id].append(i)

    named_directions: list[NamedDirection] = []
    for class_id, indices in by_class.items():
        class_regions = [regions[i] for i in indices]
        class_embeddings = embeddings[indices]

        error_idx = [i for i, r in enumerate(class_regions) if r.label == "error"]
        correct_idx = [i for i, r in enumerate(class_regions) if r.label == "correct"]
        if len(error_idx) < clustering_cfg["k_min"] or len(correct_idx) < 1:
            continue

        error_embeddings = class_embeddings[error_idx]
        correct_embeddings = class_embeddings[correct_idx]

        k = select_k_by_silhouette(
            error_embeddings, clustering_cfg["k_min"], clustering_cfg["k_max"]
        )
        cluster_labels = (
            cluster_embeddings(error_embeddings, k)
            if k > 1
            else np.zeros(len(error_embeddings), dtype=int)
        )

        _plot_class_embeddings(
            class_id,
            error_embeddings,
            correct_embeddings,
            cluster_labels,
            output_dir,
            plot_max_points,
        )

        for cluster_id in range(k):
            cluster_embeds = error_embeddings[cluster_labels == cluster_id]
            if len(cluster_embeds) == 0:
                continue
            direction = bias_direction(cluster_embeds, correct_embeddings)
            stability = bootstrap_sign_stability(
                cluster_embeds,
                correct_embeddings,
                n_resamples=naming_cfg["bootstrap_resamples"],
                cosine_threshold=naming_cfg["cosine_threshold"],
            )
            concepts = retrieve_concepts(
                direction,
                concept_texts,
                concept_embeddings,
                top_k=naming_cfg["top_k_concepts"],
            )
            named_directions.append(
                NamedDirection(
                    class_id=class_id,
                    cluster_id=cluster_id,
                    bias_vector=direction,
                    concepts=concepts,
                    stability=stability,
                )
            )
    return named_directions


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
    class_name = TRAIN_ID_NAMES.get(class_id, str(class_id))
    ax.set_title(f"{class_name} (class_id={class_id})")
    ax.legend()
    fig.savefig(plots_dir / f"class_{class_id}_{class_name}.png", dpi=150)
    plt.close(fig)


def _write_embeddings(
    regions: list[Region], embeddings: np.ndarray, path: Path
) -> None:
    np.savez(
        path,
        embeddings=embeddings,
        image_id=np.array([r.image_id for r in regions]),
        region_id=np.array([r.region_id for r in regions]),
        label=np.array([r.label for r in regions]),
        class_id=np.array([r.class_id for r in regions]),
        pixel_error_rate=np.array([r.pixel_error_rate for r in regions]),
    )


def _write_clusters(named_directions: list[NamedDirection], path: Path) -> None:
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for d in named_directions:
        by_class[d.class_id].append({"cluster_id": d.cluster_id})
    k_by_class = {class_id: len(clusters) for class_id, clusters in by_class.items()}
    path.write_text(
        json.dumps({"k_selected": k_by_class, "clusters": by_class}, indent=2)
    )


def _write_bias_directions(named_directions: list[NamedDirection], path: Path) -> None:
    payload = [
        {
            "class_id": d.class_id,
            "cluster_id": d.cluster_id,
            "bias_vector": d.bias_vector.tolist(),
            "concepts": d.concepts,
            "stability": d.stability,
            "intra_inter_flag": d.intra_inter_flag,
        }
        for d in named_directions
    ]
    path.write_text(json.dumps(payload, indent=2))


def _write_summary(
    named_directions: list[NamedDirection], stability_threshold: float, path: Path
) -> None:
    lines = ["# Bias Naming Summary\n"]
    for d in sorted(named_directions, key=lambda d: (d.class_id, d.cluster_id)):
        class_name = TRAIN_ID_NAMES.get(d.class_id, str(d.class_id))
        flag = (
            "" if d.stability >= stability_threshold else " (below stability threshold)"
        )
        lines.append(
            f"## {class_name} (class_id={d.class_id}), cluster {d.cluster_id}{flag}"
        )
        lines.append(f"- stability: {d.stability:.3f}")
        lines.append(f"- concepts: {', '.join(d.concepts)}\n")
    path.write_text("\n".join(lines))
