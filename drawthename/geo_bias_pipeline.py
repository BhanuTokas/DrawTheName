"""Orchestrates the Mapillary Vistas geo-bias replication (see
drawthename/geo_bias.py for the metric definitions this reuses): runs the
Standard CV Mode segmentation model over every Vistas image with a known
continent, accumulates one confusion matrix per continent, then reports
per-class IoU, Disp (the paper's geo-disparity metric), and the class-
merging Disp reduction for the paper's 7 shared classes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from tqdm import tqdm

from drawthename.data.mapillary_vistas import (
    DEFAULT_CONTINENT_LABELS_PATH,
    MERGE_GROUPS,
    SHARED_CLASSES,
    MapillaryVistasDataset,
)
from drawthename.geo_bias import ConfusionAccumulator, disp, iou_per_class, merged_iou, pct_reduction
from drawthename.segmentation_model import SegmentationModel


class _SampleLike(Protocol):
    image: np.ndarray
    ground_truth: np.ndarray
    continent: str


class _DatasetLike(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> _SampleLike: ...


class _ModelLike(Protocol):
    def predict(self, image: np.ndarray) -> np.ndarray: ...


def _run_inference(
    dataset: _DatasetLike, model: _ModelLike, limit: int | None = None
) -> tuple[ConfusionAccumulator, dict[str, int]]:
    """Runs model.predict over dataset and accumulates one confusion matrix
    per continent. Takes an already-constructed dataset/model (rather than a
    config) so it's testable against fakes without touching real files, a
    real checkpoint, or a GPU."""
    accumulator = ConfusionAccumulator()
    n_images_by_continent: dict[str, int] = {}
    indices = range(len(dataset)) if limit is None else range(min(limit, len(dataset)))
    for i in tqdm(indices, desc="Vistas inference"):
        sample = dataset[i]
        prediction = model.predict(sample.image)
        accumulator.update(sample.continent, sample.ground_truth, prediction)
        n_images_by_continent[sample.continent] = (
            n_images_by_continent.get(sample.continent, 0) + 1
        )
    return accumulator, n_images_by_continent


def _compute_results(
    accumulator: ConfusionAccumulator, n_images_by_continent: dict[str, int]
) -> dict[str, Any]:
    """Turns accumulated confusion matrices into per-class IoU, Disp, and the
    class-merging Disp reduction -- the paper's own metrics (drawthename/geo_bias.py)."""
    continents = sorted(accumulator.matrices)
    per_class: dict[str, Any] = {}
    for class_name, class_id in SHARED_CLASSES.items():
        iou_by_continent = {
            c: iou_per_class(accumulator.matrices[c], class_id) for c in continents
        }
        per_class[class_name] = {
            "iou_by_continent": iou_by_continent,
            "disp_before_merge": disp(list(iou_by_continent.values())),
        }

    for group_name, member_names in MERGE_GROUPS.items():
        member_ids = [SHARED_CLASSES[m] for m in member_names]
        merged_iou_by_continent = {
            c: merged_iou(accumulator.matrices[c], member_ids) for c in continents
        }
        group_disp_after = disp(list(merged_iou_by_continent.values()))
        for member_name in member_names:
            per_class[member_name]["merge_group"] = group_name
            per_class[member_name]["merged_iou_by_continent"] = merged_iou_by_continent
            per_class[member_name]["disp_after_merge"] = group_disp_after
            per_class[member_name]["disp_pct_reduction"] = pct_reduction(
                per_class[member_name]["disp_before_merge"], group_disp_after
            )

    return {
        "n_images_by_continent": n_images_by_continent,
        "continents": continents,
        "classes": per_class,
    }


def run_geo_bias_pipeline(
    config: dict[str, Any], output_dir: Path, limit: int | None = None
) -> dict[str, Any]:
    """Runs the replication end to end and writes geo_bias_results.json (plus
    returning the same payload) to output_dir. limit caps the number of
    images processed, for a fast smoke test before a full HPC run."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MapillaryVistasDataset(
        root=Path(config["data"]["root"]),
        continent_labels_path=Path(
            config["data"].get("continent_labels", DEFAULT_CONTINENT_LABELS_PATH)
        ),
    )
    print(f"{len(dataset)} Vistas images with a known continent")

    model = SegmentationModel(
        checkpoint=config["segmentation_model"]["checkpoint"],
        device=config["segmentation_model"]["device"],
    )

    accumulator, n_images_by_continent = _run_inference(dataset, model, limit)
    results = _compute_results(accumulator, n_images_by_continent)

    out_path = output_dir / "geo_bias_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")

    _print_summary(results["classes"], results["continents"])
    return results


def _print_summary(per_class: dict[str, Any], continents: list[str]) -> None:
    print("\n=== per-class IoU by continent ===")
    header = "class".ljust(12) + "".join(c[:4].ljust(8) for c in continents)
    print(header)
    for class_name, data in per_class.items():
        row = class_name.ljust(12)
        for c in continents:
            v = data["iou_by_continent"].get(c, float("nan"))
            row += f"{v:.3f}".ljust(8) if v == v else "nan".ljust(8)
        print(row)

    print("\n=== Disp (geo-disparity) before/after class-merging ===")
    print(
        "class".ljust(12)
        + "group".ljust(12)
        + "before".ljust(10)
        + "after".ljust(10)
        + "% reduction"
    )
    for class_name, data in per_class.items():
        group = data.get("merge_group", "-")
        before = data["disp_before_merge"]
        after = data.get("disp_after_merge", float("nan"))
        reduction = data.get("disp_pct_reduction", float("nan"))
        print(
            class_name.ljust(12)
            + group.ljust(12)
            + f"{before:.3f}".ljust(10)
            + f"{after:.3f}".ljust(10)
            + f"{reduction:.1f}%"
        )
