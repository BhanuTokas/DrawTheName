"""End-to-end orchestration for both pipeline modes (spec section 3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_standard_cv_pipeline(config: dict[str, Any], output_dir: Path) -> None:
    """Phase 1: Cityscapes inference -> region extraction -> embeddings ->
    clustering -> bias naming. Writes embeddings.npz, clusters.json,
    bias_directions.json, summary.md, plots/ to output_dir."""
    raise NotImplementedError


def run_ftw_pipeline(config: dict[str, Any], output_dir: Path) -> None:
    """Phase 2: PRUE inference on FTW tiles -> tile classification ->
    sub-region extraction -> embeddings -> clustering -> intra/inter-tile
    comparison -> bias naming. Writes the same output set as Standard CV Mode,
    plus intra/inter-tile flags in bias_directions.json."""
    raise NotImplementedError
