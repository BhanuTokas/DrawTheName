"""CLI entry point for the Mapillary Vistas geo-bias replication (see
drawthename/geo_bias_pipeline.py). --limit runs a fast smoke test on the
first N images instead of the full ~11,300-image pass -- use it to check
paths/config before committing to a full HPC run."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from drawthename.geo_bias_pipeline import run_geo_bias_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/mapillary_geo_bias.yaml")
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only the first N images."
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    run_geo_bias_pipeline(config, Path(config["output_dir"]), limit=args.limit)


if __name__ == "__main__":
    main()
