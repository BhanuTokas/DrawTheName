"""CLI entry point for Phase 2 -- FTW Mode."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from drawthename.pipeline import run_ftw_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/ftw.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    run_ftw_pipeline(config, Path(config["output_dir"]))


if __name__ == "__main__":
    main()
