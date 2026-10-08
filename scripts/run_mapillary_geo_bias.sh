#!/bin/bash
# SLURM job (ASU Sol) for the Mapillary Vistas geo-bias replication
# (drawthename/geo_bias_pipeline.py).
#
# One-time setup on a login node before the first submission:
#   cp configs/mapillary_geo_bias.yaml.example configs/mapillary_geo_bias.yaml
#   # then set its data.root to the Vistas v1.2 download and device: cuda
#   mkdir -p logs   # SLURM opens the log files before this script runs
#
# Submit from the repo root with: sbatch scripts/run_mapillary_geo_bias.sh

#SBATCH --job-name=drawthename-mapillary-geo-bias
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=04:00:00   # ~11,300 images; CPU ran 4.06 s/img, GPU unmeasured -- padded
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge
export PATH="$HOME/.local/bin:$PATH"
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
if [[ ! -f scripts/run_mapillary_geo_bias.py ]]; then
    echo "Submit from the DrawTheName repo root (got $PWD)" >&2
    exit 1
fi

uv run python scripts/run_mapillary_geo_bias.py --config configs/mapillary_geo_bias.yaml
