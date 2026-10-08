#!/bin/bash
# SLURM job (ASU Sol) for Vistas Mode: DrawTheName's bias naming on Mapillary
# Vistas, by continent (drawthename/pipeline.py::run_vistas_pipeline).
#
# One-time setup on a login node before the first submission:
#   cp configs/vistas.yaml.example configs/vistas.yaml
#   # then set its data.root to the Vistas v1.2 download
#   mkdir -p logs   # SLURM opens the log files before this script runs
#
# Submit from the repo root with: sbatch scripts/run_vistas.sh

#SBATCH --job-name=drawthename-vistas
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=64G   # ~11,300 images' region embeddings plus bootstrap resampling over the large car/person pools
#SBATCH --time=1-00:00:00   # unmeasured at this scale -- padded to a full day
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge
export PATH="$HOME/.local/bin:$PATH"
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
if [[ ! -f scripts/run_vistas.py ]]; then
    echo "Submit from the DrawTheName repo root (got $PWD)" >&2
    exit 1
fi

uv run python scripts/run_vistas.py --config configs/vistas.yaml
