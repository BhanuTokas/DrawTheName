#!/bin/bash
# SLURM job for the Mapillary Vistas geo-bias replication
# (drawthename/geo_bias_pipeline.py). Update the #SBATCH values below for
# your cluster (partition/account names, time budget) before submitting --
# these are placeholders, not verified against any specific cluster.
#
# One-time setup on a login node before the first submission:
#   uv sync
#   # then update configs/mapillary_geo_bias.yaml's data.root to wherever
#   # Vistas gets downloaded (v1.2, combined training+validation).
#
# Submit with: sbatch scripts/run_mapillary_geo_bias.sbatch

#SBATCH --job-name=drawthename-mapillary-geo-bias
#SBATCH --partition=gpu                # <-- set to your cluster's GPU partition
#SBATCH --account=your_account         # <-- set to your allocation/account
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00                # SegFormer-b0 inference over ~11,300 images; pad if the checkpoint changes
#SBATCH --output=logs/mapillary_geo_bias_%j.out

set -euo pipefail
mkdir -p logs

cd "$SLURM_SUBMIT_DIR"

# Compute nodes on many clusters have no network access -- activate the venv
# directly and call python rather than `uv run` (which might try to touch
# the network mid-job), matching the FTW HPC job's approach.
source .venv/bin/activate

# Quick sanity check on a handful of images first if you haven't run this on
# this cluster before -- uncomment to smoke-test paths/checkpoint download
# before committing to the full ~11,300-image pass:
# python scripts/run_mapillary_geo_bias.py --config configs/mapillary_geo_bias.yaml --limit 20

python scripts/run_mapillary_geo_bias.py --config configs/mapillary_geo_bias.yaml
