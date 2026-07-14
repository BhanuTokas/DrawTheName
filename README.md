# DrawTheName

A bias discovery and naming pipeline for segmentation models, applied to the
[Fields of the World (FTW)](https://fieldsofthe.world/) agricultural field
boundary segmentation task and the [PRUE](https://github.com/fieldsoftheworld/ftw-prue)
model. The segmentation model is treated as a black box: the pipeline only
consumes its predictions (and optionally intermediate embeddings).

## Modes

- **Standard CV Mode** (`configs/standard_cv.yaml`): validated first, on
  Cityscapes, without tile structure.
- **FTW Mode** (`configs/ftw.yaml`): extends Standard CV Mode with tile-aware
  intra-tile vs. inter-tile comparisons to flag geography/date confounds.

## Pipeline

For each mode: run inference -> extract connected-component error regions ->
embed regions with a CLIP-family backbone -> cluster per-class error
embeddings (k-means, silhouette-selected k) -> name each cluster's bias
direction (mean error - mean correct embedding) via concept-bank retrieval,
with bootstrap sign-stability validation. FTW Mode additionally compares
intra-tile vs. inter-tile bias directions to flag tile-level confounds.

Primary VLM backbone is CLIP ViT-L/14, chosen from a EuroSAT sanity check
run in a precursor repo before committing to it here (see `misc/` for the
originating project description).

## Layout

- `drawthename/` -- pipeline package (data loaders, region extraction,
  embeddings, clustering, naming, FTW tile comparison).
- `configs/` -- per-mode hyperparameters and paths.
- `concept_banks/` -- text concept descriptors for bias naming (not yet
  curated -- Phase 3).
- `scripts/` -- CLI entry points (`run_standard_cv.py`, `run_ftw.py`).
- `tests/`
- `results/` -- pipeline outputs (gitignored).

## Setup

```
uv sync --extra dev
```

`configs/*.yaml` are gitignored (machine-specific `data.root` paths, since
dataset locations differ across machines). Copy the committed `.example`
templates and fill in your local path:

```
cp configs/standard_cv.yaml.example configs/standard_cv.yaml
cp configs/ftw.yaml.example configs/ftw.yaml
```

## Status

Scaffolding only -- pipeline functions are stubbed with `NotImplementedError`.
Implementation order: Phase 1 (Standard CV Mode on Cityscapes) -> Phase 2
(FTW Mode) -> Phase 3 (concept banks).
