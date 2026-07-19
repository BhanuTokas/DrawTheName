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

Standard CV Mode (implemented) additionally:

- Treats the segmentation model as a genuine black box via a pluggable
  `SegmentationModel` wrapper (default: a Hugging Face SegFormer checkpoint
  fine-tuned on Cityscapes).
- Splits large connected components (e.g. a big sky/road/vegetation blob)
  into a grid of smaller sub-regions (`regions.subdivision_size`) instead of
  treating the whole blob as one sample -- error regions are naturally small
  (boundary slivers) so this mostly grows the correct-region pool, which
  measurably stabilizes bootstrap estimates for sparse classes.
- Computes a `GlobalErrorMode` -- the direction shared across nearly every
  class's bias vector (e.g. "errors tend to be small/blurry/oddly-cropped
  regardless of class") -- and projects it out of each class's bias vector
  before concept retrieval, so a class's named concepts reflect what's
  specific to it rather than that shared confound.
- Mean-centers the concept bank's text embeddings before retrieval,
  correcting CLIP's narrow-cone anisotropy for short phrases (a plain
  concept bank embeds into a tight cone rather than spreading over the
  sphere, which otherwise lets a handful of "hub" concepts dominate
  regardless of the actual bias direction).
- Reports both a **region error rate** (fraction of a class's regions
  labeled error) and a true **pixel error rate** per class -- these can
  diverge a lot, since region-counting weighs a tiny boundary sliver the
  same as a huge well-segmented blob.

## Layout

- `drawthename/` -- pipeline package: `data/cityscapes.py` / `data/ftw.py`
  (dataloaders), `segmentation_model.py` (black-box wrapper: SegFormer for
  Standard CV Mode, PRUE for FTW Mode), `regions.py` (error mask +
  connected-component extraction), `embeddings.py` (CLIP backbone),
  `clustering.py`, `concept_bank.py`, `naming.py` (bias direction,
  deconfounding, bootstrap stability, concept retrieval), `ftw_compare.py`
  (intra/inter-tile confound detection), `pipeline.py` (orchestration).
- `configs/` -- per-mode hyperparameters and paths.
- `concept_banks/general_concepts.txt` -- curated for Standard CV Mode:
  Broden concepts filtered to street/urban-plausible ones, a Cityscapes
  class/vehicle-part gap-fill supplement, and hand-written lighting/
  occlusion/scale/boundary-ambiguity qualifiers. `ftw_concepts.txt` is still
  a placeholder (Phase 3).
- `scripts/` -- CLI entry points (`run_standard_cv.py`, `run_ftw.py`).
- `tests/`
- `results/` -- pipeline outputs (gitignored): `embeddings.npz`,
  `clusters.json`, `bias_directions.json` (includes the global error mode),
  `pixel_accuracy.json`, `summary.md`, `plots/`.

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

**Phase 1 (Standard CV Mode) is implemented and validated** end-to-end on
the full Cityscapes val split (500 images): inference, region extraction
(with subdivision), CLIP embedding, clustering, global-error-mode
deconfounding, and concept retrieval all run via `scripts/run_standard_cv.py`
in roughly 13-17 minutes. `concept_banks/general_concepts.txt` is curated.

**Phase 2 (FTW Mode) is implemented and smoke-tested** against the real
Austria val split and the local PRUE checkpoint
(`prue-unet-logcoshdice-augs-efficientnetb3-winargb`, RGB-only:
`in_channels=3`, `temporal_options=window_a_rgb`, contradicting the original
spec's 4-band RGB+NIR assumption): inference, tile-aware region extraction,
CLIP embedding, clustering, deconfounding, intra/inter-tile confound
checking, and concept retrieval all run via `scripts/run_ftw.py`. Its concept
bank (`concept_banks/ftw_concepts.txt`) is still a placeholder -- curation is
Phase 3, so current concept names aren't meaningful yet, only the pipeline
mechanics are validated.
