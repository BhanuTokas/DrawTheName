# DrawTheName

A bias discovery and naming pipeline for segmentation models, applied to the
[Fields of the World (FTW)](https://fieldsofthe.world/) agricultural field
boundary segmentation task and the [PRUE](https://github.com/fieldsoftheworld/ftw-prue)
model. The segmentation model is treated as a black box: the pipeline only
consumes its predictions (and optionally intermediate embeddings).

## Modes

- **Standard CV Mode** (`configs/standard_cv.yaml`): validated first, on
  Cityscapes, without tile structure.
- **FTW Mode** (`configs/ftw.yaml`): extends Standard CV Mode with
  tile-aware intra-tile vs. inter-tile comparisons (geography/date
  confounds) and, for multi-country runs, intra-country vs. inter-country
  comparisons to separate genuine model bias from domain shift between
  countries.

## Pipeline

For each mode: run inference -> extract connected-component error regions ->
embed regions with a CLIP-family backbone -> cluster per-class error
embeddings (k-means, silhouette-selected k) -> name each cluster's bias
direction (mean error - mean correct embedding) via concept-bank retrieval,
with bootstrap sign-stability validation. FTW Mode additionally compares
intra-tile vs. inter-tile bias directions to flag tile-level confounds, and
-- given 2+ countries in `data.countries` -- intra-country vs. inter-country
bias directions to flag country-level confounds.

FTW Mode's country-level analysis (multi-country runs only):

- `country_confound_flag` per cluster: cosine similarity between the
  intra-country direction (pools regions within each country, tile identity
  ignored) and the inter-country direction (pairs regions across distinct
  countries, same-country pairs excluded) -- low similarity flags a likely
  country-level confound.
- `concept_comparison` per cluster: buckets each retrieved concept as
  *prevalent* (survives intra-tile, intra-country, and inter-country alike),
  *tile-sensitive* (needs cross-tile diversity to surface but not
  cross-country), or a *domain-shift candidate* (only appears once
  comparisons cross a country boundary). This pooled version averages every
  cross-country pair into one direction before retrieval, which can dilute
  or fully cancel a shift specific to just one country pair.
- `country_pair_domain_shifts`: the same domain-shift-candidate concept, but
  computed separately per (error country, correct country) pair rather than
  pooled -- so a shift specific to one country isn't hidden by averaging
  with unrelated pairs. Excludes any pair where either side has fewer than
  `ftw_compare.min_pair_region_count` regions (default 10), since concept
  retrieval always returns a full top-k list regardless of how few
  embeddings a direction was averaged from -- without this floor, a country
  with only a handful of regions can produce a confident-looking concept
  list from what's essentially a single noisy sample.
- `data.class_remap`: per-country `{old_class_id: new_class_id}` correction
  for mask label conventions that don't match the rest (see Status below).

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
  (dataloaders; `data/ftw.py` also carries each tile's `country` and the
  `class_remap` correction), `segmentation_model.py` (black-box wrapper:
  SegFormer for Standard CV Mode, PRUE for FTW Mode), `regions.py` (error
  mask + connected-component extraction; each `Region` carries `tile_id`
  and `country`), `embeddings.py` (CLIP backbone), `clustering.py`,
  `concept_bank.py`, `naming.py` (bias direction, deconfounding, bootstrap
  stability, concept retrieval), `ftw_compare.py` (intra/inter-tile *and*
  intra/inter-country confound detection, per-country-pair domain-shift
  candidates), `pipeline.py` (orchestration).
- `configs/` -- per-mode hyperparameters and paths.
- `concept_banks/general_concepts.txt` -- curated for Standard CV Mode:
  Broden concepts filtered to street/urban-plausible ones, a Cityscapes
  class/vehicle-part gap-fill supplement, and hand-written lighting/
  occlusion/scale/boundary-ambiguity qualifiers.
  `concept_banks/ftw_concepts.txt` -- curated for FTW Mode (~200
  descriptors): field boundary ambiguity, seasonal variation, cloud
  cover/shadow, mixed crop types, field size, tillage/harvest state, plus a
  soil/vegetation and general remote-sensing supplement.
- `scripts/` -- CLI entry points: `run_standard_cv.py` / `run_ftw.py` (the
  pipeline itself), `validate_naming_synthetic.py` /
  `validate_naming_synthetic_ftw.py` (synthetic-bias sanity checks -- inject
  a known transform, e.g. cloud cover or motion blur, into real crops and
  confirm the pipeline recovers the matching concept bank entry, without
  needing a real segmentation model or ground truth).
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

## Usage

```
uv run python scripts/run_standard_cv.py --config configs/standard_cv.yaml
uv run python scripts/run_ftw.py --config configs/ftw.yaml
```

`--config` defaults to `configs/standard_cv.yaml` / `configs/ftw.yaml`
respectively, so it can be omitted once those files exist. Each run writes
`embeddings.npz`, `clusters.json`, `bias_directions.json`,
`pixel_accuracy.json`, `summary.md`, and `plots/` to `output_dir` (set in
the config).

For FTW Mode, list 2+ countries in `data.countries` to additionally get the
intra/inter-country confound check and per-country-pair domain-shift
breakdown described above:

```yaml
data:
  countries: [austria, belgium, kenya, south_africa]
```

If a country's mask labels use a different convention than the rest (see
the Kenya case under Status), correct it with `data.class_remap` rather
than special-casing it in code:

```yaml
data:
  class_remap:
    kenya:
      3: 255 # unconfirmed area -> ignore, not forced into a real class
```

Before trusting a concept bank's naming output on real data, sanity-check
it against known injected biases (no segmentation model or ground truth
needed):

```
uv run python scripts/validate_naming_synthetic.py --config configs/standard_cv.yaml
uv run python scripts/validate_naming_synthetic_ftw.py --config configs/ftw.yaml
```

## Status

**Phase 1 (Standard CV Mode) is implemented and validated** end-to-end on
the full Cityscapes val split (500 images): inference, region extraction
(with subdivision), CLIP embedding, clustering, global-error-mode
deconfounding, and concept retrieval all run via `scripts/run_standard_cv.py`
in roughly 13-17 minutes. `concept_banks/general_concepts.txt` is curated.

**Phase 2 (FTW Mode) is implemented and validated**, including full runs
against real multi-country val splits (Austria, Belgium, Kenya, South
Africa, Portugal), with a curated concept bank
(`concept_banks/ftw_concepts.txt`). The PRUE checkpoint in use
(`prue-unet-logcoshdice-augs-efficientnetb3-winargb`) is one of PRUE's own
ablation variants: single-window, RGB-only (`in_channels=3`,
`temporal_options=window_a_rgb`) -- not the 4-band RGB+NIR, dual-window
input the original spec assumed. NIR-dropping logic (`to_rgb`) is kept for
any future checkpoint that does take more bands, but is a no-op against
this one.

**Phase 2's multi-country analysis surfaced two real data issues, both now
handled via `data.class_remap` rather than special-cased in code:**

- Kenya's val-split masks are presence-only annotated (fields confirmed,
  background not), so unconfirmed area is labeled `3` rather than `0` --
  remapped to `255` (excluded from analysis) rather than forced into any
  real class. An earlier attempt at remapping `3` into field-boundary
  manufactured a spurious domain-shift signal that didn't survive
  correction.
- Small val splits (e.g. Portugal's 9 tiles, Kenya's confirmed-region count
  in the single digits) can make a country's own bias direction unreliable
  on its own -- `ftw_compare.min_pair_region_count` (default 10) excludes
  any country pair from the per-pair domain-shift breakdown where either
  side falls below that floor, since concept retrieval returns a
  full-looking top-k list regardless of how few regions backed it.
