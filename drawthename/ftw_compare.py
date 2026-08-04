"""Intra-tile/inter-tile and intra-country/inter-country comparisons, FTW
Mode only (spec section 3.2 step 5, extended to country scope).

Intra-tile comparisons control for tile-level confounds (geography, acquisition
date, sensor conditions); divergence between intra- and inter-tile bias
directions flags likely confounds rather than genuine model bias.

Intra-country/inter-country comparisons extend the same idea one level up:
intra-country pools tiles within one country (controls for country-level
effects but allows tile-to-tile variation), inter-country explicitly crosses
country boundaries (isolates domain shift). compare_concept_sets buckets the
named concepts from all three scopes to separate prevalent, tile-sensitive,
and domain-shift-candidate signal.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from drawthename.naming import bias_direction
from drawthename.regions import Region


def intra_tile_direction(
    error_regions: list[Region],
    error_embeddings: np.ndarray,
    correct_regions: list[Region],
    correct_embeddings: np.ndarray,
) -> np.ndarray | None:
    """Averages, over tiles containing both error and correct regions of this
    class/cluster, that tile's (error mean - correct mean) -- differencing
    within the same tile cancels out any tile-level confound shared by both
    sides. Returns None if no tile has both error and correct regions."""
    error_by_tile: dict[str | None, list[np.ndarray]] = defaultdict(list)
    for region, embedding in zip(error_regions, error_embeddings, strict=True):
        error_by_tile[region.tile_id].append(embedding)
    correct_by_tile: dict[str | None, list[np.ndarray]] = defaultdict(list)
    for region, embedding in zip(correct_regions, correct_embeddings, strict=True):
        correct_by_tile[region.tile_id].append(embedding)

    shared_tiles = set(error_by_tile) & set(correct_by_tile)
    if not shared_tiles:
        return None

    per_tile_directions = [
        np.mean(error_by_tile[tile_id], axis=0)
        - np.mean(correct_by_tile[tile_id], axis=0)
        for tile_id in shared_tiles
    ]
    return np.mean(per_tile_directions, axis=0)


def inter_tile_direction(
    error_regions: list[Region],
    error_embeddings: np.ndarray,
    correct_regions: list[Region],
    correct_embeddings: np.ndarray,
) -> np.ndarray:
    """Plain mean(error) - mean(correct), pooling across all tiles regardless
    of which tile each region came from -- the naive baseline that
    intra_tile_direction is compared against to detect tile-level confounds.
    (error_regions/correct_regions are unused here -- only their embeddings
    matter -- but kept in the signature to mirror intra_tile_direction so
    both functions can be called interchangeably.)"""
    del error_regions, correct_regions
    return bias_direction(error_embeddings, correct_embeddings)


def flag_confound(
    a_direction: np.ndarray, b_direction: np.ndarray, threshold: float = 0.5
) -> bool:
    """True if cosine similarity between two directions is below threshold.
    Used for both the tile-level (intra-tile vs. inter-tile) and
    country-level (intra-country vs. inter-country) confound checks -- the
    comparison itself doesn't care which scope the two directions came
    from."""
    a_norm = np.linalg.norm(a_direction) + 1e-12
    b_norm = np.linalg.norm(b_direction) + 1e-12
    cos_sim = np.dot(a_direction, b_direction) / (a_norm * b_norm)
    return bool(cos_sim < threshold)


def intra_country_direction(
    error_regions: list[Region],
    error_embeddings: np.ndarray,
    correct_regions: list[Region],
    correct_embeddings: np.ndarray,
) -> np.ndarray | None:
    """Averages, over countries containing both error and correct regions of
    this class/cluster, that country's (error mean - correct mean) --
    pooling across all of a country's tiles on each side, so whatever is
    specific to that country (not any one tile) cancels out. This isolates
    tile-to-tile variation *within* a country from country-level effects,
    the way intra_tile_direction isolates region-to-region variation within
    a tile from tile-level effects. Returns None if no country has both
    error and correct regions."""
    error_by_country: dict[str | None, list[np.ndarray]] = defaultdict(list)
    for region, embedding in zip(error_regions, error_embeddings, strict=True):
        error_by_country[region.country].append(embedding)
    correct_by_country: dict[str | None, list[np.ndarray]] = defaultdict(list)
    for region, embedding in zip(correct_regions, correct_embeddings, strict=True):
        correct_by_country[region.country].append(embedding)

    shared_countries = set(error_by_country) & set(correct_by_country)
    if not shared_countries:
        return None

    per_country_directions = [
        np.mean(error_by_country[country], axis=0)
        - np.mean(correct_by_country[country], axis=0)
        for country in shared_countries
    ]
    return np.mean(per_country_directions, axis=0)


def inter_country_direction(
    error_regions: list[Region],
    error_embeddings: np.ndarray,
    correct_regions: list[Region],
    correct_embeddings: np.ndarray,
) -> np.ndarray | None:
    """Averages (error mean in country X) - (correct mean in country Y) over
    every ordered pair of *distinct* countries X != Y that respectively have
    error and correct regions -- deliberately excludes same-country pairs
    (unlike a naive globally-pooled bias_direction, which would dilute the
    cross-country signal with them), isolating the component of the
    direction driven specifically by crossing a country boundary. Returns
    None if fewer than two distinct countries are represented across the
    error and correct pools combined."""
    error_by_country: dict[str | None, list[np.ndarray]] = defaultdict(list)
    for region, embedding in zip(error_regions, error_embeddings, strict=True):
        error_by_country[region.country].append(embedding)
    correct_by_country: dict[str | None, list[np.ndarray]] = defaultdict(list)
    for region, embedding in zip(correct_regions, correct_embeddings, strict=True):
        correct_by_country[region.country].append(embedding)

    pair_directions = [
        np.mean(error_by_country[x], axis=0) - np.mean(correct_by_country[y], axis=0)
        for x in error_by_country
        for y in correct_by_country
        if x != y
    ]
    if not pair_directions:
        return None
    return np.mean(pair_directions, axis=0)


def compare_concept_sets(
    intra_tile_concepts: list[str],
    intra_country_concepts: list[str],
    inter_country_concepts: list[str],
) -> dict[str, list[str]]:
    """Buckets concepts by which comparison scope(s) retrieve them, from most
    to least controlled: intra-tile (same tile), intra-country (same
    country, different tile), inter-country (crosses a country boundary). A
    concept surviving all three scopes is the most robust evidence of
    genuine, tile/country-independent model bias ("prevalent"). One that
    shows up once tile-diversity is allowed but not within a single tile is
    "tile_sensitive" -- plausibly real but needed a bigger, less noisy pool
    to surface. One that only appears once comparisons are allowed to cross
    a country boundary is a "domain_shift_candidate", not necessarily a
    genuine model failure mode -- see flag_confound for the corresponding
    quantitative (cosine similarity) check."""
    intra_tile_set = set(intra_tile_concepts)
    intra_country_set = set(intra_country_concepts)
    inter_country_set = set(inter_country_concepts)
    return {
        "prevalent": sorted(intra_tile_set & intra_country_set & inter_country_set),
        "tile_sensitive": sorted(
            (intra_country_set & inter_country_set) - intra_tile_set
        ),
        "domain_shift_candidates": sorted(
            inter_country_set - intra_tile_set - intra_country_set
        ),
    }
