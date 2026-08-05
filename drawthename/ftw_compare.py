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

import warnings
from collections import defaultdict

import numpy as np

from drawthename.naming import bias_direction
from drawthename.regions import Region


def _group_by_country(
    regions: list[Region], embeddings: np.ndarray, side: str
) -> dict[str, list[np.ndarray]]:
    """Groups embeddings by region.country for the country-level comparison
    functions below (FTW Mode only). Regions with no country set (e.g.
    Standard CV Mode regions, which never populate Region.country) are
    excluded rather than silently grouped under a "None" country -- and
    since that should never happen when these functions are only ever
    called from run_ftw_pipeline, it's surfaced as a warning (with a count)
    rather than dropped with no visibility, in case a future caller or
    dataset loader accidentally routes non-FTW regions through here."""
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    n_dropped = 0
    for region, embedding in zip(regions, embeddings, strict=True):
        if region.country is None:
            n_dropped += 1
            continue
        grouped[region.country].append(embedding)
    if n_dropped:
        warnings.warn(
            f"{n_dropped} {side} region(s) had no country set and were "
            "excluded from country-level comparison -- these should always "
            "be FTW Mode regions with Region.country populated",
            stacklevel=3,
        )
    return grouped


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
    error and correct regions. See _group_by_country re: regions with no
    country set."""
    error_by_country = _group_by_country(error_regions, error_embeddings, "error")
    correct_by_country = _group_by_country(
        correct_regions, correct_embeddings, "correct"
    )

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
    error and correct pools combined. See _group_by_country re: regions with
    no country set."""
    error_by_country = _group_by_country(error_regions, error_embeddings, "error")
    correct_by_country = _group_by_country(
        correct_regions, correct_embeddings, "correct"
    )

    pair_directions = [
        np.mean(error_by_country[x], axis=0) - np.mean(correct_by_country[y], axis=0)
        for x in error_by_country
        for y in correct_by_country
        if x != y
    ]
    if not pair_directions:
        return None
    return np.mean(pair_directions, axis=0)


def inter_country_pair_directions(
    error_regions: list[Region],
    error_embeddings: np.ndarray,
    correct_regions: list[Region],
    correct_embeddings: np.ndarray,
    min_region_count: int = 1,
) -> dict[tuple[str, str], np.ndarray]:
    """Like inter_country_direction, but returns each (error_country,
    correct_country) pair's own direction separately instead of averaging
    them into one vector. Averaging over many pairs (as inter_country_direction
    does) can suppress a real, country-specific shift when it doesn't point
    the same way as other countries' pairs -- with N countries contributing
    up to N*(N-1) pairs, a single country's distinct signal becomes a
    shrinking fraction of that average as more countries are added.

    min_region_count excludes a pair if either side has fewer than that many
    regions: retrieve_concepts always returns a full top-k list regardless of
    how many embeddings a direction was averaged from, so a country with only
    a handful of regions can otherwise produce a confident-looking concept
    list from essentially a single noisy sample -- indistinguishable in the
    output from a direction backed by thousands of regions.

    Only includes pairs with distinct countries and pools meeting the
    minimum on each side. See _group_by_country re: regions with no country
    set."""
    error_by_country = _group_by_country(error_regions, error_embeddings, "error")
    correct_by_country = _group_by_country(
        correct_regions, correct_embeddings, "correct"
    )

    return {
        (x, y): np.mean(error_by_country[x], axis=0)
        - np.mean(correct_by_country[y], axis=0)
        for x in error_by_country
        if len(error_by_country[x]) >= min_region_count
        for y in correct_by_country
        if x != y and len(correct_by_country[y]) >= min_region_count
    }


def domain_shift_candidates_per_pair(
    intra_tile_concepts: list[str],
    intra_country_concepts: list[str],
    pair_concepts: dict[tuple[str, str], list[str]],
) -> dict[tuple[str, str], list[str]]:
    """For each country pair's own concept list (retrieved from that pair's
    own direction, not a blended average), returns the concepts unique to
    that pair -- not in intra-tile or intra-country. This is the per-pair
    counterpart to compare_concept_sets' domain_shift_candidates, which
    pools every pair into one direction before retrieval and so can dilute
    or cancel a shift specific to just one country pair."""
    intra_tile_set = set(intra_tile_concepts)
    intra_country_set = set(intra_country_concepts)
    return {
        pair: sorted(set(concepts) - intra_tile_set - intra_country_set)
        for pair, concepts in pair_concepts.items()
    }


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
