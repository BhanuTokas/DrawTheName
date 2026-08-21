"""One-off preprocessing: converts the per-image Mapillary Vistas geolocation
CSV published alongside "Classification Drives Geographic Bias in Street
Scene Segmentation" (https://zenodo.org/records/11459554) into continent
labels, entirely offline (reverse_geocoder's bundled place database -- no
network calls at lookup time beyond the initial CSV download).

Not run automatically -- its output, drawthename/data/mapillary_vistas_continents.csv,
is already committed. Re-run only if the upstream Zenodo file changes.
Needs the dev extras: `uv sync --extra dev`.

Of the CSV's 20,000 rows, only ~11,300 have coordinates; that count matches
the paper's own reported post-preprocessing image count, so this is very
likely the exact metadata source the paper used.
"""

from __future__ import annotations

import csv
import urllib.request
from collections import Counter
from pathlib import Path

import reverse_geocoder as rg
from pycountry_convert import (
    country_alpha2_to_continent_code,
    country_alpha2_to_country_name,
)

ZENODO_CSV_URL = (
    "https://zenodo.org/api/records/11459554/files/"
    "mapillary_vistas_geographic_metadata.csv/content"
)
OUT_PATH = (
    Path(__file__).parent.parent
    / "drawthename"
    / "data"
    / "mapillary_vistas_continents.csv"
)

CONTINENT_CODE_TO_NAME = {
    "EU": "Europe",
    "NA": "North America",
    "SA": "South America",
    "AF": "Africa",
    "AS": "Asia",
    "OC": "Oceania",
    "AN": "Antarctica",
}

# pycountry_convert's continent table omits a handful of micro-states/edge
# cases; patched by hand rather than pulling in another dependency for it.
MANUAL_COUNTRY_OVERRIDES = {
    "VA": ("Holy See (Vatican City State)", "Europe"),  # enclaved within Rome
}


def main() -> None:
    print(f"downloading {ZENODO_CSV_URL} ...")
    with urllib.request.urlopen(ZENODO_CSV_URL) as response:
        text = response.read().decode("utf-8")
    rows = list(csv.DictReader(text.splitlines()))
    print(f"read {len(rows)} rows")

    valid_rows = [r for r in rows if r["Latitude"].strip() and r["Longitude"].strip()]
    print(
        f"{len(valid_rows)} rows have coordinates, "
        f"{len(rows) - len(valid_rows)} missing (dropped)"
    )
    coords = [(float(r["Latitude"]), float(r["Longitude"])) for r in valid_rows]

    print("reverse-geocoding (offline, bundled place database)...")
    # mode=1 forces single-process querying; the default multiprocessing mode
    # spawns workers via sys.executable, which can resolve to an unrelated
    # interpreter in some environments (it did in ours) and crash.
    geocoder = rg.RGeocoder(mode=1, verbose=True)
    results = geocoder.query(coords)

    unmapped_countries: set[str] = set()
    out_rows = []
    for row, res in zip(valid_rows, results, strict=True):
        cc = res["cc"]
        if cc in MANUAL_COUNTRY_OVERRIDES:
            country_name, continent = MANUAL_COUNTRY_OVERRIDES[cc]
        else:
            try:
                continent = CONTINENT_CODE_TO_NAME[country_alpha2_to_continent_code(cc)]
                country_name = country_alpha2_to_country_name(cc)
            except KeyError:
                unmapped_countries.add(cc)
                continue
        out_rows.append(
            {
                "Mapillary_ID": row["Mapillary_ID"],
                "Latitude": row["Latitude"],
                "Longitude": row["Longitude"],
                "country_code": cc,
                "country_name": country_name,
                "continent": continent,
                "geocoded_city": res["name"],
            }
        )

    if unmapped_countries:
        print(
            f"WARNING: {len(unmapped_countries)} country code(s) had no continent "
            f"mapping and were dropped: {sorted(unmapped_countries)}"
        )

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Mapillary_ID",
                "Latitude",
                "Longitude",
                "country_code",
                "country_name",
                "continent",
                "geocoded_city",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {len(out_rows)} labeled rows to {OUT_PATH}")

    print("\ncontinent distribution:")
    for continent, n in Counter(r["continent"] for r in out_rows).most_common():
        print(f"  {continent}: {n}")


if __name__ == "__main__":
    main()
