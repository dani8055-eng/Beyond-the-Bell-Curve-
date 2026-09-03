"""
Ingest monthly exchange-rate data (domestic currency per USD).

Source: codeforIATI/imf-exchangerates — a nightly scrape of IMF International
Financial Statistics (IFS) exchange rates, published as a consolidated CSV.
Original source is IMF IFS (https://data.imf.org/en/datasets/IMF.STA:ER);
attribution: "Source: International Monetary Fund".

We use this mirror because the IMF SDMX API was non-functional when tested
(2026-09-03): new host returned HTTP 403, legacy host no longer resolves.
The mirror is IMF data, properly attributed, and documented here for
reproducibility (see docs/research_decisions.md, Decision 008).

This module:
  - reads the raw consolidated CSV from data/raw/
  - maps ISO2 country codes -> ISO3 (to match our sample)
  - patches Namibia (reported under ZAR with a blank code; rand-pegged)
  - filters to our sample countries and the 1990-2016 window
  - saves a tidy monthly exchange-rate dataset

KNOWN LIMITATIONS (documented, not hidden):
  - Iran (IRN) and Mauritania (MRT): no usable historical data in this source
    for 1990-2016. They will lack an exchange-rate-based target.
  - Currency redenominations (Angola, Brazil, Argentina, Turkey, DRC, etc.)
    create huge level breaks in the raw series. This is REAL and must be
    handled at the feature stage (depreciation across a redenomination is not
    a real market move). Not corrected here.

INPUT  (data/raw/):     imf_exchangerates_raw.csv  (the downloaded mirror)
INPUT  (data/interim/): final_sample_countries.csv
OUTPUT (data/interim/): exchange_rates_monthly.csv  (iso3, date, year, month, exchange_rate, currency)

Requires: pip install pycountry
"""

import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
INTERIM = Path("data/interim")

YEAR_START = 1990
YEAR_END = 2016
RAW_FX_FILE = "imf_exchangerates_raw.csv"


def iso2_to_iso3(iso2):
    """Map a 2-letter country code to 3-letter, via pycountry."""
    import pycountry
    if pd.isna(iso2):
        return None
    try:
        return pycountry.countries.get(alpha_2=iso2).alpha_3
    except Exception:
        return None


def main():
    INTERIM.mkdir(parents=True, exist_ok=True)

    fx_path = RAW / RAW_FX_FILE
    if not fx_path.exists():
        raise FileNotFoundError(
            f"{fx_path} not found. Download the consolidated CSV from "
            "https://raw.githubusercontent.com/codeforIATI/imf-exchangerates/"
            "gh-pages/imf_exchangerates.csv and save it there."
        )

    fx = pd.read_csv(fx_path)
    print(f"Raw exchange-rate rows: {len(fx):,}")

    # Map country codes
    fx['iso3'] = fx['Country code'].map(iso2_to_iso3)
    # Patch Namibia (blank code, reported under ZAR, rand-pegged)
    fx.loc[fx['Country'] == 'Namibia', 'iso3'] = 'NAM'

    # Dates
    fx['Date'] = pd.to_datetime(fx['Date'])
    fx['year'] = fx['Date'].dt.year
    fx['month'] = fx['Date'].dt.month

    # Filter to sample + window
    sample = pd.read_csv(INTERIM / "final_sample_countries.csv")
    sample_iso3 = set(sample['iso3'].dropna())

    keep = fx[
        fx['iso3'].isin(sample_iso3)
        & (fx['year'] >= YEAR_START)
        & (fx['year'] <= YEAR_END)
    ].copy()

    out = keep[['iso3', 'Date', 'year', 'month', 'Rate', 'Currency']].rename(
        columns={'Date': 'date', 'Rate': 'exchange_rate', 'Currency': 'currency'}
    )
    out = out.sort_values(['iso3', 'date']).reset_index(drop=True)

    out.to_csv(INTERIM / "exchange_rates_monthly.csv", index=False)

    # Report
    covered = set(out['iso3'].unique())
    missing = sample_iso3 - covered
    iso_to_name = dict(zip(sample['iso3'], sample['lv_name']))
    print(f"Countries covered: {len(covered)} of {len(sample_iso3)}")
    print(f"Monthly observations: {len(out):,}")
    print(f"Date range: {out['date'].min().date()} to {out['date'].max().date()}")
    print(f"\nSample countries WITHOUT exchange-rate data ({len(missing)}):")
    for iso in sorted(missing):
        print(f"  {iso}: {iso_to_name.get(iso, '?')}")
    print(f"\nSaved: {INTERIM / 'exchange_rates_monthly.csv'}")


if __name__ == "__main__":
    main()
