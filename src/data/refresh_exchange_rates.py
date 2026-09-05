"""
CHUNK 1 (extend-to-present): refresh monthly exchange rates through 2023.

Re-downloads the codeforIATI/imf-exchangerates mirror (nightly-updated IMF IFS
exchange rates) and rebuilds the monthly exchange-rate file for the SAME sample
countries, but through 2023 instead of 2016.

This mirrors src/data/ingest_exchange_rates.py, just with a later end year, and
writes to a NEW file so the original 2016 study is untouched.

Requires: pip install pycountry  (already installed)

INPUT:  data/interim/final_sample_countries.csv
OUTPUT: data/raw/imf_exchangerates_raw_2023.csv   (the fresh download)
        data/interim/exchange_rates_monthly_2023.csv
"""

import io
import urllib.request
from pathlib import Path
import pandas as pd
import pycountry

INTERIM = Path("data/interim")
RAW = Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)

MIRROR_URL = ("https://raw.githubusercontent.com/codeforIATI/imf-exchangerates/"
              "gh-pages/imf_exchangerates.csv")
YEAR_START, YEAR_END = 1990, 2023


def iso2_to_iso3(iso2):
    if pd.isna(iso2):
        return None
    try:
        return pycountry.countries.get(alpha_2=iso2).alpha_3
    except Exception:
        return None


def main():
    print("Downloading the latest exchange-rate mirror...", flush=True)
    req = urllib.request.Request(MIRROR_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    (RAW / "imf_exchangerates_raw_2023.csv").write_bytes(raw)
    fx = pd.read_csv(io.BytesIO(raw))
    print(f"Downloaded {len(raw)/1e6:.1f} MB, {len(fx):,} rows", flush=True)
    print(f"Mirror date range: {fx['Date'].min()} to {fx['Date'].max()}", flush=True)

    fx['iso3'] = fx['Country code'].map(iso2_to_iso3)
    fx.loc[fx['Country'] == 'Namibia', 'iso3'] = 'NAM'
    fx['Date'] = pd.to_datetime(fx['Date'])
    fx['year'] = fx['Date'].dt.year
    fx['month'] = fx['Date'].dt.month

    sample = pd.read_csv(INTERIM / "final_sample_countries.csv")
    sample_iso3 = set(sample['iso3'].dropna())

    keep = fx[fx['iso3'].isin(sample_iso3)
              & (fx['year'] >= YEAR_START) & (fx['year'] <= YEAR_END)].copy()
    out = keep[['iso3', 'Date', 'year', 'month', 'Rate', 'Currency']].rename(
        columns={'Date': 'date', 'Rate': 'exchange_rate', 'Currency': 'currency'})
    out = out.sort_values(['iso3', 'date']).reset_index(drop=True)
    out.to_csv(INTERIM / "exchange_rates_monthly_2023.csv", index=False)

    covered = set(out['iso3'].unique())
    print(f"\nSaved: {INTERIM / 'exchange_rates_monthly_2023.csv'}")
    print(f"Countries covered: {len(covered)} of {len(sample_iso3)}")
    print(f"Rows: {len(out):,}   Date range: {out['date'].min().date()} to {out['date'].max().date()}")

    # how many months does each country now have past 2016?
    recent = out[out['year'] > 2016]
    print(f"\nNew months added (2017-2023): {len(recent):,} rows across "
          f"{recent['iso3'].nunique()} countries")
    print("Sample of recent coverage (a few big countries):")
    for iso in ['ARG', 'TUR', 'BRA', 'ZAF', 'EGY']:
        d = out[out['iso3'] == iso]
        if len(d):
            print(f"  {iso}: through {d['date'].max().date()} ({len(d)} months total)")


if __name__ == "__main__":
    main()
