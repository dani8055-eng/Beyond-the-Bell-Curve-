"""
Pull monthly FX reserves (Reserves Excluding Gold, USD) from FRED.

v3: correct query + title match, based on diagnostic of FRED's actual responses.

FRED titles these series "Reserves Excluding Gold for <Country>" (NOT "Total
Reserves..."), USD version = id ending M052N, SDR version = M194N. We search
"reserves <country>", then pick the USD monthly reserves-excl-gold series.

HONEST COVERAGE NOTE: FRED carries these series for major/mid emerging economies
but NOT for many smaller countries (e.g. Angola has none). Expect coverage well
below the full 112 -- this is a real limitation of the source, reported honestly.

Source: FRED (Federal Reserve Bank of St. Louis), republishing IMF IFS.
API key: .fred_api_key in project root (git-ignored).

INPUT:  data/interim/final_sample_countries.csv
OUTPUT: data/raw/fred_reserves_raw.csv  (iso3, date, reserves_usd_mn, series_id)
"""

import time
from pathlib import Path
import pandas as pd
import pycountry
from fredapi import Fred

INTERIM = Path("data/interim")
RAW = Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)


def load_key():
    return Path(".fred_api_key").read_text().strip()


def names_for(iso3):
    """Return candidate names FRED might use for this country."""
    try:
        c = pycountry.countries.get(alpha_3=iso3)
    except Exception:
        return []
    cands = []
    for attr in ('common_name', 'name', 'official_name'):
        v = getattr(c, attr, None)
        if v:
            cands.append(v)
    return list(dict.fromkeys(cands))  # dedupe, keep order


def find_usd_reserves_series(fred, names):
    """
    Search FRED for 'reserves <name>' and return the id of the USD monthly
    'Reserves Excluding Gold' series, or None.
    """
    for name in names:
        try:
            res = fred.search(f"reserves {name}")
        except Exception:
            continue
        if res is None or len(res) == 0:
            continue
        r = res.copy()
        r['title_l'] = r['title'].str.lower()
        # must be "reserves excluding gold" and this country
        r = r[r['title_l'].str.contains('reserves excluding gold', na=False)]
        if len(r) == 0:
            continue
        # prefer USD (id ends M052N) over SDR (M194N)
        usd = r[r['id'].str.endswith('M052N')]
        if len(usd) > 0:
            return usd.iloc[0]['id']
        # fall back to any monthly reserves-excl-gold if no M052N
        if len(r) > 0:
            return r.iloc[0]['id']
    return None


def main():
    fred = Fred(api_key=load_key())
    sample = pd.read_csv(INTERIM / "final_sample_countries.csv")
    iso3_list = sorted(sample['iso3'].dropna().unique())
    print(f"Searching FRED reserves for {len(iso3_list)} countries...\n")

    all_rows, got, missing = [], [], []
    for iso3 in iso3_list:
        names = names_for(iso3)
        if not names:
            missing.append(iso3); continue
        try:
            sid = find_usd_reserves_series(fred, names)
            if sid is None:
                missing.append(iso3); time.sleep(0.2); continue
            s = fred.get_series(sid)
            if s is None or len(s.dropna()) == 0:
                missing.append(iso3); time.sleep(0.2); continue
            df = s.dropna().reset_index()
            df.columns = ['date', 'reserves_usd_mn']
            df['iso3'] = iso3; df['series_id'] = sid
            all_rows.append(df[['iso3', 'date', 'reserves_usd_mn', 'series_id']])
            got.append(iso3)
            print(f"  {iso3}: {sid} ({len(df)} obs, "
                  f"{df['date'].dt.year.min()}-{df['date'].dt.year.max()})")
        except Exception:
            missing.append(iso3)
        time.sleep(0.25)

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True).sort_values(['iso3', 'date'])
        out.to_csv(RAW / "fred_reserves_raw.csv", index=False)
        print(f"\nSaved: {RAW / 'fred_reserves_raw.csv'}  ({len(out):,} rows)")

    print(f"\nCoverage: {len(got)} of {len(iso3_list)} countries have reserves")
    print(f"Countries WITH reserves: {got}")
    print(f"\nMissing ({len(missing)}) -- not on FRED:")
    print("  " + ", ".join(missing))


if __name__ == "__main__":
    main()
