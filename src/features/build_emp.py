"""
Build the Exchange Market Pressure (EMP) crisis measure — EMP sub-study.

SCOPE: this runs only on the subsample of countries with monthly reserves data
from FRED (Argentina, Brazil, Indonesia, Korea, Mexico, Russia, Turkey, South
Africa). It is a self-contained sub-study (see docs/research_decisions.md,
Decision 017), separate from the main 110-country Laeven-Valencia analysis.

WHAT EMP IS:
  A currency can be under severe pressure without a large observed depreciation,
  because the central bank burns FX reserves to defend it. EMP captures both:
      EMP_t = (depreciation_t / sd_depreciation) - (reserve_growth_t / sd_reserve_growth)
  Depreciation UP -> pressure up; reserves DOWN -> pressure up. Each component is
  scaled by its own standard deviation so neither dominates by units alone.

  NOTE: the textbook EMP has three components (depreciation, reserves, interest
  rates). Monthly interest rates are unavailable, so this is the common
  TWO-COMPONENT version (depreciation + reserves), documented as such.

EMP CRISIS DEFINITION:
  A crisis month is one where EMP exceeds its country mean by k standard
  deviations. We build both k=1.5 and k=2.0 (standard Kaminsky-style thresholds).
  Thresholds are computed PER COUNTRY (each currency has its own normal range).

  Look-ahead note: the mean/sd here are full-sample (standard in the EMP crisis-
  DATING literature, as the crisis label is a definition, not a prediction). The
  PREDICTION step later still uses only backward-looking features and walk-forward
  validation, so the model never sees the future.

INPUT:  data/raw/fred_reserves_raw.csv          (iso3, date, reserves_usd_mn, series_id)
        data/interim/fx_depreciation_monthly.csv (depreciation)
OUTPUT: data/interim/emp_measure.csv             (iso3, date, emp, emp_crisis_15, emp_crisis_20)
"""

import numpy as np
import pandas as pd
from pathlib import Path

INTERIM = Path("data/interim")
RAW = Path("data/raw")

SUBSAMPLE = ['ARG', 'BRA', 'IDN', 'KOR', 'MEX', 'RUS', 'TUR', 'ZAF']
YEAR_START, YEAR_END = 1990, 2016


def main():
    # --- Reserves ---
    res = pd.read_csv(RAW / "fred_reserves_raw.csv")
    res['date'] = pd.to_datetime(res['date'])
    res = res[res['iso3'].isin(SUBSAMPLE)].copy()
    # month-end align to match fx dates (fx dates are month-end)
    res['date'] = res['date'] + pd.offsets.MonthEnd(0)
    res = res[['iso3', 'date', 'reserves_usd_mn']]

    # reserve growth (month-over-month % change)
    res = res.sort_values(['iso3', 'date'])
    res['reserve_growth'] = res.groupby('iso3')['reserves_usd_mn'].pct_change()

    # --- Depreciation ---
    dep = pd.read_csv(INTERIM / "fx_depreciation_monthly.csv")
    dep['date'] = pd.to_datetime(dep['date'])
    dep = dep[dep['iso3'].isin(SUBSAMPLE)][['iso3', 'date', 'depreciation']]

    # --- Merge on iso3+date ---
    df = dep.merge(res[['iso3', 'date', 'reserve_growth']], on=['iso3', 'date'], how='inner')
    df = df[(df['date'].dt.year >= YEAR_START) & (df['date'].dt.year <= YEAR_END)]
    df = df.dropna(subset=['depreciation', 'reserve_growth']).reset_index(drop=True)

    # --- Build EMP per country (scale each component by its country sd) ---
    parts = []
    for iso, g in df.groupby('iso3'):
        g = g.copy()
        sd_dep = g['depreciation'].std()
        sd_res = g['reserve_growth'].std()
        if sd_dep == 0 or pd.isna(sd_dep) or sd_res == 0 or pd.isna(sd_res):
            continue
        g['emp'] = (g['depreciation'] / sd_dep) - (g['reserve_growth'] / sd_res)
        m, s = g['emp'].mean(), g['emp'].std()
        g['emp_crisis_15'] = (g['emp'] > m + 1.5 * s).astype(int)
        g['emp_crisis_20'] = (g['emp'] > m + 2.0 * s).astype(int)
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    out = out[['iso3', 'date', 'depreciation', 'reserve_growth',
               'emp', 'emp_crisis_15', 'emp_crisis_20']]
    out = out.sort_values(['iso3', 'date']).reset_index(drop=True)
    out.to_csv(INTERIM / "emp_measure.csv", index=False)

    # --- Report ---
    print(f"Saved: {INTERIM / 'emp_measure.csv'}  ({len(out):,} country-months)")
    print(f"Countries: {sorted(out['iso3'].unique())}")
    print(f"\nEMP crisis months (k=1.5): {out['emp_crisis_15'].sum()} "
          f"({100*out['emp_crisis_15'].mean():.1f}%)")
    print(f"EMP crisis months (k=2.0): {out['emp_crisis_20'].sum()} "
          f"({100*out['emp_crisis_20'].mean():.1f}%)")
    print("\nEMP crises by country (k=1.5):")
    print(out.groupby('iso3')['emp_crisis_15'].sum().to_string())


if __name__ == "__main__":
    main()
