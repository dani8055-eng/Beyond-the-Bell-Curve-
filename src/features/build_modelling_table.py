"""
Assemble the single modelling table.

Joins three pieces into one country-month table (features + target):
  1. monthly_target.csv     - the six-month-ahead crisis target (Decision 009)
  2. fx_features.csv         - 21 exchange-rate features (Decision 011)
  3. weo_annual_long.csv     - 11 annual macro variables (Decision 007)

LOOK-AHEAD HANDLING (the critical part):
  Monthly FX features are real-time (known as they happen) -> no artificial lag.
  Annual WEO macro is published in arrears and revised, so it is LAGGED 2 YEARS:
  a year Y macro value only becomes visible to the model from calendar year Y+2
  onward. This guarantees that when predicting in year C, the model sees macro
  only up to year C-2 - never the current or prior year, both of which would not
  have been published/final in real time. (See Decision 012.)

  Verified: e.g. Argentina's 2001 crisis-year GDP growth is not visible in the
  table until 2003 rows. Cost: 1990-1991 rows have no macro (earliest macro is
  1990 -> usable 1992). This is the honest, intended cost of the lag.

Only ACTUAL WEO values are used (forecasts excluded; see Decision 007).

INPUT  (data/interim/): monthly_target.csv, fx_features.csv, weo_annual_long.csv
OUTPUT (data/interim/): modelling_table.csv
"""

import pandas as pd
from pathlib import Path

INTERIM = Path("data/interim")

MACRO_LAG_YEARS = 2


def main():
    target = pd.read_csv(INTERIM / "monthly_target.csv")
    fxf = pd.read_csv(INTERIM / "fx_features.csv")
    weo = pd.read_csv(INTERIM / "weo_annual_long.csv")

    target['date'] = pd.to_datetime(target['date'])
    fxf['date'] = pd.to_datetime(fxf['date'])

    # WEO long -> wide, actuals only
    weo_act = weo[~weo['is_forecast']].copy()
    weo_wide = weo_act.pivot_table(
        index=['iso3', 'year'], columns='variable', values='value'
    ).reset_index()

    # Apply the 2-year lag: a year-Y value becomes usable in year Y+MACRO_LAG_YEARS
    weo_wide['year'] = weo_wide['year'] + MACRO_LAG_YEARS
    macro_vars = [c for c in weo_wide.columns if c not in ('iso3', 'year')]
    weo_wide = weo_wide.rename(columns={c: f'macro_{c}' for c in macro_vars})

    # Merge: target + fx on (iso3, monthly date); + macro on (iso3, year)
    df = target.merge(fxf.drop(columns=['year', 'month']), on=['iso3', 'date'], how='left')
    df = df.merge(weo_wide, on=['iso3', 'year'], how='left')

    df = df.sort_values(['iso3', 'date']).reset_index(drop=True)
    df.to_csv(INTERIM / "modelling_table.csv", index=False)

    fx_cols = [c for c in df.columns if c.startswith('dep_') or c.startswith('extreme')]
    macro_cols = [c for c in df.columns if c.startswith('macro_')]

    print(f"Saved: {INTERIM / 'modelling_table.csv'}")
    print(f"Rows: {len(df):,}  Columns: {df.shape[1]}")
    print(f"FX features: {len(fx_cols)}  Macro features: {len(macro_cols)}")
    print(f"Target positive rate: {100*df['target'].mean():.2f}%")
    print(f"Rows with all FX features: {df[fx_cols].notna().all(axis=1).sum():,}")
    print(f"Rows with any macro (1992+ due to 2y lag): {df[macro_cols].notna().any(axis=1).sum():,}")
    print(f"Macro lag applied: {MACRO_LAG_YEARS} years")


if __name__ == "__main__":
    main()
