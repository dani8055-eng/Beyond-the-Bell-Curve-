"""
CHUNK 2b (extend-to-present): build the extended modelling table through 2023.

Extension of build_modelling_table.py. Joins the 2023 features with the target
and the 2-year-lagged WEO macro. Reuses the SAME WEO file (Oct 2024), which
already contains actuals through ~2023 -- so no new macro download is needed.

TARGET NOTE: the Laeven-Valencia crisis labels end in 2017, so a real six-month-
ahead crisis target only exists up to ~2016. For 2017-2023 we still build the
FEATURES and macro (so the model can PREDICT), but the target is left NaN there
(nothing to score against -- there is no published crisis list for those years).
This is the honest "features available, answer key unavailable" situation.

INPUT:  data/interim/fx_features_2023.csv
        data/interim/monthly_target.csv      (LV target, <=2016)
        data/interim/weo_annual_long.csv      (macro, reused)
OUTPUT: data/interim/modelling_table_2023.csv
"""

import pandas as pd
from pathlib import Path

INTERIM = Path("data/interim")
MACRO_LAG_YEARS = 2


def main():
    fxf = pd.read_csv(INTERIM / "fx_features_2023.csv")
    fxf['date'] = pd.to_datetime(fxf['date'])

    # target only exists to 2016 (LV labels); left NaN for later years
    tgt = pd.read_csv(INTERIM / "monthly_target.csv")
    tgt['date'] = pd.to_datetime(tgt['date'])
    tgt = tgt[['iso3', 'date', 'target']]

    weo = pd.read_csv(INTERIM / "weo_annual_long.csv")
    weo_act = weo[~weo['is_forecast']].copy()
    weo_wide = weo_act.pivot_table(index=['iso3', 'year'], columns='variable',
                                   values='value').reset_index()
    weo_wide['year'] = weo_wide['year'] + MACRO_LAG_YEARS
    macro_vars = [c for c in weo_wide.columns if c not in ('iso3', 'year')]
    weo_wide = weo_wide.rename(columns={c: f'macro_{c}' for c in macro_vars})

    df = fxf.merge(tgt, on=['iso3', 'date'], how='left')
    df = df.merge(weo_wide, on=['iso3', 'year'], how='left')
    df = df.sort_values(['iso3', 'date']).reset_index(drop=True)
    df.to_csv(INTERIM / "modelling_table_2023.csv", index=False)

    fx_cols = [c for c in df.columns if c.startswith('dep_') or c.startswith('extreme')]
    macro_cols = [c for c in df.columns if c.startswith('macro_')]
    has_target = df['target'].notna().sum()
    recent = df[df['year'] > 2016]

    print(f"Saved: {INTERIM / 'modelling_table_2023.csv'}")
    print(f"Rows: {len(df):,}  through {df['date'].max().date()}")
    print(f"Rows WITH a crisis label (<=2016): {has_target:,}")
    print(f"Rows in 2017-2023 (features only, for prediction): {len(recent):,}")
    print(f"  of those, with all FX features present: {recent[fx_cols].notna().all(axis=1).sum():,}")
    print(f"  of those, with macro present: {recent[macro_cols].notna().any(axis=1).sum():,}")


if __name__ == "__main__":
    main()
