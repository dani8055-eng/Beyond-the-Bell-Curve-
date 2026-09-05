"""
CHUNK 2a (extend-to-present): compute depreciation + features on the 2023 data.

Direct extension of compute_depreciation.py and build_fx_features.py, reading the
extended exchange-rate file (through 2023) and writing new _2023 outputs. Logic
is identical (same structural-break handling, same 21 features) -- only the input
file and the output names change.

INPUT:  data/interim/exchange_rates_monthly_2023.csv
OUTPUT: data/interim/fx_depreciation_monthly_2023.csv
        data/interim/fx_features_2023.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

INTERIM = Path("data/interim")
ARTIFACT_RATIO_THRESHOLD = 0.2
WINDOWS = [3, 6, 12]
TAIL_WINDOWS = [6, 12]
EXTREME_THRESHOLD = 0.10
ES_QUANTILE = 0.90


def expected_shortfall(x, q=ES_QUANTILE):
    if len(x) < 3:
        return np.nan
    t = np.quantile(x, q)
    tail = x[x >= t]
    return tail.mean() if len(tail) else np.nan


def main():
    # --- depreciation with structural-break handling ---
    fx = pd.read_csv(INTERIM / "exchange_rates_monthly_2023.csv")
    fx['date'] = pd.to_datetime(fx['date'])
    fx = fx.sort_values(['iso3', 'date']).reset_index(drop=True)
    fx['prev_rate'] = fx.groupby('iso3')['exchange_rate'].shift(1)
    fx['ratio'] = fx['exchange_rate'] / fx['prev_rate']
    fx['depreciation'] = fx.groupby('iso3')['exchange_rate'].pct_change()
    fx['is_structural_break'] = fx['ratio'] < ARTIFACT_RATIO_THRESHOLD
    fx.loc[fx['is_structural_break'], 'depreciation'] = np.nan

    dep = fx[['iso3', 'date', 'year', 'month', 'exchange_rate',
              'depreciation', 'is_structural_break']].copy()
    dep.to_csv(INTERIM / "fx_depreciation_monthly_2023.csv", index=False)
    print(f"Depreciation: {len(dep):,} rows, "
          f"{int(dep['is_structural_break'].sum())} breaks neutralized")

    # --- features ---
    g = dep.groupby('iso3', group_keys=False)
    feat = dep[['iso3', 'date', 'year', 'month']].copy()
    feat['dep_1m'] = dep['depreciation']
    for w in WINDOWS:
        mp = max(2, w // 2)
        feat[f'dep_mean_{w}m'] = g['depreciation'].apply(lambda s: s.rolling(w, min_periods=mp).mean())
        feat[f'dep_vol_{w}m'] = g['depreciation'].apply(lambda s: s.rolling(w, min_periods=mp).std())
        feat[f'dep_cum_{w}m'] = g['depreciation'].apply(
            lambda s: s.rolling(w, min_periods=mp).apply(lambda x: np.prod(1 + x) - 1, raw=True))
    for w in TAIL_WINDOWS:
        feat[f'dep_skew_{w}m'] = g['depreciation'].apply(lambda s: s.rolling(w, min_periods=max(3, w // 2)).skew())
        feat[f'dep_kurt_{w}m'] = g['depreciation'].apply(lambda s: s.rolling(w, min_periods=max(4, w // 2)).kurt())
        feat[f'dep_max_{w}m'] = g['depreciation'].apply(lambda s: s.rolling(w, min_periods=max(2, w // 2)).max())
        feat[f'dep_es90_{w}m'] = g['depreciation'].apply(
            lambda s: s.rolling(w, min_periods=max(3, w // 2)).apply(lambda x: expected_shortfall(x), raw=True))
    dep['is_extreme'] = (dep['depreciation'] > EXTREME_THRESHOLD).astype(float)
    ge = dep.groupby('iso3', group_keys=False)['is_extreme']
    for w in WINDOWS:
        feat[f'extreme_count_{w}m'] = ge.apply(lambda s: s.rolling(w, min_periods=1).sum())

    feat.to_csv(INTERIM / "fx_features_2023.csv", index=False)
    print(f"Features: {len(feat):,} rows, {feat.shape[1]-4} features, "
          f"through {feat['date'].max().date()}")


if __name__ == "__main__":
    main()
