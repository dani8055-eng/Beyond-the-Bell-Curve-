"""
Build monthly exchange-rate features (conventional + tail-aware).

Built on the CLEANED depreciation series (Decision 010), so redenomination
artifacts do not corrupt any rolling statistic.

ALL features are BACKWARD-LOOKING rolling windows: a feature at month t uses only
depreciation values up to and including month t. No future information enters any
feature (look-ahead rule / Rule 4). This is what makes the eventual out-of-sample
evaluation credible.

Two explicitly-labelled groups (see CONVENTIONAL_FEATURES / TAIL_FEATURES below),
because the project's central experiment is "conventional vs conventional+tail".

Windows: 3, 6, 12 months. Distribution-shape stats (skew, kurtosis, Expected
Shortfall) use only the 6- and 12-month windows (need enough points to be
meaningful).

Feature list:
  Conventional:
    dep_1m                     current 1-month depreciation
    dep_mean_{3,6,12}m         rolling mean depreciation
    dep_vol_{3,6,12}m          rolling volatility (std)
    dep_cum_{3,6,12}m          cumulative (compounded) depreciation over window
  Tail-aware:
    dep_skew_{6,12}m           rolling skewness
    dep_kurt_{6,12}m           rolling excess kurtosis
    dep_max_{6,12}m            worst (max) monthly depreciation in window
    dep_es90_{6,12}m           Expected Shortfall: mean of worst 10% of months
    extreme_count_{3,6,12}m    count of months with depreciation > 10%

See: docs/research_decisions.md (Decision 011).

INPUT  (data/interim/): fx_depreciation_monthly.csv
OUTPUT (data/interim/): fx_features.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

INTERIM = Path("data/interim")

WINDOWS = [3, 6, 12]
TAIL_WINDOWS = [6, 12]      # skew/kurt/ES need more points
EXTREME_THRESHOLD = 0.10    # a monthly depreciation > 10% counts as "extreme"
ES_QUANTILE = 0.90          # Expected Shortfall over worst 10%

# Explicit group membership (used by the conventional-vs-tail experiment later)
CONVENTIONAL_FEATURES = (
    ['dep_1m']
    + [f'dep_mean_{w}m' for w in WINDOWS]
    + [f'dep_vol_{w}m' for w in WINDOWS]
    + [f'dep_cum_{w}m' for w in WINDOWS]
)
TAIL_FEATURES = (
    [f'dep_skew_{w}m' for w in TAIL_WINDOWS]
    + [f'dep_kurt_{w}m' for w in TAIL_WINDOWS]
    + [f'dep_max_{w}m' for w in TAIL_WINDOWS]
    + [f'dep_es90_{w}m' for w in TAIL_WINDOWS]
    + [f'extreme_count_{w}m' for w in WINDOWS]
)


def expected_shortfall(x, q=ES_QUANTILE):
    """Mean of the worst (1-q) tail (large depreciations = the dangerous side)."""
    if len(x) < 3:
        return np.nan
    thresh = np.quantile(x, q)
    tail = x[x >= thresh]
    return tail.mean() if len(tail) else np.nan


def main():
    dep = pd.read_csv(INTERIM / "fx_depreciation_monthly.csv")
    dep['date'] = pd.to_datetime(dep['date'])
    dep = dep.sort_values(['iso3', 'date']).reset_index(drop=True)

    g = dep.groupby('iso3', group_keys=False)
    feat = dep[['iso3', 'date', 'year', 'month']].copy()

    # ---- Conventional ----
    feat['dep_1m'] = dep['depreciation']
    for w in WINDOWS:
        mp = max(2, w // 2)
        feat[f'dep_mean_{w}m'] = g['depreciation'].apply(
            lambda s: s.rolling(w, min_periods=mp).mean())
        feat[f'dep_vol_{w}m'] = g['depreciation'].apply(
            lambda s: s.rolling(w, min_periods=mp).std())
        feat[f'dep_cum_{w}m'] = g['depreciation'].apply(
            lambda s: s.rolling(w, min_periods=mp).apply(
                lambda x: np.prod(1 + x) - 1, raw=True))

    # ---- Tail-aware ----
    for w in TAIL_WINDOWS:
        feat[f'dep_skew_{w}m'] = g['depreciation'].apply(
            lambda s: s.rolling(w, min_periods=max(3, w // 2)).skew())
        feat[f'dep_kurt_{w}m'] = g['depreciation'].apply(
            lambda s: s.rolling(w, min_periods=max(4, w // 2)).kurt())
        feat[f'dep_max_{w}m'] = g['depreciation'].apply(
            lambda s: s.rolling(w, min_periods=max(2, w // 2)).max())
        feat[f'dep_es90_{w}m'] = g['depreciation'].apply(
            lambda s: s.rolling(w, min_periods=max(3, w // 2)).apply(
                lambda x: expected_shortfall(x), raw=True))

    dep['is_extreme'] = (dep['depreciation'] > EXTREME_THRESHOLD).astype(float)
    ge = dep.groupby('iso3', group_keys=False)['is_extreme']
    for w in WINDOWS:
        feat[f'extreme_count_{w}m'] = ge.apply(
            lambda s: s.rolling(w, min_periods=1).sum())

    feat.to_csv(INTERIM / "fx_features.csv", index=False)

    print(f"Saved: {INTERIM / 'fx_features.csv'}")
    print(f"Rows: {len(feat):,}")
    print(f"Conventional features ({len(CONVENTIONAL_FEATURES)}): {CONVENTIONAL_FEATURES}")
    print(f"Tail features ({len(TAIL_FEATURES)}): {TAIL_FEATURES}")
    print(f"Total features: {len(CONVENTIONAL_FEATURES) + len(TAIL_FEATURES)}")


if __name__ == "__main__":
    main()
