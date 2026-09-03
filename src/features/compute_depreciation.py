"""
Compute clean monthly exchange-rate depreciation, handling structural breaks.

This is the FIRST feature-engineering step. It produces the monthly depreciation
series that all downstream exchange-rate features (volatility, extreme moves,
tail measures) and the EMP robustness measure will build on.

STRUCTURAL-BREAK HANDLING (Method 1 — neutralize artifacts, keep real moves):
  A currency redenomination (dropping zeros) or dollarization (adopting the USD)
  produces a SUDDEN STRENGTHENING of the exchange rate — the rate is divided by a
  large factor in a single month. This is an ACCOUNTING artifact, not a market
  move, so the naive percent-change at that month is meaningless and would poison
  volatility / skewness / kurtosis / tail features.

  Rule: at any month where the rate falls to < 20% of the previous month
  (ratio < 0.2 = a >5x sudden strengthening), set depreciation = NaN for that
  ONE month. This removes the artifact while preserving every genuine move.

  CRUCIAL: large WEAKENING jumps (ratio > 1) are NOT touched — those are real
  currency crises / hyperinflations (e.g. Zimbabwe 2007-2008), which are exactly
  the tail events this project studies. We keep them fully intact.

  Verified artifact months (2026-09-03): Ecuador 2000-01 (dollarization),
  El Salvador 2001-01 (dollarization), Zimbabwe 2008-08 (redenomination).
  Only 3 months across 110 countries are neutralized.

  Why NaN rather than splicing the levels: dollarization has no clean rescale
  factor (the currency ceases to exist), and a slightly-wrong splice fabricates
  a depreciation number that never happened — worse than a missing value. For
  the model we need CHANGES, not continuous LEVELS, and the real moves around the
  break are already captured. (If continuous level charts are wanted for the
  presentation layer later, splice for VISUALIZATION ONLY, never into the model.)

See: docs/research_decisions.md (Decision 010).

INPUT  (data/interim/): exchange_rates_monthly.csv
OUTPUT (data/interim/): fx_depreciation_monthly.csv
                        (iso3, date, year, month, exchange_rate, depreciation,
                         log_depreciation, is_structural_break)
"""

import numpy as np
import pandas as pd
from pathlib import Path

INTERIM = Path("data/interim")

# A month where the rate falls below this fraction of the prior month is treated
# as a structural-break artifact (redenomination / dollarization).
ARTIFACT_RATIO_THRESHOLD = 0.2


def main():
    fx = pd.read_csv(INTERIM / "exchange_rates_monthly.csv")
    fx['date'] = pd.to_datetime(fx['date'])
    fx = fx.sort_values(['iso3', 'date']).reset_index(drop=True)

    # Month-over-month ratio and percent change
    fx['prev_rate'] = fx.groupby('iso3')['exchange_rate'].shift(1)
    fx['ratio'] = fx['exchange_rate'] / fx['prev_rate']
    fx['depreciation'] = fx.groupby('iso3')['exchange_rate'].pct_change()

    # Identify artifact break-months (sudden strengthening only)
    fx['is_structural_break'] = fx['ratio'] < ARTIFACT_RATIO_THRESHOLD

    # Neutralize depreciation at those months only
    fx.loc[fx['is_structural_break'], 'depreciation'] = np.nan

    # Log-depreciation (more stable for very large moves); also NaN at breaks
    # and undefined where rate/prev_rate <= 0 (shouldn't happen, but guard).
    with np.errstate(divide='ignore', invalid='ignore'):
        fx['log_depreciation'] = np.log(fx['exchange_rate'] / fx['prev_rate'])
    fx.loc[fx['is_structural_break'], 'log_depreciation'] = np.nan

    out = fx[['iso3', 'date', 'year', 'month', 'exchange_rate',
              'depreciation', 'log_depreciation', 'is_structural_break']].copy()
    out.to_csv(INTERIM / "fx_depreciation_monthly.csv", index=False)

    # Report
    breaks = fx[fx['is_structural_break']]
    print(f"Saved: {INTERIM / 'fx_depreciation_monthly.csv'}")
    print(f"Rows: {len(out):,}")
    print(f"Structural-break months neutralized: {len(breaks)}")
    for _, r in breaks.sort_values(['iso3', 'date']).iterrows():
        print(f"  {r['iso3']} {r['date'].date()}: "
              f"{r['prev_rate']:.4g} -> {r['exchange_rate']:.4g} "
              f"(ratio {r['ratio']:.2e})")
    print(f"\nDepreciation now available for "
          f"{out['depreciation'].notna().sum():,} country-months.")


if __name__ == "__main__":
    main()
