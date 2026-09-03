"""
Build the monthly six-month-ahead currency-crisis target (primary, Option A).

Implements Decision 003 on the monthly panel grid:
  If a country has an LV currency crisis in year Y, then every month in year
  (Y-1) is labelled target = 1 ("a crisis occurs within the following window").
  All other country-months are target = 0.

The panel grid is defined by the monthly exchange-rate data (the country-months
we actually observe), so the target lines up row-for-row with the modelling data.

IMPORTANT — honest sample note (see docs/research_decisions.md, Decision 009):
  Of 227 in-sample LV crises, only ~118 land inside the 1990-2016 window as
  usable positive labels. The rest have target years before 1990 (a direct,
  expected consequence of the 1990 panel start in Decision 005) and cannot be
  labelled because we have no data for those months. This is not a bug; it is
  the honest effective crisis count for the chosen window.

This is the PRIMARY target. A monthly EMP-based target (using exchange rates +
reserves) is planned as a separate robustness definition once reserves are
sourced.

INPUT  (data/interim/):
  - lv_currency_crises_parsed.csv   (country, crisis_year)
  - final_sample_countries.csv      (lv_name, oghist_name, iso3)
  - exchange_rates_monthly.csv      (iso3, date, year, month, ...)  -> panel grid
OUTPUT (data/interim/):
  - monthly_target.csv              (iso3, date, year, month, target)
"""

import pandas as pd
from pathlib import Path

INTERIM = Path("data/interim")


def main():
    crises = pd.read_csv(INTERIM / "lv_currency_crises_parsed.csv")
    sample = pd.read_csv(INTERIM / "final_sample_countries.csv")
    fx = pd.read_csv(INTERIM / "exchange_rates_monthly.csv")

    # Map LV crisis country names -> iso3 via the sample crosswalk
    name_to_iso = dict(zip(sample['lv_name'], sample['iso3']))
    crises = crises.copy()
    crises['iso3'] = crises['country'].map(name_to_iso)
    crises_in = crises[crises['iso3'].notna()].copy()

    # Decision 003: crisis in year Y -> positive label for all months of Y-1
    target_years = set(zip(crises_in['iso3'], crises_in['crisis_year'] - 1))

    # Apply on the monthly panel grid defined by the exchange-rate data
    panel = fx[['iso3', 'date', 'year', 'month']].copy()
    panel['target'] = [
        1 if (i, y) in target_years else 0
        for i, y in zip(panel['iso3'], panel['year'])
    ]
    panel = panel.sort_values(['iso3', 'date']).reset_index(drop=True)

    panel.to_csv(INTERIM / "monthly_target.csv", index=False)

    # Honest reporting
    captured = target_years & set(zip(panel['iso3'], panel['year']))
    n_pos = int(panel['target'].sum())
    n_tot = len(panel)
    countries_with_pos = panel[panel['target'] == 1]['iso3'].nunique()

    print(f"Saved: {INTERIM / 'monthly_target.csv'}")
    print(f"Monthly rows: {n_tot:,}")
    print(f"Positive rows (target=1): {n_pos:,} ({100*n_pos/n_tot:.2f}%)")
    print(f"Negative rows: {n_tot - n_pos:,}")
    print(f"Imbalance: ~1 crisis-month per {n_tot // n_pos} non-crisis-months")
    print(f"Distinct crisis episodes captured in window: {len(captured)} "
          f"(of {len(target_years)} in-sample crisis target-years)")
    print(f"Countries with >=1 positive label: {countries_with_pos} "
          f"of {panel['iso3'].nunique()}")


if __name__ == "__main__":
    main()
