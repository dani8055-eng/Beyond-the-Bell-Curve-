"""
Ingest IMF World Economic Outlook (WEO) annual macro data.

Reads the WEO bulk file (e.g. WEOOct2024all.xls), filters to our sample
countries and the annual variables we need, reshapes wide->long (tidy),
restricts to the sample window, and marks IMF forecast values so they are
NOT mistaken for observed actuals.

Source file format notes (these trip most people up):
  - Despite the .xls extension, it is a TAB-separated text file.
  - Encoding is UTF-16 LE.
  - Numeric cells may be blank or "n/a"; year columns are strings.
  - "Estimates Start After" gives the last ACTUAL year per series; values in
    years AFTER that are IMF forecasts, not observations.

See: docs/research_decisions.md (Decision 006 - to be added).

INPUT  (data/raw/):  WEO bulk file, auto-detected
INPUT  (data/interim/): final_sample_countries.csv (our 112-country sample)
OUTPUT (data/interim/): weo_annual_long.csv  (tidy: iso3, year, variable, value, is_forecast)
"""

import pandas as pd
from pathlib import Path


# Annual variables we want, mapped to WEO subject codes.
# (Descriptor kept for readability / documentation.)
WEO_VARIABLES = {
    'NGDP_RPCH':    'gdp_growth_pct',            # Real GDP growth, % change
    'PCPIPCH':      'inflation_avg_pct',         # Inflation, avg CPI, % change
    'PCPIEPCH':     'inflation_eop_pct',         # Inflation, end-of-period CPI, % change
    'BCA_NGDPD':    'current_account_pct_gdp',   # Current account, % of GDP
    'GGXCNL_NGDP':  'fiscal_balance_pct_gdp',    # Govt net lending/borrowing, % of GDP
    'GGXWDG_NGDP':  'govt_gross_debt_pct_gdp',   # Govt gross debt, % of GDP
    'NID_NGDP':     'investment_pct_gdp',        # Total investment, % of GDP
    'NGSD_NGDP':    'gross_savings_pct_gdp',     # Gross national savings, % of GDP
    'TX_RPCH':      'exports_vol_pct',           # Export volume, % change
    'TM_RPCH':      'imports_vol_pct',           # Import volume, % change
    'NGDPD':        'gdp_usd_bn',                # GDP, current US$ (billions)
}

YEAR_START = 1990
YEAR_END = 2016


def find_weo_file(raw_dir: str = "data/raw") -> Path:
    """Auto-detect the WEO bulk file in data/raw/ by its distinctive columns."""
    raw = Path(raw_dir)
    for cand in list(raw.glob("*.xls")) + list(raw.glob("*.xlsx")) + list(raw.glob("*.txt")):
        try:
            head = pd.read_csv(cand, sep='\t', encoding='utf-16-le', nrows=1)
            if 'WEO Subject Code' in head.columns and 'ISO' in head.columns:
                return cand
        except Exception:
            continue
    raise FileNotFoundError(
        f"No WEO bulk file found in {raw_dir}/. Expected a tab-separated, "
        "UTF-16-LE file with columns 'WEO Subject Code' and 'ISO'."
    )


def load_weo(weo_path: Path) -> pd.DataFrame:
    """Load the raw WEO file with the correct format settings."""
    df = pd.read_csv(weo_path, sep='\t', encoding='utf-16-le', low_memory=False)
    return df


def clean_value(x):
    """Convert a WEO numeric cell to float. Blanks / 'n/a' -> NaN."""
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    if s in ('', 'n/a', '--', 'NaN'):
        return pd.NA
    # WEO uses commas as thousands separators in some large-value series
    s = s.replace(',', '')
    try:
        return float(s)
    except ValueError:
        return pd.NA


def build_long(df: pd.DataFrame, sample_iso: set) -> pd.DataFrame:
    """
    Filter to sample countries + wanted variables, reshape to tidy long format,
    restrict to the year window, and flag forecast values.
    """
    year_cols = [str(y) for y in range(YEAR_START, YEAR_END + 1)]

    # Filter rows: our countries, our variables
    sub = df[df['ISO'].isin(sample_iso) & df['WEO Subject Code'].isin(WEO_VARIABLES)].copy()

    keep_cols = ['ISO', 'WEO Subject Code', 'Estimates Start After'] + year_cols
    sub = sub[keep_cols]

    # Melt wide -> long
    long = sub.melt(
        id_vars=['ISO', 'WEO Subject Code', 'Estimates Start After'],
        value_vars=year_cols,
        var_name='year',
        value_name='value_raw',
    )

    long['year'] = long['year'].astype(int)
    long['value'] = long['value_raw'].map(clean_value)
    long = long.drop(columns='value_raw')

    # Flag forecasts: value is a forecast if year > Estimates Start After
    def is_fc(row):
        esa = row['Estimates Start After']
        if pd.isna(esa):
            return False
        try:
            return row['year'] > int(esa)
        except (ValueError, TypeError):
            return False

    long['is_forecast'] = long.apply(is_fc, axis=1)

    # Rename columns to clean names
    long = long.rename(columns={'ISO': 'iso3', 'WEO Subject Code': 'weo_code'})
    long['variable'] = long['weo_code'].map(WEO_VARIABLES)

    long = long[['iso3', 'year', 'variable', 'weo_code', 'value', 'is_forecast']]
    long = long.sort_values(['iso3', 'variable', 'year']).reset_index(drop=True)
    return long


def main(raw_dir: str = "data/raw", interim_dir: str = "data/interim") -> None:
    Path(interim_dir).mkdir(parents=True, exist_ok=True)

    weo_path = find_weo_file(raw_dir)
    print(f"Using WEO file: {weo_path}")

    df = load_weo(weo_path)
    print(f"Raw WEO rows: {len(df)}")

    sample = pd.read_csv(Path(interim_dir) / "final_sample_countries.csv")
    sample_iso = set(sample['iso3'].dropna())
    print(f"Sample countries: {len(sample_iso)}")

    long = build_long(df, sample_iso)

    # Report
    n_countries = long['iso3'].nunique()
    n_vars = long['variable'].nunique()
    n_forecast = int(long['is_forecast'].sum())
    n_actual_nonnull = int((~long['is_forecast'] & long['value'].notna()).sum())
    print(f"\nTidy dataset: {len(long)} rows")
    print(f"  Countries covered: {n_countries}")
    print(f"  Variables: {n_vars}")
    print(f"  Non-null ACTUAL observations: {n_actual_nonnull}")
    print(f"  Forecast-flagged rows (1990-2016, rare): {n_forecast}")

    out = Path(interim_dir) / "weo_annual_long.csv"
    long.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Quick missingness snapshot per variable
    print("\nNon-null ACTUAL coverage by variable (of possible country-years):")
    total_cells = n_countries * (YEAR_END - YEAR_START + 1)
    for var in sorted(long['variable'].unique()):
        v = long[(long['variable'] == var) & (~long['is_forecast'])]
        nn = int(v['value'].notna().sum())
        pct = 100 * nn / total_cells
        print(f"  {var:26} {nn:5d} / {total_cells}  ({pct:.0f}%)")


if __name__ == "__main__":
    main()
