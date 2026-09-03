"""
Build the emerging-market modeling sample.

Combines:
  1. Laeven-Valencia currency crisis countries (from parse_lv_crises.py output)
  2. World Bank OGHIST historical income classification

Applies Decision 005: keep any country classified NON-high-income at any point
in 1990-2016 ("ever non-high-income"), then intersect with LV crisis countries.

See: docs/research_decisions.md (Decision 005).

INPUTS (in data/raw/ and data/interim/):
  - OGHIST Excel file (auto-detected by sheet name "Country Analytical History")
  - data/interim/lv_currency_crises_parsed.csv (from parse_lv_crises.py)

OUTPUTS (data/interim/):
  - em_country_list_oghist.csv       : all ever-non-high-income economies
  - final_sample_countries.csv       : EM countries that also had an LV crisis
"""

import pandas as pd
from pathlib import Path
from typing import Optional


# Verified manual crosswalk: LV country name -> OGHIST country name.
# These are the names that do not match exactly between the two sources.
# Each mapping was verified by hand (see Decision 005 / notebook).
LV_TO_OGHIST = {
    'Central African Rep.': 'Central African Republic',
    'Congo, Dem. Rep. of': 'Congo, Dem. Rep.',
    'Congo, Rep. of': 'Congo, Rep.',
    'Côte d’Ivoire': "Côte d'Ivoire",
    'Egypt': 'Egypt, Arab Rep.',
    'Iran, I.R. of': 'Iran, Islamic Rep.',
    'Korea': 'Korea, Rep.',            # South Korea (NOT Korea, Dem. Rep.)
    'Lao People’s Dem. Rep.': 'Lao PDR',
    'Russia': 'Russian Federation',
    'Serbia, Republic of': 'Serbia',   # NOT "Serbia and Montenegro (former)"
    'Swaziland': 'Eswatini',
    'São Tomé and Principe': 'São Tomé and Príncipe',
    'Turkey': 'Türkiye',               # NOT Turkmenistan
    'Venezuela': 'Venezuela, RB',
    'Vietnam': 'Viet Nam',
    'Yemen': 'Yemen, Rep.',
}

# Non-high-income codes in OGHIST
NON_HIGH_CODES = {'L', 'LM', 'UM'}
ALL_CODES = {'L', 'LM', 'UM', 'H'}

# Sample window (Decision 005: 1990 start)
YEAR_START = 1990
YEAR_END = 2016


def find_oghist_file(raw_dir: str = "data/raw") -> Path:
    """
    Auto-detect the OGHIST file in data/raw/ by looking for the
    'Country Analytical History' sheet. Avoids depending on the exact filename.
    """
    raw = Path(raw_dir)
    for xlsx in raw.glob("*.xlsx"):
        try:
            sheets = pd.ExcelFile(xlsx).sheet_names
            if 'Country Analytical History' in sheets:
                return xlsx
        except Exception:
            continue
    raise FileNotFoundError(
        f"No OGHIST file found in {raw_dir}/ "
        "(looked for a workbook with a 'Country Analytical History' sheet)."
    )


def load_oghist(oghist_path: Path) -> pd.DataFrame:
    """
    Load and clean the OGHIST 'Country Analytical History' sheet.

    Returns a DataFrame: columns = ['iso3', 'country', <year int cols...>]
    """
    df = pd.read_excel(oghist_path, sheet_name='Country Analytical History', header=None)

    # Locate the year header row ("Data for calendar year :")
    year_row_idx = None
    for i in range(15):
        if 'Data for calendar year' in str(df.iloc[i, 1]):
            year_row_idx = i
            break
    if year_row_idx is None:
        raise ValueError("Could not locate 'Data for calendar year' header row in OGHIST.")

    years = df.iloc[year_row_idx, 2:].tolist()
    col_names = ['iso3', 'country'] + [
        int(y) if pd.notna(y) else f'unk{j}' for j, y in enumerate(years)
    ]

    countries = df.iloc[year_row_idx + 1:, :].copy()
    countries.columns = col_names
    countries = countries[countries['iso3'].notna()].reset_index(drop=True)
    return countries


def apply_ever_non_high(countries: pd.DataFrame,
                        year_start: int = YEAR_START,
                        year_end: int = YEAR_END) -> pd.DataFrame:
    """
    Apply the 'ever non-high-income' rule over [year_start, year_end].

    Returns the subset of economies with >=1 non-high-income classification
    in the window.
    """
    year_cols = [c for c in countries.columns
                 if isinstance(c, int) and year_start <= c <= year_end]
    if not year_cols:
        raise ValueError(f"No year columns found in range {year_start}-{year_end}.")

    def ever_non_high(row):
        vals = [row[c] for c in year_cols]
        return any(v in NON_HIGH_CODES for v in vals)

    mask = countries.apply(ever_non_high, axis=1)
    return countries[mask].reset_index(drop=True)


def build_final_sample(em_countries: pd.DataFrame,
                       lv_crises: pd.DataFrame) -> pd.DataFrame:
    """
    Intersect EM economies with LV crisis countries, using the verified crosswalk.

    Returns DataFrame: ['lv_name', 'oghist_name', 'iso3']
    Raises if any LV country resolves to a name not present in the EM list
    UNLESS it is a known advanced-economy exclusion.
    """
    em_set = set(em_countries['country'])
    iso_map = dict(zip(em_countries['country'], em_countries['iso3']))

    def resolve(lv_name: str) -> Optional[str]:
        target = LV_TO_OGHIST.get(lv_name, lv_name)
        return target if target in em_set else None

    rows, excluded = [], []
    for c in sorted(lv_crises['country'].unique()):
        resolved = resolve(c)
        if resolved is not None:
            rows.append({'lv_name': c, 'oghist_name': resolved, 'iso3': iso_map[resolved]})
        else:
            excluded.append(c)

    final_df = pd.DataFrame(rows).sort_values('lv_name').reset_index(drop=True)

    print(f"Final sample: {len(final_df)} EM crisis countries "
          f"(excluded {len(excluded)} from LV list)")
    print(f"Excluded (expected: advanced economies only): {excluded}")
    return final_df


def main(raw_dir: str = "data/raw",
         interim_dir: str = "data/interim") -> None:
    Path(interim_dir).mkdir(parents=True, exist_ok=True)

    oghist_path = find_oghist_file(raw_dir)
    print(f"Using OGHIST file: {oghist_path}")

    countries = load_oghist(oghist_path)
    print(f"Economies in OGHIST: {len(countries)}")

    em = apply_ever_non_high(countries)
    print(f"Ever non-high-income ({YEAR_START}-{YEAR_END}): {len(em)}")
    em[['iso3', 'country']].to_csv(
        Path(interim_dir) / "em_country_list_oghist.csv", index=False)

    lv_path = Path(interim_dir) / "lv_currency_crises_parsed.csv"
    lv = pd.read_csv(lv_path)

    final_df = build_final_sample(em, lv)
    final_df.to_csv(Path(interim_dir) / "final_sample_countries.csv", index=False)

    # Report crisis retention
    lv_final = lv[lv['country'].isin(final_df['lv_name'])]
    print(f"Crises retained: {len(lv_final)} of {len(lv)}")
    print(f"Saved: {interim_dir}/final_sample_countries.csv")


if __name__ == "__main__":
    main()
