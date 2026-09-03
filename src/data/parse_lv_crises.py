"""
Parse Laeven-Valencia currency crisis database (Crisis Years sheet).

This module reads the raw LV Excel file and extracts currency crisis years
into a structured format for target construction.

Note: This module is READ-ONLY for the LV source file. We never modify the
original Excel file.

See: docs/research_decisions.md (Decision 003) for the target construction rule.
"""

import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Tuple


def read_lv_raw(lv_file_path: str) -> pd.DataFrame:
    """
    Read the LV Crisis Years sheet and return as DataFrame.
    
    Args:
        lv_file_path: Path to the LV Excel workbook
        
    Returns:
        DataFrame with columns: Country, Currency Crisis (and others)
    """
    df = pd.read_excel(lv_file_path, sheet_name='Crisis Years')
    return df


def parse_currency_crisis_years(crisis_string) -> List[int]:
    """
    Parse a Currency Crisis cell into a list of crisis years.
    
    The LV database stores crisis years as:
    - Single year: 1997
    - Multiple years (comma-separated): 1975, 1981, 1987, 2002, 2013
    - Header or NaN: (year), NaN, etc.
    
    Args:
        crisis_string: Value from the Currency Crisis column
        
    Returns:
        List of integer years, or empty list if no crises
        
    Examples:
        >>> parse_currency_crisis_years(1997)
        [1997]
        
        >>> parse_currency_crisis_years("1975, 1981, 1987")
        [1975, 1981, 1987]
        
        >>> parse_currency_crisis_years("(year)")
        []
        
        >>> parse_currency_crisis_years(None)
        []
    """
    # Handle NaN, None, empty
    if pd.isna(crisis_string):
        return []
    
    # Convert to string
    crisis_str = str(crisis_string).strip()
    
    # Skip header row
    if crisis_str == "(year)" or crisis_str == "":
        return []
    
    # Try to parse as integer
    try:
        year = int(crisis_str)
        return [year]
    except ValueError:
        pass
    
    # Parse comma-separated values
    # Handle spaces around commas and within years
    years = []
    parts = crisis_str.split(',')
    
    for part in parts:
        part = part.strip()
        # Handle ranges like "2001,2007" (no space after comma)
        try:
            year = int(part)
            years.append(year)
        except ValueError:
            # Skip non-numeric parts
            pass
    
    return years


def construct_crisis_chronology(df: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Extract currency crisis years for each country.
    
    Args:
        df: DataFrame from read_lv_raw()
        
    Returns:
        Dictionary mapping country name to list of crisis years
        {
            "Argentina": [1975, 1981, 1987, 2002, 2013],
            "Brazil": [1976, 1982, 1987, 1992, 1999, 2015],
            ...
        }
    """
    chronology = {}
    
    for idx, row in df.iterrows():
        country = row['Country']
        crisis_value = row['Currency Crisis']
        
        # Skip NaN countries
        if pd.isna(country):
            continue
        
        country = str(country).strip()
        
        # Parse the crisis years
        crisis_years = parse_currency_crisis_years(crisis_value)
        
        if crisis_years:
            chronology[country] = sorted(crisis_years)
    
    return chronology


def create_crisis_year_dataframe(chronology: Dict[str, List[int]]) -> pd.DataFrame:
    """
    Convert the chronology dict into a long-format DataFrame for inspection.
    
    This creates one row per (country, crisis_year) pair, useful for validation
    and as an intermediate output.
    
    Args:
        chronology: Dictionary from construct_crisis_chronology()
        
    Returns:
        DataFrame with columns: country, crisis_year
        Sorted by country, then crisis_year
        
    Example:
        country         crisis_year
        Albania         1997
        Algeria         1988
        Algeria         1994
        ...
    """
    rows = []
    for country, years in chronology.items():
        for year in years:
            rows.append({'country': country, 'crisis_year': year})
    
    df_crises = pd.DataFrame(rows)
    df_crises = df_crises.sort_values(['country', 'crisis_year']).reset_index(drop=True)
    
    return df_crises


def generate_target_year_mapping(chronology: Dict[str, List[int]]) -> Dict[Tuple[str, int], bool]:
    """
    Generate the target year mapping (implements Decision 003).
    
    Rule: If a country experiences a crisis in year Y, then year (Y-1) is marked
    as having a positive target (six-month-ahead prediction window reaches into Y).
    
    Args:
        chronology: Dictionary from construct_crisis_chronology()
        
    Returns:
        Dictionary mapping (country, year) -> bool, where True means this year
        should have target = 1 for six-month-ahead predictions
        
    Example:
        If Argentina had crises in 1975, 1981, 1987, 2002, 2013:
        ("Argentina", 1974) -> True
        ("Argentina", 1975) -> False (crisis year itself, not target year)
        ("Argentina", 1980) -> True
        ("Argentina", 1981) -> False
        ...
    """
    target_years = {}
    
    for country, crisis_years in chronology.items():
        for crisis_year in crisis_years:
            # The preceding year gets the positive target
            target_year = crisis_year - 1
            target_years[(country, target_year)] = True
    
    return target_years


def main(lv_file_path: str, output_dir: str = "data/interim") -> None:
    """
    Main workflow: read LV, parse, and write intermediate outputs.
    
    Args:
        lv_file_path: Path to the LV Excel workbook
        output_dir: Directory to write intermediate CSV files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read and parse
    print(f"Reading LV database from: {lv_file_path}")
    df_lv = read_lv_raw(lv_file_path)
    print(f"  Shape: {df_lv.shape}")
    
    chronology = construct_crisis_chronology(df_lv)
    print(f"\nExtracted {len(chronology)} countries with currency crises")
    
    # Create crisis year dataframe for inspection
    df_crises = create_crisis_year_dataframe(chronology)
    print(f"Total crisis observations: {len(df_crises)}")
    print(f"Date range: {df_crises['crisis_year'].min()}–{df_crises['crisis_year'].max()}")
    
    # Generate target years (implements Decision 003)
    target_years = generate_target_year_mapping(chronology)
    print(f"\nTarget years (Y-1 for each crisis year Y): {len(target_years)}")
    
    # Save to CSV for inspection
    crisis_csv_path = Path(output_dir) / "lv_currency_crises_parsed.csv"
    df_crises.to_csv(crisis_csv_path, index=False)
    print(f"\nSaved parsed crises to: {crisis_csv_path}")
    
    # Save target years mapping as CSV
    target_df = pd.DataFrame([
        {'country': k[0], 'target_year': k[1], 'has_positive_target': v}
        for k, v in target_years.items()
    ]).sort_values(['country', 'target_year']).reset_index(drop=True)
    
    target_csv_path = Path(output_dir) / "lv_target_years_mapping.csv"
    target_df.to_csv(target_csv_path, index=False)
    print(f"Saved target year mapping to: {target_csv_path}")
    
    # Print sample
    print("\n" + "="*80)
    print("SAMPLE: Currency crises for Argentina")
    print("="*80)
    sample_country = "Argentina"
    if sample_country in chronology:
        crisis_years = chronology[sample_country]
        print(f"\nCrisis years: {crisis_years}")
        print(f"\nTarget years (Y-1 for each crisis year Y):")
        for cy in crisis_years:
            print(f"  {cy} → target year {cy-1}")
    
    print("\n" + "="*80)
    print("Parsing complete. Intermediate files written to data/interim/")
    print("="*80)


if __name__ == "__main__":
    # Example usage
    lv_path = "data/raw/1788427154723_SYSTEMIC_BANKING_CRISES_DATABASE_2018.xlsx"
    main(lv_path)
