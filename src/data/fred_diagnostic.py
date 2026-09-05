"""
DIAGNOSTIC: show what FRED actually returns for reserves searches.
Not a data pull -- just prints raw search results so we can see the real
series-ID and title formats, then fix the puller correctly.
"""
from pathlib import Path
from fredapi import Fred

fred = Fred(api_key=Path(".fred_api_key").read_text().strip())

for query in [
    "Total Reserves excluding Gold Argentina",
    "reserves Argentina",
    "Total Reserves excluding Gold Angola",
    "reserves Angola",
    "Total Reserves excluding Gold Chile",
]:
    print("=" * 70)
    print("QUERY:", query)
    print("=" * 70)
    try:
        res = fred.search(query)
        if res is None or len(res) == 0:
            print("  (no results)")
            continue
        # show top 5 results: id, title, frequency, units
        cols = [c for c in ['id', 'title', 'frequency', 'units'] if c in res.columns]
        print(res[cols].head(5).to_string())
    except Exception as e:
        print("  ERROR:", e)
    print()
