"""
IMF API connectivity test — ONE series, Argentina monthly exchange rate.

Purpose: find out which (if any) IMF API endpoint works from your machine
RIGHT NOW, before we build anything bigger. This does not save data. It just
tests and reports.

It tries, in order:
  1. The newer IMF API host (api.imf.org, SDMX 3.0)
  2. The legacy host (dataservices.imf.org, SDMX 2.1) — being retired, may fail

Just run:  python src/data/test_imf_api.py
Then paste the ENTIRE output back.
"""

import sys

try:
    import requests
except ImportError:
    print("The 'requests' library isn't installed. Install it by running:")
    print("    pip install requests")
    sys.exit(1)


def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def try_endpoint(name, url, headers=None, params=None):
    """Attempt one GET request and report clearly what happened."""
    print(f"\n[{name}]")
    print(f"URL: {url}")
    if params:
        print(f"Params: {params}")
    try:
        r = requests.get(url, headers=headers or {}, params=params or {}, timeout=30)
        print(f"HTTP status: {r.status_code}")
        body = r.text
        print(f"Response length: {len(body)} characters")
        preview = body[:600].replace("\n", " ")
        print(f"Preview (first 600 chars):\n{preview}")
        if r.status_code == 200 and len(body) > 200:
            print(f"\n>>> [{name}] LOOKS LIKE IT WORKED (status 200, got content).")
            return True
        else:
            print(f"\n>>> [{name}] did NOT return usable data.")
            return False
    except requests.exceptions.Timeout:
        print(f">>> [{name}] TIMED OUT (no response in 30s).")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f">>> [{name}] CONNECTION ERROR: {e}")
        return False
    except Exception as e:
        print(f">>> [{name}] UNEXPECTED ERROR: {type(e).__name__}: {e}")
        return False


def main():
    hr("IMF API TEST — Argentina monthly exchange rate (period average, vs USD)")
    print("Series wanted: monthly (M), Argentina (AR / ARG),")
    print("exchange rate indicator ENDA_XDC_USD_RATE (domestic currency per USD).")

    results = {}

    url_new = "https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/IFS/+/M.ARG.ENDA_XDC_USD_RATE"
    results["NEW api.imf.org"] = try_endpoint(
        "NEW api.imf.org",
        url_new,
        headers={"Accept": "text/csv"},
        params={"c[TIME_PERIOD]": "ge:2000-01"},
    )

    url_legacy = ("https://dataservices.imf.org/REST/SDMX_JSON.svc/"
                  "CompactData/IFS/M.AR.ENDA_XDC_USD_RATE")
    results["LEGACY dataservices.imf.org"] = try_endpoint(
        "LEGACY dataservices.imf.org",
        url_legacy,
        headers={"Accept": "application/json"},
        params={"startPeriod": "2000", "endPeriod": "2016"},
    )

    hr("SUMMARY — what worked")
    any_worked = False
    for name, ok in results.items():
        status = "WORKED" if ok else "failed"
        print(f"  {name:32} : {status}")
        any_worked = any_worked or ok

    hr("WHAT TO DO NEXT")
    if any_worked:
        print("At least one endpoint returned data. Good — paste this whole")
        print("output back and we'll build the real puller around the one that worked.")
    else:
        print("Neither endpoint worked from your machine right now.")
        print("That's OK and not your fault — the IMF API is mid-migration.")
        print("Paste this whole output back and we'll switch to the MANUAL")
        print("download route (I'll walk you through the IMF website click by click).")


if __name__ == "__main__":
    main()
