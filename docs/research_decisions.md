# Beyond the Bell Curve — Research Decisions Log

This document records all major methodological decisions made during the project. Every decision includes:
- **Date decided**
- **What changed** (if it's a revision)
- **Rationale** (why this choice)
- **Expected/actual effects** (what this means for the model)

The objective is to ensure reproducibility and transparency. Future researchers should understand not just *what* we did, but *why*.

---

## Decision 001 — Crisis Definition (Primary)

**Status:** Approved
**Date:** [Project initiation]
**Component:** Dependent variable definition

**Decision:** Use the Laeven–Valencia (LV) currency-crisis definition as the primary framework.

**Operationalization:**
- A currency crisis occurs when:
  - Nominal depreciation against the US dollar ≥ 30%, **AND**
  - Depreciation exceeds prior year by ≥ 10 percentage points

**Source:** Laeven & Valencia (2018) Systemic Banking Crises Database

**Rationale:**
- Well-established in empirical literature
- Transparent and reproducible
- Clear binary classification
- Provides strong benchmark for model evaluation

**Data coverage:** LV records currency crises 1970–2017.

---

## Decision 002 — Crisis Definition (Robustness)

**Status:** Approved
**Date:** [Project initiation]
**Component:** Robustness / alternative dependent variable

**Decision:** Implement an Exchange Market Pressure (EMP) index as an alternative crisis measure.

**Operationalization:** To be formalized after monthly exchange-rate and reserves data are acquired. (Exchange rates now acquired — Decision 008. Still requires monthly reserves.)

**Rationale:**
- Captures currency pressure that may not result in ≥30% depreciation
- Central banks may prevent exchange-rate collapse through reserve losses
- Allows testing whether conclusions are robust to crisis definition

**Use:** Robustness tests only (not primary results).

**Status of implementation:** Pending (still requires monthly FX reserves; exchange rates done).

---

## Decision 003 — Monthly Six-Month-Ahead Target Construction

**Status:** Approved
**Date:** September 3, 2026
**Component:** Target variable construction / look-ahead prevention

**Problem:**
LV records crises at annual frequency. Our model requires monthly observations with six-month-ahead binary targets. LV does not specify the exact month within a crisis year.

**Decision:**

If a country experiences a currency crisis in year Y (per LV), then year (Y-1) is marked with a positive target for six-month-ahead prediction:

```
For each month t in year (Y-1):
    Target(t, t+1:t+6) = 1    [crisis within next 6 months]
For all other months:
    Target(t, t+1:t+6) = 0    [no crisis]
```

**Rationale:**
1. Conservative timing: does not assume the exact crisis month. Any month in year Y-1 has a 6-month window extending into year Y.
2. Reproducible: mechanical rule, no arbitrary date-picking.
3. Look-ahead clean: a prediction at month t cannot use information from t+1..t+6.
4. Matches the six-month prediction horizon.

**Example (Argentina, LV crises 1975, 1981, 1987, 2002, 2013):**
Target years (Y-1): 1974, 1980, 1986, 2001, 2012 → all months in those years get target = 1.

**Data implications:**
- The parsed LV crisis chronology contains 239 crises across 119 countries (1971–2016), before any sample restriction.
- After the sample restriction (Decision 005) the modelling sample is 112 countries / 227 crises.
- Crises are rare — a strongly imbalanced positive class — which is why evaluation focuses on rare-event metrics rather than accuracy.

**Implementation:** `src/data/parse_lv_crises.py`

**Intermediate outputs:**
- `data/interim/lv_currency_crises_parsed.csv` — one row per (country, crisis_year)
- `data/interim/lv_target_years_mapping.csv` — mapping of (country, target_year) → positive_target

---

## Decision 004 — Data Sourcing Strategy

**Status:** Approved
**Date:** September 3, 2026
**Component:** Data acquisition

**Decision:** Two-source strategy split by frequency.
- **Annual macro variables:** IMF World Economic Outlook (WEO) bulk file. See Decision 007.
- **Monthly variables (exchange rates, reserves, interest rates, money):** IMF International Financial Statistics (IFS). The IMF SDMX API was tested on 2026-09-03 and did not return data (new host `api.imf.org` returned HTTP 403; legacy host `dataservices.imf.org` no longer resolves). Monthly exchange rates were therefore obtained via a documented IMF-IFS mirror (Decision 008). Remaining monthly variables (reserves, interest rates, money) still to be sourced.

**Rationale:**
- WEO is a single, well-documented bulk file covering all needed annual variables for our country set.
- IFS is the standard source for monthly external-sector series and is required for both features and the EMP robustness measure.

**Status:** Annual data DONE (Decision 007). Monthly exchange rates DONE (Decision 008). Reserves / interest rates / money PENDING.

---

## Decision 005 — Sample: Country Inclusion & Panel Period

**Status:** Approved
**Date:** September 3, 2026
**Component:** Sample construction

**Panel period:** 1990–2016 (revised down from an aspirational 1970 to avoid severe pre-1990 data gaps, especially for post-Soviet and post-Yugoslav states). The panel is UNBALANCED; each country's usable window is set by data availability.

**Country inclusion:** Emerging markets = any economy classified by the World Bank (OGHIST) as NON-high-income at any point in 1990–2016 ("ever non-high-income"), intersected with LV currency-crisis countries.

**Result:** 112 EM crisis countries; 227 of 239 LV crises retained. 7 advanced economies excluded (Finland, Iceland, Israel, Italy, New Zealand, Spain, Sweden).

**Name crosswalk:** 96 exact matches; 16 verified manual mappings (see `LV_TO_OGHIST` in `src/data/build_sample.py`); 7 excluded as advanced. Zero accidental drops (verified: all exclusions are advanced economies).

**Look-ahead note:** Using full-sample income status to define the country universe is standard and is DISTINCT from the predictor look-ahead rule (Rule 4), which governs which features enter a prediction at time t.

**Source:** World Bank OGHIST historical income classification (1987–2025), sheet "Country Analytical History".

**Implementation:** `src/data/build_sample.py`
**Output:** `data/interim/final_sample_countries.csv`

---

## Decision 006 — Train/Test Split Strategy

**Status:** Pending
**Component:** Modelling

To be completed during the modelling phase. Must respect the look-ahead rule (Rule 4) — splits will be time-based (out-of-sample = later period), not random, to avoid leakage across the panel.

---

## Decision 007 — Annual Macro Data Source (WEO)

**Status:** Approved
**Date:** September 3, 2026
**Component:** Annual feature data

**Source:** IMF World Economic Outlook, October 2024 vintage (`WEOOct2024all.xls`), a tab-separated / UTF-16-LE bulk file.

**Variables ingested (11):** GDP growth, inflation (avg & eop), current account (%GDP), fiscal balance (%GDP), govt gross debt (%GDP), investment (%GDP), gross savings (%GDP), export & import volume (%chg), GDP (current US$).

**Coverage:** 111 of 112 sample countries. New Caledonia (NCL) absent from WEO (non-IMF-member territory) — dropped, no macro data available.

**Actuals vs forecasts:** WEO mixes historical actuals with IMF projections. Values in years after each series' "Estimates Start After" are flagged `is_forecast=True`. Only 69 such rows fall in 1990–2016; they are marked, not used as observations.

**Non-null actual coverage:** 75%–97% by variable (govt debt weakest at 75%; core macro series 91–97%).

**Implementation:** `src/data/ingest_weo.py`
**Output:** `data/interim/weo_annual_long.csv` (tidy long format: iso3, year, variable, weo_code, value, is_forecast)

---

## Decision 008 — Monthly Exchange-Rate Data Source

**Status:** Approved
**Date:** September 3, 2026
**Component:** Monthly feature data / target input

**Source:** codeforIATI/imf-exchangerates — nightly scrape of IMF IFS exchange rates, consolidated CSV. Underlying source: IMF International Financial Statistics (`https://data.imf.org/en/datasets/IMF.STA:ER`). Attribution: "Source: International Monetary Fund."

**Why a mirror:** The IMF SDMX API was non-functional when tested 2026-09-03 (new host returned HTTP 403; legacy host no longer resolves). The mirror is IMF data, properly attributed, and documented here for reproducibility. Preferred over manual portal download, which would be error-prone across 110+ countries. Using a cited mirror of a primary source is standard research practice.

**Variable:** Monthly nominal exchange rate, domestic currency per US dollar.

**Result:** 110 of 112 sample countries covered, 1990–2016, ~33,656 monthly observations.

**Known gaps:** Iran (IRN) and Mauritania (MRT) — no usable historical exchange-rate data in this source for 1990–2016; they will lack an exchange-rate-based target.

**Namibia note:** Namibia is reported under the South African rand (ZAR, blank country code) because the Namibian dollar is pegged 1:1 to the rand; patched to iso3 = NAM in ingestion. Its rate therefore moves identically to South Africa's.

**Known issue for feature stage:** Several currencies redenominated (dropped zeros) during 1990–2016 — Angola, Brazil, Argentina, Turkey, DRC and others. This creates large artificial level breaks in the raw series; a "depreciation" across a redenomination is not a real market move and MUST be handled during feature engineering (not corrected in ingestion).

**Implementation:** `src/data/ingest_exchange_rates.py`
**Inputs:** `data/raw/imf_exchangerates_raw.csv`
**Output:** `data/interim/exchange_rates_monthly.csv` (iso3, date, year, month, exchange_rate, currency)

---

# Amendment Log

*If a decision is revised, amendments are logged below with the original decision retained above.*

**Amendment A (2026-09-03):** Panel period revised from 1970 to 1990 (see Decision 005) — pre-1990 data coverage for emerging markets, especially post-Soviet/Yugoslav states, was too sparse to support a 1970 start.

**Amendment B (2026-09-03):** Monthly exchange rates obtained via a documented IMF-IFS mirror rather than the IMF API or manual portal download, after the API was found non-functional (see Decisions 004 and 008).

---

# How to use this log

1. **Before making a methodological change**, check this log.
2. **After making a decision**, add a new entry with date, rationale, and expected effects.
3. **If revising a past decision**, add an amendment section rather than deleting the original.
4. **When writing the final report**, reference decisions by number (e.g., "per Decision 003").

---

**Last updated:** September 3, 2026
**Version:** 1.2
**Maintained by:** Danial
