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

**Status of implementation:** Pending (still requires monthly FX reserves; exchange rates done). Note: EMP will be built on the CLEANED depreciation series (Decision 010), since redenomination artifacts would otherwise corrupt the EMP exchange-rate component too.

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

**Known issue for feature stage:** Several currencies redenominated (dropped zeros) during 1990–2016. Handled in Decision 010.

**Implementation:** `src/data/ingest_exchange_rates.py`
**Inputs:** `data/raw/imf_exchangerates_raw.csv`
**Output:** `data/interim/exchange_rates_monthly.csv` (iso3, date, year, month, exchange_rate, currency)

---

## Decision 009 — Primary Monthly Target (Option A)

**Status:** Approved
**Date:** September 3, 2026
**Component:** Target variable construction

**Rule:** Decision 003 applied to the monthly panel — a country's LV crisis in year Y labels all months of year Y-1 as target = 1; all else target = 0. The panel grid is defined by the monthly exchange-rate data (the country-months actually observed), so the target aligns row-for-row with the modelling data.

**Why Option A (not month-pinned):** LV provides only the crisis year, not the month. Option A invents no crisis month and therefore carries zero contestable dating assumptions. A monthly-precise EMP-based target (built from exchange rates + reserves) is planned as the robustness definition once reserves are sourced (see Decision 002).

**Result:** 33,656 country-months; 1,399 positive (4.16%); ~1 crisis-month per 24 non-crisis-months. 118 distinct crisis episodes captured in the 1990–2016 window; 81 of 110 countries have ≥1 positive label.

**Honest sample note:** Of 227 in-sample LV crises, only 118 land inside the 1990–2016 window as usable positive labels. The remainder have target years before 1990 — a direct, expected consequence of the 1990 panel start (Decision 005) — and cannot be labelled because there is no data for those months. This 118 is the honest effective crisis count for the chosen window.

**Implementation:** `src/data/build_target.py`
**Output:** `data/interim/monthly_target.csv` (iso3, date, year, month, target)

---

## Decision 010 — Exchange-Rate Structural-Break Handling (Depreciation)

**Status:** Approved
**Date:** September 3, 2026
**Component:** Feature engineering (first step)

**Problem:** Currency redenominations (dropping zeros) and dollarization (adopting the USD) create sudden STRENGTHENING of the raw exchange rate — the rate is divided by a large factor in one month. This is an accounting artifact, not a market move. A naive percent-change at that month (e.g. −99.995%) is meaningless and would poison volatility, skewness, kurtosis and tail features.

**Decision (Method 1 — neutralize artifacts, keep real moves):** At any month where the rate falls to < 20% of the previous month (ratio < 0.2, i.e. a >5× sudden strengthening), set that month's depreciation to NaN. Large WEAKENING jumps (ratio > 1) are NOT touched — those are genuine currency crises / hyperinflations (e.g. Zimbabwe 2007–2008) and are exactly the tail events this project studies.

**Why NaN, not splicing the levels:** Dollarization has no clean rescale factor (the currency ceases to exist), and a slightly-wrong splice fabricates a depreciation that never happened — worse than a missing value. The model consumes CHANGES, not continuous LEVELS, and the real moves around each break are already captured. If continuous level charts are wanted for the presentation layer, splicing may be done for VISUALIZATION ONLY, clearly labelled, and never fed into the model.

**Why this is upstream of EMP:** EMP is computed from the same exchange-rate series (plus reserves), so it would inherit the same artifact. Cleaning here protects both the primary depreciation features and the future EMP robustness measure. (EMP addresses a different problem — crises concealed via reserve losses — and is complementary to, not a substitute for, this cleaning.)

**Result:** 3 artifact months neutralized across 110 countries — Ecuador 2000-01 (dollarization), El Salvador 2001-01 (dollarization), Zimbabwe 2008-08 (redenomination). Depreciation available for 33,543 of 33,656 country-months. Zimbabwe's real hyperinflation (the +11,000%, +1.9m% weakening months) verified intact.

**Implementation:** `src/features/compute_depreciation.py`
**Output:** `data/interim/fx_depreciation_monthly.csv` (iso3, date, year, month, exchange_rate, depreciation, log_depreciation, is_structural_break)

---

## Decision 011 — Exchange-Rate Features (Conventional + Tail-Aware)

**Status:** Approved
**Date:** September 3, 2026
**Component:** Feature engineering

**Decision:** Build 21 monthly exchange-rate features on the cleaned depreciation series (Decision 010), in two explicitly-labelled groups matching the project's central experiment (conventional vs conventional+tail). Rolling windows: 3, 6, 12 months.

**Conventional (10):** `dep_1m`; rolling mean, volatility, and cumulative depreciation over 3/6/12m.

**Tail-aware (11):** rolling skewness (6/12m), excess kurtosis (6/12m), worst monthly move (6/12m), Expected Shortfall over the worst 10% (6/12m), and count of extreme months (depreciation > 10%) over 3/6/12m.

**Look-ahead safety:** Every feature is a BACKWARD-LOOKING rolling window — a feature at month t uses only depreciation up to and including t. No future data enters any feature (Rule 4). This is what makes out-of-sample evaluation credible.

**Design choices:** Distribution-shape stats (skew, kurtosis, ES) use only 6/12m windows (need enough points). Extreme threshold set at a fixed, economically-meaningful 10% monthly depreciation. Group membership is hard-coded (`CONVENTIONAL_FEATURES`, `TAIL_FEATURES`) so the conventional-vs-tail model comparison is a simple column swap.

**Validation:** Turkey around its Feb-2001 crisis — tail features confirmed dormant in the calm pre-crisis months then spiking sharply at onset (skew −0.6 → +2.4; kurtosis −0.9 → +5.8; extreme count 0 → 3), demonstrating the features capture the tail behaviour the hypothesis concerns.

**Implementation:** `src/features/build_fx_features.py`
**Output:** `data/interim/fx_features.csv` (33,656 rows; 21 features)

---

## Decision 012 — Modelling Table Assembly & Macro Lag

**Status:** Approved
**Date:** September 3, 2026
**Component:** Feature assembly / look-ahead prevention

**Decision:** Join the target, the 21 exchange-rate features, and the 11 annual macro variables into one country-month modelling table. Monthly FX features are real-time and enter with NO artificial lag. Annual WEO macro is LAGGED 2 YEARS: a year-Y macro value becomes visible to the model only from calendar year Y+2 onward.

**Why lag the macro (and why 2 years):** Annual macro is published in arrears and later revised — e.g. 2025 GDP is not released until ~mid-2026. A naive join (or a 1-year lag) would place a not-yet-published figure into early-year rows, letting the model use data that did not exist at prediction time. This is a look-ahead leak AND a data-misplacement that inflates test performance and breaks in real deployment. A 2-year lag guarantees the figure was genuinely published (and mostly revision-settled) before the model sees it, for every row, with margin. A flat 2-year lag is used as a transparent, defensible proxy for true per-country publication dates (which are not in the data). The project's star features — real-time FX tail measures — are unaffected by this lag, so the cost (slightly staler macro context) is small and honest.

**Look-ahead verification:** Confirmed on Argentina — its 2001 crisis-year GDP growth (−4.41) is not visible in the table until 2003 rows; in year Y the model sees only year-(Y−2) macro. This is the single most important leakage check in the project and it passes.

**Result:** 33,656 country-months × (21 FX + 11 macro features + target). Target positive rate preserved at 4.16%. Macro is null for 1990–1991 by design (earliest macro 1990 → usable 1992); 30,750 rows carry macro; 32,306 rows carry all FX features. Missing-feature handling (drop vs impute) deferred to the modelling stage.

**Implementation:** `src/features/build_modelling_table.py`
**Output:** `data/interim/modelling_table.csv`

---

## Decision 006 — Train/Test Split Strategy (RESOLVED)

**Status:** Approved
**Date:** September 4, 2026
**Component:** Modelling / validation

**Decision:** Walk-forward (expanding-window) validation only. For each test year, train on all years strictly before it, predict that year, then expand and step forward. Predictions are POOLED across all test years and scored once (appropriate for rare events — individual years may contain few or zero crises). First test year 2002 (earlier years reserved to give the first training window enough crises); test years 2002–2016.

**Why:** A random split would let the model train on the future to predict the past — the same look-ahead leak avoided elsewhere. Walk-forward mirrors real-time deployment (retrain as data arrives, forecast the next period) and is the standard for early-warning systems. A single time split was considered and rejected in favour of the more rigorous walk-forward.

**Primary metric:** PR-AUC (precision-recall area under curve), appropriate for a rare positive class. ROC-AUC and lift-over-base-rate reported alongside. Accuracy is NOT used.

---

## Decision 013 — Baseline Modelling Setup & First Results

**Status:** Approved
**Date:** September 4, 2026
**Component:** Modelling

**Setup:** Three model families — Logistic Regression (linear baseline), Random Forest, XGBoost — each run on three feature sets: Conventional(+macro), Conventional+Tail(+macro), Tail(+macro). Class imbalance handled via balanced class weights (logistic, RF) and scale_pos_weight (XGBoost). Missing values handled by dropping incomplete rows (baseline choice; imputation / native-missing deferred as refinements). Scaling (logistic) fit on training folds only.

**Missing-data note:** Dropping incomplete rows reduced the usable sample from 33,656 to ~20,600 country-months and the positive rate from 4.16% to ~2.6% (~600 crisis-months), because warm-up months and 1990–91 (dropped by the 2-year macro lag) carried some crisis labels. Honest, documented cost of the drop-rows choice.

**First results (walk-forward, pooled OOS PR-AUC; base rate ~2.56%):**
- Logistic: Conventional 0.030 → Conv+Tail 0.040 (tail effect +33%)
- Random Forest: Conventional 0.033 → Conv+Tail 0.036 (tail effect +7%)
- XGBoost: Conventional 0.025 → Conv+Tail 0.027 (tail effect +7%)

**Preliminary findings (tentative, not yet robustness-tested):**
1. All models sit only modestly above the base rate — conventional crisis prediction is genuinely weak (supports H1).
2. Tail features improved PR-AUC in every model — consistent directional support for H2, though modest.
3. The tail benefit is large for the linear model (+33%) and small for the nonlinear models (+7%), suggesting flexible models can partly reconstruct tail behaviour from conventional features themselves.
4. XGBoost, the most complex model, performed WORST (near base rate) — almost certainly under-tuned on this rare, small-positive dataset. To be re-tested after tuning before any conclusion. This echoes H3 (complexity does not automatically help for rare extreme events).

**Caveat:** These are first-pass results with near-default hyperparameters and no uncertainty quantification. Next steps: tune XGBoost for a fair comparison, then bootstrap/per-fold spread to test whether the tail effect is signal or noise, then feature importance.

**Implementation:** `src/models/run_core_experiment.py`
**Output:** `outputs/model_results.csv`

---

## Decision 014 — XGBoost Tuning (Fair-Comparison Result)

**Status:** Approved
**Date:** September 4, 2026
**Component:** Modelling

**Why:** In Decision 013, default XGBoost performed worst of the three models. Before concluding "complexity doesn't help," XGBoost had to be given a fair, tuned shot.

**Method (leak-free, time-aware tuning):** A 48-combination grid over max_depth, learning_rate, n_estimators, min_child_weight, reg_lambda (subsample/colsample fixed) was searched using a TIME-BASED split — train on years ≤ 2009, select on validation years 2010–2013 — so the 2014–2016 test years never influenced tuning. Random/shuffled CV was deliberately NOT used, as it would reintroduce look-ahead leakage. The tuned model was then re-run through the full walk-forward (2002–2016).

**Result:** Tuning did NOT rescue XGBoost. Best validation PR-AUC across the whole grid was 0.0186 — below the base rate. Tuned walk-forward: Conventional 0.023 → Conv+Tail 0.026 (tail effect +12%), essentially unchanged from default XGBoost (0.025 → 0.027) and still near the base rate (ROC-AUC ~0.48–0.52). Best tuned params: max_depth 4, learning_rate 0.1, n_estimators 400, min_child_weight 5, reg_lambda 5.0.

**Interpretation:** Given a fair, properly-tuned shot, the most flexible model still cannot beat the simple linear one. The likely cause is data scarcity in the positive class (~600 crisis-months, noisy and heterogeneous) — flexible models overfit patterns that do not generalise out-of-sample. This strengthens H3: model complexity does not automatically help for rare extreme events. Across ALL five model runs, the best performer remains Logistic Regression + tail features (PR-AUC 0.040) — the simplest model with the tail twist.

**Standing finding:** "Simple model + tail-risk features beats brute-force complexity for predicting rare currency crises." Tail effect is positive in every model (Logistic +33%, RF +7%, default XGB +7%, tuned XGB +12%) — consistent direction, modest size. Absolute PR-AUC remains low throughout; framing is "tail features modestly and consistently help a fundamentally hard problem," not "crisis prediction solved."

**Next:** statistical robustness (bootstrap / per-fold spread) to test whether the consistent tail effect is signal vs noise — needs no new data. Then EMP-based definitional robustness — blocked on sourcing FX reserves.

**Implementation:** `src/models/tune_xgboost.py`
**Output:** `outputs/xgb_tuning_results.csv`

---

## Decision 015 — Robustness of the Tail-Feature Finding

**Status:** Approved
**Date:** September 4, 2026
**Component:** Modelling / robustness

**Question:** Is "tail features improve crisis prediction" a real, reliable effect or an artifact of sampling luck? Tested on the best model (Logistic Regression, largest tail effect), both feature sets evaluated on the SAME test rows.

**Check 1 — per-year breakdown:** Tail features helped in 9 of 12 test years that contained crises; the other 3 years showed differences of essentially zero (largest −0.003), and no year showed a clear decline. The largest gains fell in the most crisis-heavy years (2007 +0.047, 2014 +0.035), which is economically sensible.

**Check 2 — bootstrap (1000 resamples of pooled OOS predictions):** Observed pooled benefit +0.0099 (conv 0.030 → conv+tail 0.040). Bootstrap mean +0.0103, 95% range +0.0060 to +0.0154, and tail features helped in 100% of 1000 resamples — the benefit never turned negative.

**Conclusion:** The tail-feature improvement is SMALL but RELIABLE. It is consistent across time (most years) and across resampling (every resample), so it is signal, not noise — while remaining modest in absolute size. Honest framing: "tail-risk features give a small but statistically reliable improvement to a fundamentally hard prediction problem."

**Implementation:** `src/models/run_robustness.py`
**Outputs:** `outputs/robustness_per_year.csv`, `outputs/robustness_bootstrap.csv`

---

## Decision 016 — Feature Importance (Cornerstone Finding)

**Status:** Approved
**Date:** September 4, 2026
**Component:** Modelling / interpretation

**Question:** Which features actually drive the model's crisis predictions — does it rely on the tail-risk features, or ignore them?

**Method:** Two views on the best model (Logistic Regression), trained on years ≤ 2009, evaluated on 2010–2016 (time-based, no look-ahead).
1. Standardized coefficients — reported as CONTEXT only. Misleading here because the depreciation features are highly correlated (near-duplicates), so the crude coefficient view over-credits conventional features.
2. Permutation importance (HEADLINE) — shuffle each feature to noise and measure the drop in PR-AUC. Robust to correlation because it measures each feature's UNIQUE contribution (duplicated features score low, as their copies cover for them). 20 shuffles averaged.

**Result (permutation, base PR-AUC 0.038):** Of all positive predictive importance, TAIL features carry ~52%, macro ~28%, conventional ~20%. The strongest individual exchange-rate features are skewness (dep_skew_6m/12m) and extreme-move counts (extreme_count_*) — i.e. the tail-risk measures. Conventional features individually score low precisely because they are redundant with one another, not because currency dynamics are irrelevant.

**Interpretation (cornerstone):** When importance is measured by actual predictive contribution, tail-risk features carry the MAJORITY of the useful crisis signal — more than conventional or macro indicators. This is direct support for the project's central, Taleb-inspired hypothesis: information about impending crises is disproportionately concentrated in the tails (extreme, rare moves), not in conventional averages.

**Caveats to keep attached (non-negotiable for honest reporting):**
1. Absolute predictive skill remains modest (PR-AUC ~0.04). The claim is "tail features are the most useful ingredients in a hard problem," NOT "tail features predict crises well."
2. One dataset, one crisis definition (Laeven–Valencia), one model family. Generalisation is open — motivating the planned EMP definitional-robustness test.

**Implementation:** `src/models/feature_importance.py`
**Outputs:** `outputs/feature_importance_coef.csv`, `outputs/feature_importance_permutation.csv`

---

## Decision 017 — EMP Robustness Sub-Study (reserves-available subsample)

**Status:** Approved
**Date:** September 4, 2026
**Component:** Robustness (branched sub-study; git branch `emp-robustness-subsample`)
**Scope:** 8 countries with monthly FRED reserves — ARG, BRA, IDN, KOR, MEX, RUS, TUR, ZAF. Self-contained and separate from the main 110-country Laeven–Valencia analysis. Small sample → results are INDICATIVE, not conclusive.

**Why this sub-study exists (data-availability scoping):** The EMP crisis definition needs monthly FX reserves. Reserves were sourced from FRED (republishing IMF IFS), which only carries usable monthly series for these 8 (mostly large, crisis-prone) economies — not the full sample. Rather than abandon the test or compare mismatched samples, the whole analysis (H1, H2, and the EMP check) is run WITHIN this fixed subsample, so no cross-sample confound arises. The scope limitation is stated plainly rather than hidden.

**EMP construction:** Two-component EMP (depreciation + reserve losses; interest rates unavailable), each component scaled by its country standard deviation. EMP crisis = EMP above the country mean by k standard deviations, for k=1.5 and k=2.0. Validation: the EMP crisis dates independently reproduce known history — Korea 1997, Russia 1998, Argentina 2001–02, Turkey 2001, Brazil 1999 — confirming the measure is capturing real crises. Target: a TRUE six-month-ahead monthly window (cleaner than the annual LV approximation, since EMP is monthly).

**Result (walk-forward, pooled OOS PR-AUC):**
- k=2.0 (strict): Conventional 0.109 → Conv+Tail 0.115 (tail effect +5.4%) — consistent with the main finding.
- k=1.5 (loose): Conventional 0.190 → Conv+Tail 0.176 (tail effect −7.5%) — tail features did not help.
EMP crises are far more common here (10–17%) than LV crises (3%), so absolute PR-AUC is higher; EMP is a different, easier prediction problem.

**Interpretation (consistent with, not proven):** The tail-feature benefit is sensitive to how the crisis is defined. It appears for STRICT, clear-cut crises (k=2.0) but not for the LOOSER definition (k=1.5). A plausible economic reading, consistent with first-generation currency-crisis theory (Krugman 1979): the looser threshold captures episodes where a government suppresses depreciation by burning FX reserves, keeping the exchange rate artificially stable. Exchange-rate-based tail features are, by construction, blind to that reserve-defended phase — they only activate once reserves are exhausted and the currency is allowed to move (the collapse). Thus the tail advantage concentrates in visibly exchange-rate-driven crises. This is an interpretation the data is consistent with, not a demonstrated causal claim (8 countries; possible noise / EMP-weighting effects).

**Bootstrap test of the threshold difference (1000 resamples each):** Run to check whether the k=1.5-vs-k=2.0 difference is a real pattern in the subsample or sampling noise. Result: NEITHER effect is statistically distinguishable from noise at this sample size. k=1.5: mean −0.014, 95% range −0.032 to +0.001 (4% of resamples positive) — leans negative but the range crosses zero. k=2.0: mean +0.006, 95% range −0.006 to +0.018 (87% positive) — leans positive but the range crosses zero. So the directions match the interpretation above (negative at loose, positive at strict), but with only 8 countries the effects are UNDERPOWERED and cannot be claimed as established. Honest status: SUGGESTIVE BUT NOT CONCLUSIVE.

**Honest headline for the sub-study:** On the 8-country subsample, the tail-feature effect leaned negative under the loose EMP definition and positive under the strict one — directionally consistent with the reserve-defence interpretation — but bootstrap testing shows neither is distinguishable from noise at this sample size. A larger monthly-reserves dataset would be needed to test the pattern properly. (This is a deliberate integrity check: an appealing interpretation was tested against the data's actual power and reported as inconclusive rather than overclaimed.)

**Value / future work:** This points directly to reserve-based tail features (extreme reserve drawdowns, reserve volatility) as the natural extension — to catch the hidden reserve-defence phase that exchange-rate tails miss.

**Implementation:** `src/data/pull_reserves.py`, `src/features/build_emp.py`, `src/models/emp_substudy.py`, `src/models/emp_bootstrap.py`
**Outputs:** `data/interim/emp_measure.csv`, `outputs/emp_substudy_results.csv`, `outputs/emp_bootstrap.csv`

---

# Amendment Log

*If a decision is revised, amendments are logged below with the original decision retained above.*

**Amendment A (2026-09-03):** Panel period revised from 1970 to 1990 (see Decision 005) — pre-1990 data coverage for emerging markets, especially post-Soviet/Yugoslav states, was too sparse to support a 1970 start.

**Amendment B (2026-09-03):** Monthly exchange rates obtained via a documented IMF-IFS mirror rather than the IMF API or manual portal download, after the API was found non-functional (see Decisions 004 and 008).

**Amendment C (2026-09-03):** Primary monthly target built via Option A (Decision 009). A monthly EMP-based target was chosen (over a month-pinning heuristic) as the robustness definition, to be built once FX reserves are sourced — giving monthly timing precision via a principled economic index rather than an arbitrary dating rule.

---

# How to use this log

1. **Before making a methodological change**, check this log.
2. **After making a decision**, add a new entry with date, rationale, and expected effects.
3. **If revising a past decision**, add an amendment section rather than deleting the original.
4. **When writing the final report**, reference decisions by number (e.g., "per Decision 003").

---

**Last updated:** September 4, 2026
**Version:** 2.2
**Maintained by:** Danial
