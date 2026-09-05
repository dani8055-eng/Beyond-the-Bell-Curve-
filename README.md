# Beyond the Bell Curve
### A machine-learning early-warning system for currency crises

### ▶️ **[Open the live interactive dashboard →](https://dani8055-eng.github.io/Beyond-the-Bell-Curve-/outputs/dashboard.html)**
*Explore the findings, compare models, and check any country's crisis-risk history — no install, opens in your browser.*

---

Can we see a currency crisis coming six months in advance — and does information
about *extreme, rare* currency moves help more than conventional indicators?

This project builds a look-ahead-clean early-warning system across **110
emerging-market economies (1990–2016)** and tests, honestly, what historical data
can and cannot tell us about predicting rare currency crises.

---

## The question

A currency crisis is when a country's money suddenly loses a large share of its
value. They are **rare** (~3% of country-months here), **extreme**, and
**consequential** — which, following Nassim Taleb's work on fat tails, is exactly
the kind of event conventional statistical models tend to handle badly.

The central hypothesis: *information relevant to crises may be disproportionately
concentrated in the **tails** — extreme moves, volatility shifts, distributional
asymmetry — rather than in conventional averages.* This is treated as something to
**test**, not to prove; a negative result is a valid result.

## What was built

- A global panel of 110 emerging markets, crises labelled via the
  **Laeven–Valencia** definition, with a strict six-month-ahead target.
- **21 exchange-rate features** in two groups: *conventional* (depreciation,
  volatility) and *tail-aware* (skewness, kurtosis, Expected Shortfall,
  extreme-move counts), plus 11 lagged macro features.
- Rigorous **look-ahead prevention** throughout (macro data lagged 2 years to
  respect real publication timing; all features backward-looking; walk-forward
  validation).
- Three model families (Logistic Regression, Random Forest, XGBoost), compared
  fairly (including proper time-aware tuning).

## Headline findings

1. **Conventional crisis prediction is weak.** Every model sits only modestly
   above the base rate — confirming how genuinely hard rare-event prediction is.
2. **Tail features give a small but statistically reliable improvement.** Adding
   them helped in every model and in 100% of 1,000 bootstrap resamples.
3. **Cornerstone: tail-risk features carry the *majority* of the useful
   predictive signal** (~52%, vs ~28% macro and ~20% conventional, by permutation
   importance). The strongest single features are currency *skewness* and
   *extreme-move counts*.
4. **Complexity does not rescue the problem.** The most flexible model (XGBoost),
   even properly tuned, could not beat the simplest one — evidence that model
   sophistication alone does not overcome data scarcity for rare events.

An honest robustness sub-study on an 8-country subsample (using an Exchange
Market Pressure crisis definition) found the tail advantage is
*definition-sensitive* and, at that small sample size, not statistically
distinguishable from noise — reported transparently rather than overstated.

## Honest limits

- Absolute predictive skill is modest — this is a **research tool, not a
  forecast**. It estimates *relative* risk (ranking), not reliable probabilities.
- One primary crisis definition, one dataset, ends in 2016.
- The EMP sub-study is small and indicative, not conclusive.

## Repository

```
data/          raw and processed datasets
src/data/      sourcing, cleaning, sample construction, target
src/features/  feature engineering, modelling-table assembly, EMP
src/models/    experiments, tuning, robustness, importance, dashboard
outputs/       results, feature importance, dashboard.html
docs/research_decisions.md   full decision log (every methodological choice + why)
```

- **`outputs/dashboard.html`** — interactive dashboard (open in a browser):
  findings, model comparison, and a per-country risk explorer.
- **`docs/research_decisions.md`** — a numbered log of all 17 methodological
  decisions with rationale, kept for transparency and reproducibility.

Branches: `main` (the 110-country study) and `emp-robustness-subsample`
(the EMP sub-study).

## Data sources

IMF World Economic Outlook (macro); IMF International Financial Statistics via a
cited mirror (exchange rates) and FRED (reserves); Laeven–Valencia Systemic
Banking Crises Database (crisis dates); World Bank income classification (sample).

---

*Built as an independent research project. The emphasis throughout is
methodological honesty over headline performance — including reporting what the
data cannot show.*
