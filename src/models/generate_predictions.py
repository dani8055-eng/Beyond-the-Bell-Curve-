"""
Generate per-country, per-month crisis-risk predictions from the best model.

The modelling scripts computed pooled metrics but did not save individual
predictions. The interactive dashboard needs a risk estimate for every
country-month, so this script runs the best model (Logistic Regression on
conventional + tail features) in walk-forward fashion and saves the predicted
crisis probability for each out-of-sample country-month.

Look-ahead safe: each year's predictions come from a model trained only on
earlier years (same walk-forward as the evaluation).

Honest labelling note for the dashboard: these are ESTIMATED RISK values from a
research model with modest absolute skill -- not a forecast or guarantee. The
most recent value (Dec 2016, the data's end) is the "headline" risk per country.

INPUT:  data/interim/modelling_table.csv
OUTPUT: outputs/country_risk_predictions.csv
        (iso3, date, year, risk, actual_target)
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

INTERIM = Path("data/interim")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

CONVENTIONAL = ['dep_1m', 'dep_mean_3m', 'dep_vol_3m', 'dep_cum_3m', 'dep_mean_6m',
                'dep_vol_6m', 'dep_cum_6m', 'dep_mean_12m', 'dep_vol_12m', 'dep_cum_12m']
TAIL = ['dep_skew_6m', 'dep_kurt_6m', 'dep_max_6m', 'dep_es90_6m', 'dep_skew_12m',
        'dep_kurt_12m', 'dep_max_12m', 'dep_es90_12m', 'extreme_count_3m',
        'extreme_count_6m', 'extreme_count_12m']

FIRST_TEST_YEAR = 2002
LAST_YEAR = 2016


def main():
    df = pd.read_csv(INTERIM / "modelling_table.csv")
    df['date'] = pd.to_datetime(df['date'])
    macro = [c for c in df.columns if c.startswith('macro_')]
    feat = CONVENTIONAL + TAIL + macro

    d = df.dropna(subset=feat + ['target']).reset_index(drop=True)

    preds = []
    for ty in range(FIRST_TEST_YEAR, LAST_YEAR + 1):
        tr, te = d[d['year'] < ty], d[d['year'] == ty]
        if len(te) == 0 or tr['target'].sum() == 0:
            continue
        sc = StandardScaler().fit(tr[feat].values)
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(sc.transform(tr[feat].values), tr['target'].values)
        p = clf.predict_proba(sc.transform(te[feat].values))[:, 1]
        out = te[['iso3', 'date', 'year', 'target']].copy()
        out['risk'] = p
        preds.append(out)

    allpreds = pd.concat(preds, ignore_index=True)
    allpreds = allpreds.rename(columns={'target': 'actual_target'})
    allpreds = allpreds.sort_values(['iso3', 'date']).reset_index(drop=True)
    allpreds.to_csv(OUT / "country_risk_predictions.csv", index=False)

    print(f"Saved: {OUT / 'country_risk_predictions.csv'}  ({len(allpreds):,} rows)")
    print(f"Countries with predictions: {allpreds['iso3'].nunique()}")
    print(f"Date range: {allpreds['date'].min().date()} to {allpreds['date'].max().date()}")
    print(f"\nHighest current (Dec 2016) risk countries:")
    latest = allpreds.sort_values('date').groupby('iso3').tail(1)
    top = latest.sort_values('risk', ascending=False).head(8)
    for _, r in top.iterrows():
        print(f"  {r['iso3']}: {100*r['risk']:.1f}%  (as of {r['date'].date()})")


if __name__ == "__main__":
    main()
