"""
Generate CALIBRATED per-country, per-month crisis-risk probabilities.

The raw model (with balanced class weights) ranks risk well but its probabilities
are inflated (e.g. "100%"), because balancing makes it aggressive. Calibration
corrects the numbers so a stated probability means what it says: among months
scored ~p, crises actually occur about p of the time.

METHOD: Platt scaling (a logistic correction on the raw score). Chosen over
isotonic regression because the positive class is small and isotonic overfits.

LOOK-AHEAD SAFE: calibration is fit INSIDE each walk-forward step, using only
past years, then applied to the current test year. Learning the correction from
all years at once would leak future information -- deliberately avoided.

Honest expectation: calibrated probabilities are mostly LOW (crises are ~3% of
months), so even risky countries show modest percentages. A calibrated 15% for a
risky country is honest and still well above the ~3% base rate. The drama of
"100%" was the inflation; removing it is the point.

INPUT:  data/interim/modelling_table.csv
OUTPUT: outputs/country_risk_calibrated.csv  (iso3, date, year, risk_calibrated,
                                              risk_raw, actual_target)
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


def platt_fit(scores, y):
    """Fit a 1-D logistic mapping raw score -> calibrated probability."""
    lr = LogisticRegression(max_iter=1000)
    lr.fit(scores.reshape(-1, 1), y)
    return lr


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

        # split TRAIN further: fit model on the earlier part, calibrate on the
        # most recent 3 training years (still all in the past vs the test year)
        cal_years = sorted(tr['year'].unique())[-3:]
        fit_part = tr[~tr['year'].isin(cal_years)]
        cal_part = tr[tr['year'].isin(cal_years)]
        # if too little to split, fall back to using all train for both
        if len(fit_part) < 100 or cal_part['target'].sum() < 3:
            fit_part = tr; cal_part = tr

        sc = StandardScaler().fit(fit_part[feat].values)
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(sc.transform(fit_part[feat].values), fit_part['target'].values)

        # raw scores on calibration set -> fit Platt
        cal_raw = clf.predict_proba(sc.transform(cal_part[feat].values))[:, 1]
        platt = platt_fit(cal_raw, cal_part['target'].values)

        # apply to test year
        te_raw = clf.predict_proba(sc.transform(te[feat].values))[:, 1]
        te_cal = platt.predict_proba(te_raw.reshape(-1, 1))[:, 1]

        out = te[['iso3', 'date', 'year', 'target']].copy()
        out['risk_raw'] = te_raw
        out['risk_calibrated'] = te_cal
        preds.append(out)

    allpreds = pd.concat(preds, ignore_index=True).rename(columns={'target': 'actual_target'})
    allpreds = allpreds.sort_values(['iso3', 'date']).reset_index(drop=True)
    allpreds.to_csv(OUT / "country_risk_calibrated.csv", index=False)

    print(f"Saved: {OUT / 'country_risk_calibrated.csv'}  ({len(allpreds):,} rows)")
    print(f"Countries: {allpreds['iso3'].nunique()}")

    # Calibration sanity: bucket calibrated risk, show actual crisis frequency
    print("\nCalibration check (does stated risk match actual crisis frequency?):")
    b = allpreds.copy()
    b['bucket'] = pd.cut(b['risk_calibrated'], [0, .05, .1, .2, .4, 1.0])
    tab = b.groupby('bucket').agg(
        n=('actual_target', 'size'),
        stated_avg=('risk_calibrated', 'mean'),
        actual_freq=('actual_target', 'mean'))
    for idx, r in tab.iterrows():
        if r['n'] > 0:
            print(f"  risk {str(idx):<12} n={int(r['n']):5d}  "
                  f"stated~{100*r['stated_avg']:4.0f}%  actual={100*r['actual_freq']:4.0f}%")

    print("\nHighest calibrated risk, Dec 2016:")
    latest = allpreds.sort_values('date').groupby('iso3').tail(1)
    for _, r in latest.sort_values('risk_calibrated', ascending=False).head(8).iterrows():
        print(f"  {r['iso3']}: {100*r['risk_calibrated']:.1f}%  (as of {r['date'].date()})")


if __name__ == "__main__":
    main()
