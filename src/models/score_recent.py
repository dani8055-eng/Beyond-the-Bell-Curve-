"""
CHUNK 3 (extend-to-present): score crisis risk through 2023 + refresh dashboard data.

Trains the best model (Logistic + tail, calibrated) on the LABELLED data (<=2016),
then produces risk estimates for EVERY country-month through 2023 -- including the
2017-2023 period that has no crisis labels (features-only prediction).

Missing-macro handling: recent rows often lack macro (2-year lag + WEO actuals
boundary). Rather than drop them, we median-impute macro using ONLY the training
(<=2016) medians -- leak-safe -- so the model can still score recent months on the
strength of the (always-present) FX features. Rows missing FX features are skipped.

HONEST NOTE for the dashboard: 2017-2023 values are PREDICTIONS with no answer key
(Laeven-Valencia labels end 2017). They show which countries the model considers
most fragile recently -- not graded accuracy.

INPUT:  data/interim/modelling_table_2023.csv
        data/interim/modelling_table.csv          (the labelled <=2016 training base)
OUTPUT: outputs/country_risk_2023.csv
        (iso3, date, year, risk_calibrated, risk_raw, actual_target, has_label)
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


def platt_fit(scores, y):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(scores.reshape(-1, 1), y)
    return lr


def main():
    full = pd.read_csv(INTERIM / "modelling_table_2023.csv")
    full['date'] = pd.to_datetime(full['date'])
    macro = [c for c in full.columns if c.startswith('macro_')]
    feat = CONVENTIONAL + TAIL + macro

    # Training base: labelled rows with all features present (<=2016)
    train = full[(full['year'] <= 2016) & full['target'].notna()].copy()
    train = train.dropna(subset=CONVENTIONAL + TAIL + ['target'])  # FX must be present
    # leak-safe macro medians from TRAIN only
    macro_medians = train[macro].median()
    train[macro] = train[macro].fillna(macro_medians)

    # Fit model on part of train, calibrate on recent train years (<=2016)
    cal_years = sorted(train['year'].unique())[-3:]
    fit_part = train[~train['year'].isin(cal_years)]
    cal_part = train[train['year'].isin(cal_years)]

    sc = StandardScaler().fit(fit_part[feat].values)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(sc.transform(fit_part[feat].values), fit_part['target'].values)
    cal_raw = clf.predict_proba(sc.transform(cal_part[feat].values))[:, 1]
    platt = platt_fit(cal_raw, cal_part['target'].values)

    # Score EVERY row through 2023 that has FX features present
    scoreable = full.dropna(subset=CONVENTIONAL + TAIL).copy()
    scoreable[macro] = scoreable[macro].fillna(macro_medians)  # impute macro w/ train medians

    raw = clf.predict_proba(sc.transform(scoreable[feat].values))[:, 1]
    cal = platt.predict_proba(raw.reshape(-1, 1))[:, 1]
    scoreable['risk_raw'] = raw
    scoreable['risk_calibrated'] = cal
    scoreable['has_label'] = scoreable['target'].notna()
    scoreable = scoreable.rename(columns={'target': 'actual_target'})

    out = scoreable[['iso3', 'date', 'year', 'risk_raw', 'risk_calibrated',
                     'actual_target', 'has_label']].sort_values(['iso3', 'date'])
    out.to_csv(OUT / "country_risk_2023.csv", index=False)

    print(f"Saved: {OUT / 'country_risk_2023.csv'}  ({len(out):,} rows)")
    print(f"Through: {out['date'].max().date()}")
    print(f"Countries: {out['iso3'].nunique()}")

    # Current (latest month) risk ranking
    latest = out.sort_values('date').groupby('iso3').tail(1)
    latest = latest.sort_values('risk_raw', ascending=False)
    print(f"\nMost fragile countries at {out['date'].max().date()} "
          f"(model ranking; 2017-2023 are predictions, not graded):")
    for _, r in latest.head(10).iterrows():
        print(f"  {r['iso3']}: rank risk {100*r['risk_calibrated']:.1f}%  "
              f"(as of {r['date'].date()})")


if __name__ == "__main__":
    main()
