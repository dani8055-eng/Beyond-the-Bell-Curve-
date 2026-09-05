"""
EMP sub-study: does the tail-feature finding survive an alternative crisis
definition (Exchange Market Pressure) on the reserves-available subsample?

SCOPE (Decision 017): 8 countries with monthly FRED reserves — ARG, BRA, IDN,
KOR, MEX, RUS, TUR, ZAF. Self-contained; separate from the main 110-country
Laeven-Valencia analysis. Small sample => results are INDICATIVE, not conclusive.

STEP 1 — EMP target: EMP crisis months come from build_emp.py. Because EMP is
monthly, we can use a TRUE six-month-ahead window (cleaner than the annual LV
approximation): for each month t, target=1 if an EMP crisis occurs in t+1..t+6.

STEP 2 — Prediction comparison: same features as the main study (conventional vs
conventional+tail), walk-forward validation, pooled out-of-sample PR-AUC. Run for
both EMP thresholds (k=1.5 and k=2.0).

This tests H1 (conventional weak) and H2 (tail features help) under the EMP
definition, on the same countries — a clean, self-contained robustness check.

INPUT:  data/interim/emp_measure.csv   (emp_crisis_15, emp_crisis_20)
        data/interim/fx_features.csv    (the 21 features)
OUTPUT: prints results; saves outputs/emp_substudy_results.csv
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

INTERIM = Path("data/interim")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

SUBSAMPLE = ['ARG', 'BRA', 'IDN', 'KOR', 'MEX', 'RUS', 'TUR', 'ZAF']
HORIZON = 6
FIRST_TEST_YEAR = 2000  # subsample is small; start test later so train has crises

CONVENTIONAL = ['dep_1m', 'dep_mean_3m', 'dep_vol_3m', 'dep_cum_3m', 'dep_mean_6m',
                'dep_vol_6m', 'dep_cum_6m', 'dep_mean_12m', 'dep_vol_12m', 'dep_cum_12m']
TAIL = ['dep_skew_6m', 'dep_kurt_6m', 'dep_max_6m', 'dep_es90_6m', 'dep_skew_12m',
        'dep_kurt_12m', 'dep_max_12m', 'dep_es90_12m', 'extreme_count_3m',
        'extreme_count_6m', 'extreme_count_12m']


def build_emp_target(emp, crisis_col):
    """For each month t, target=1 if an EMP crisis (crisis_col) occurs in t+1..t+6."""
    emp = emp.sort_values(['iso3', 'date']).reset_index(drop=True)
    out = []
    for iso, g in emp.groupby('iso3'):
        g = g.sort_values('date').reset_index(drop=True)
        crisis = g[crisis_col].values
        n = len(g)
        target = np.zeros(n, dtype=int)
        for t in range(n):
            window = crisis[t + 1: t + 1 + HORIZON]
            if window.sum() > 0:
                target[t] = 1
        g = g.copy()
        g['target'] = target
        out.append(g[['iso3', 'date', 'target']])
    return pd.concat(out, ignore_index=True)


def walk_forward(df, feat):
    d = df.dropna(subset=feat + ['target']).reset_index(drop=True)
    P, T = [], []
    for ty in range(FIRST_TEST_YEAR, 2017):
        tr, te = d[d['year'] < ty], d[d['year'] == ty]
        if len(te) == 0 or tr['target'].sum() == 0:
            continue
        sc = StandardScaler().fit(tr[feat].values)
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(sc.transform(tr[feat].values), tr['target'].values)
        P.extend(clf.predict_proba(sc.transform(te[feat].values))[:, 1])
        T.extend(te['target'].values)
    P, T = np.array(P), np.array(T)
    if T.sum() == 0:
        return np.nan, np.nan, T.mean(), len(T)
    return average_precision_score(T, P), roc_auc_score(T, P), T.mean(), len(T)


def main():
    emp = pd.read_csv(INTERIM / "emp_measure.csv")
    emp['date'] = pd.to_datetime(emp['date'])
    fxf = pd.read_csv(INTERIM / "fx_features.csv")
    fxf['date'] = pd.to_datetime(fxf['date'])
    fxf = fxf[fxf['iso3'].isin(SUBSAMPLE)]

    feat_conv = CONVENTIONAL
    feat_full = CONVENTIONAL + TAIL

    rows = []
    print("=" * 68)
    print("EMP SUB-STUDY — 8 countries, walk-forward, pooled OOS PR-AUC")
    print("H1: is conventional weak?   H2: do tail features help?")
    print("=" * 68)

    for kcol, klabel in [('emp_crisis_15', 'k=1.5'), ('emp_crisis_20', 'k=2.0')]:
        tgt = build_emp_target(emp, kcol)
        df = fxf.merge(tgt, on=['iso3', 'date'], how='inner')
        base_rate = df['target'].mean()

        print(f"\n--- EMP threshold {klabel}  (positive rate {100*base_rate:.1f}%, "
              f"{int(df['target'].sum())} positive months) ---")
        print(f"{'Features':<16}{'PR-AUC':>9}{'ROC-AUC':>9}{'lift':>7}")
        prs = {}
        for name, feat in [('Conventional', feat_conv), ('Conv+Tail', feat_full)]:
            pr, roc, base, n = walk_forward(df, feat)
            prs[name] = pr
            rows.append({'threshold': klabel, 'features': name,
                         'pr_auc': pr, 'roc_auc': roc, 'base_rate': base})
            print(f"{name:<16}{pr:>9.4f}{roc:>9.4f}{pr/base:>6.2f}x")
        eff = prs['Conv+Tail'] - prs['Conventional']
        print(f"tail effect: {eff:+.4f} ({100*eff/prs['Conventional']:+.1f}%)")

    pd.DataFrame(rows).to_csv(OUT / "emp_substudy_results.csv", index=False)
    print(f"\nSaved: {OUT / 'emp_substudy_results.csv'}")
    print("\nReminder: 8-country subsample -> indicative, not conclusive.")


if __name__ == "__main__":
    main()
