"""
Bootstrap test for the EMP sub-study tail effect.

Question: is the tail-feature effect at each EMP threshold (k=1.5: -7.5%,
k=2.0: +5.4%) a real pattern in the 8-country subsample, or just noise?

Method (same as the main-study robustness): build the walk-forward pooled
out-of-sample predictions for BOTH feature sets (conventional, conventional+tail)
on the SAME rows, then resample those pooled predictions 1000x and recompute the
tail effect (PR-AUC difference) each time. Report the 95% range and the fraction
of resamples where tail helped.

Interpretation:
  - If the effect stays reliably one side of zero -> real pattern in this sample.
  - If it wanders across zero -> noise; the k=1.5-vs-k=2.0 difference cannot be
    trusted with only 8 countries.

This tests whether there is a real pattern to interpret. It does NOT test the
reserve-defence MECHANISM (that stays an interpretation regardless).

INPUT:  data/interim/emp_measure.csv, data/interim/fx_features.csv
OUTPUT: prints bootstrap results per threshold; saves outputs/emp_bootstrap.csv
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

INTERIM = Path("data/interim")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

SUBSAMPLE = ['ARG', 'BRA', 'IDN', 'KOR', 'MEX', 'RUS', 'TUR', 'ZAF']
HORIZON = 6
FIRST_TEST_YEAR = 2000
N_BOOT = 1000
SEED = 42

CONVENTIONAL = ['dep_1m', 'dep_mean_3m', 'dep_vol_3m', 'dep_cum_3m', 'dep_mean_6m',
                'dep_vol_6m', 'dep_cum_6m', 'dep_mean_12m', 'dep_vol_12m', 'dep_cum_12m']
TAIL = ['dep_skew_6m', 'dep_kurt_6m', 'dep_max_6m', 'dep_es90_6m', 'dep_skew_12m',
        'dep_kurt_12m', 'dep_max_12m', 'dep_es90_12m', 'extreme_count_3m',
        'extreme_count_6m', 'extreme_count_12m']


def build_emp_target(emp, crisis_col):
    out = []
    for iso, g in emp.groupby('iso3'):
        g = g.sort_values('date').reset_index(drop=True)
        crisis = g[crisis_col].values
        n = len(g)
        target = np.zeros(n, dtype=int)
        for t in range(n):
            if crisis[t + 1: t + 1 + HORIZON].sum() > 0:
                target[t] = 1
        g = g.copy(); g['target'] = target
        out.append(g[['iso3', 'date', 'target']])
    return pd.concat(out, ignore_index=True)


def wf_preds(df, feat):
    """Walk-forward pooled predictions + truths (same row order for any feat)."""
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
    return np.array(P), np.array(T)


def main():
    emp = pd.read_csv(INTERIM / "emp_measure.csv"); emp['date'] = pd.to_datetime(emp['date'])
    fxf = pd.read_csv(INTERIM / "fx_features.csv"); fxf['date'] = pd.to_datetime(fxf['date'])
    fxf = fxf[fxf['iso3'].isin(SUBSAMPLE)]

    feat_c = CONVENTIONAL
    feat_f = CONVENTIONAL + TAIL

    rows = []
    rng = np.random.default_rng(SEED)

    print("=" * 64)
    print("EMP SUB-STUDY BOOTSTRAP — is the tail effect real or noise?")
    print("=" * 64)

    for kcol, klabel in [('emp_crisis_15', 'k=1.5'), ('emp_crisis_20', 'k=2.0')]:
        tgt = build_emp_target(emp, kcol)
        # evaluate both feature sets on the SAME rows (fuller set is stricter)
        d = fxf.merge(tgt, on=['iso3', 'date'], how='inner')
        d = d.dropna(subset=feat_f + ['target']).reset_index(drop=True)

        Pc, Tc = wf_preds(d, feat_c)
        Pf, Tf = wf_preds(d, feat_f)  # same rows

        obs = average_precision_score(Tf, Pf) - average_precision_score(Tc, Pc)

        n = len(Tc)
        diffs = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            try:
                diffs.append(average_precision_score(Tf[idx], Pf[idx])
                             - average_precision_score(Tc[idx], Pc[idx]))
            except Exception:
                pass
        diffs = np.array(diffs)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        frac_pos = 100 * (diffs > 0).mean()

        # is it reliable? reliable if the 95% range doesn't straddle 0
        reliable = (lo > 0) or (hi < 0)
        verdict = ("RELIABLE (" + ("positive" if lo > 0 else "negative") + ")") \
            if reliable else "NOISE (range crosses zero)"

        print(f"\n--- {klabel} ---")
        print(f"Observed tail effect: {obs:+.4f}")
        print(f"Bootstrap mean:       {diffs.mean():+.4f}")
        print(f"95% range:            {lo:+.4f} to {hi:+.4f}")
        print(f"Fraction positive:    {frac_pos:.1f}%")
        print(f"VERDICT: {verdict}")

        rows.append({'threshold': klabel, 'observed': obs, 'boot_mean': diffs.mean(),
                     'lo95': lo, 'hi95': hi, 'frac_positive': frac_pos,
                     'reliable': reliable})

    pd.DataFrame(rows).to_csv(OUT / "emp_bootstrap.csv", index=False)
    print(f"\nSaved: {OUT / 'emp_bootstrap.csv'}")
    print("\nIf a threshold's range crosses zero, its tail effect is NOISE in this")
    print("8-country sample -> report as 'cannot distinguish', not a real difference.")


if __name__ == "__main__":
    main()
