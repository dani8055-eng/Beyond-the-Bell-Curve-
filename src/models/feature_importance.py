"""
Feature importance: which clues actually drive crisis predictions?

Two complementary views, both on Logistic Regression (the best model):

  1. STANDARDIZED COEFFICIENTS
     Direction + magnitude of each feature's effect. Fast, but can be misleading
     when features are correlated (the depreciation features overlap heavily), so
     it is reported as context, not the headline.

  2. PERMUTATION IMPORTANCE  (the headline, more trustworthy)
     For each feature: shuffle its values in the test set so it becomes noise,
     and measure how much the PR-AUC drops. A big drop = the model relied on that
     feature. Robust to correlation, because it measures each feature's UNIQUE
     contribution (a feature duplicated elsewhere shows low importance, since the
     duplicates cover for it).

Setup: train on years < 2010, test on 2010-2016 (time-based, no look-ahead).
Rows with missing features are dropped (consistent with the modelling choice).

Headline finding (2026-09-04): of all positive predictive importance, TAIL
features carry ~52%, macro ~28%, conventional ~20%. The single strongest
exchange-rate features are skewness (dep_skew_6m/12m) and extreme-move counts
(extreme_count_*) -- i.e. the tail-risk measures. This supports the project's
core hypothesis: extreme-event information carries unique crisis signal that
conventional averages do not.

INPUT:  data/interim/modelling_table.csv
OUTPUT: outputs/feature_importance_coef.csv
        outputs/feature_importance_permutation.csv
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

CONVENTIONAL = ['dep_1m','dep_mean_3m','dep_vol_3m','dep_cum_3m','dep_mean_6m',
                'dep_vol_6m','dep_cum_6m','dep_mean_12m','dep_vol_12m','dep_cum_12m']
TAIL = ['dep_skew_6m','dep_kurt_6m','dep_max_6m','dep_es90_6m','dep_skew_12m',
        'dep_kurt_12m','dep_max_12m','dep_es90_12m','extreme_count_3m',
        'extreme_count_6m','extreme_count_12m']

TRAIN_END = 2009
N_SHUFFLE = 20
SEED = 0


def main():
    df = pd.read_csv(INTERIM / "modelling_table.csv")
    macro = [c for c in df.columns if c.startswith('macro_')]
    feat = CONVENTIONAL + TAIL + macro
    group_of = {f: ('tail' if f in TAIL else 'macro' if f in macro else 'conventional')
                for f in feat}

    d = df.dropna(subset=feat + ['target']).reset_index(drop=True)
    train = d[d['year'] <= TRAIN_END]
    test = d[d['year'] > TRAIN_END]

    sc = StandardScaler().fit(train[feat].values)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf.fit(sc.transform(train[feat].values), train['target'].values)

    Xte = sc.transform(test[feat].values)
    yte = test['target'].values
    base = average_precision_score(yte, clf.predict_proba(Xte)[:, 1])

    # ---- 1. Coefficients ----
    coef = pd.DataFrame({'feature': feat, 'coef': clf.coef_[0]})
    coef['group'] = coef['feature'].map(group_of)
    coef['abs'] = coef['coef'].abs()
    coef = coef.sort_values('abs', ascending=False)
    coef.to_csv(OUT / "feature_importance_coef.csv", index=False)

    print("=" * 62)
    print("1. STANDARDIZED COEFFICIENTS (context; correlated features caveat)")
    print("   + raises crisis prob, - lowers it")
    print("=" * 62)
    print(f"{'feature':<26}{'group':<13}{'coef':>9}")
    for _, r in coef.head(10).iterrows():
        print(f"{r['feature']:<26}{r['group']:<13}{r['coef']:>+9.2f}")

    # ---- 2. Permutation importance ----
    rng = np.random.default_rng(SEED)
    rows = []
    for i, f in enumerate(feat):
        drops = []
        for _ in range(N_SHUFFLE):
            Xp = Xte.copy()
            Xp[:, i] = rng.permutation(Xp[:, i])
            drops.append(base - average_precision_score(yte, clf.predict_proba(Xp)[:, 1]))
        rows.append({'feature': f, 'group': group_of[f], 'importance': float(np.mean(drops))})
    imp = pd.DataFrame(rows).sort_values('importance', ascending=False)
    imp.to_csv(OUT / "feature_importance_permutation.csv", index=False)

    print("\n" + "=" * 62)
    print(f"2. PERMUTATION IMPORTANCE (headline). base PR-AUC = {base:.4f}")
    print("   drop in PR-AUC when the feature is shuffled to noise")
    print("=" * 62)
    print(f"{'feature':<26}{'group':<13}{'importance':>11}")
    for _, r in imp.head(12).iterrows():
        print(f"{r['feature']:<26}{r['group']:<13}{r['importance']:>+11.4f}")

    pos = imp[imp['importance'] > 0]
    tot = pos['importance'].sum()
    print("\nShare of positive predictive importance by group:")
    for g, v in pos.groupby('group')['importance'].sum().sort_values(ascending=False).items():
        print(f"  {g:<13}{100*v/tot:>4.0f}%")

    print("\nPlain conclusion: tail-risk features (skewness, extreme-move counts)")
    print("carry the majority of the useful predictive signal.")


if __name__ == "__main__":
    main()
