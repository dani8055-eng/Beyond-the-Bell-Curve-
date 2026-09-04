"""
Core experiment: conventional vs conventional+tail features, across three models.

Walk-forward (expanding-window) validation, pooled out-of-sample predictions.
Primary metric: PR-AUC (rare-event appropriate). ROC-AUC and lift reported too.

Models: Logistic Regression, Random Forest, XGBoost.
Feature sets: Conventional(+macro), Conventional+Tail(+macro), Tail(+macro).

Missing-value handling: drop incomplete rows (baseline choice, Decision 013).
Look-ahead: each fold trains only on years strictly before the test year
(scaler fit on train only). No future data enters any prediction.

See: docs/research_decisions.md (Decisions 006, 013).

INPUT:  data/interim/modelling_table.csv
OUTPUT: prints results table (and saves to outputs/model_results.csv)
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

INTERIM = Path("data/interim")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

CONVENTIONAL = ['dep_1m','dep_mean_3m','dep_vol_3m','dep_cum_3m','dep_mean_6m',
                'dep_vol_6m','dep_cum_6m','dep_mean_12m','dep_vol_12m','dep_cum_12m']
TAIL = ['dep_skew_6m','dep_kurt_6m','dep_max_6m','dep_es90_6m','dep_skew_12m',
        'dep_kurt_12m','dep_max_12m','dep_es90_12m','extreme_count_3m',
        'extreme_count_6m','extreme_count_12m']

FIRST_TEST_YEAR = 2002
LAST_YEAR = 2016


def walk_forward(df, feat, kind, spw):
    """Expanding-window walk-forward; returns pooled PR-AUC, ROC-AUC, base rate."""
    d = df.dropna(subset=feat + ['target']).reset_index(drop=True)
    P, T = [], []
    for ty in range(FIRST_TEST_YEAR, LAST_YEAR + 1):
        tr, te = d[d['year'] < ty], d[d['year'] == ty]
        if len(te) == 0 or tr['target'].sum() == 0:
            continue
        Xtr, ytr = tr[feat].values, tr['target'].values
        Xte = te[feat].values

        if kind == 'logit':
            sc = StandardScaler().fit(Xtr)
            m = LogisticRegression(max_iter=1000, class_weight='balanced')
            m.fit(sc.transform(Xtr), ytr)
            p = m.predict_proba(sc.transform(Xte))[:, 1]
        elif kind == 'rf':
            m = RandomForestClassifier(n_estimators=300, max_depth=6,
                    class_weight='balanced', min_samples_leaf=20,
                    random_state=42, n_jobs=-1)
            m.fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
        elif kind == 'xgb':
            m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                    eval_metric='aucpr', random_state=42, n_jobs=-1)
            m.fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]

        P.extend(p); T.extend(te['target'].values)
    P, T = np.array(P), np.array(T)
    return average_precision_score(T, P), roc_auc_score(T, P), T.mean()


def main():
    df = pd.read_csv(INTERIM / "modelling_table.csv")
    macro = [c for c in df.columns if c.startswith('macro_')]
    pos = df['target'].mean(); spw = (1 - pos) / pos

    feature_sets = {
        'Conventional': CONVENTIONAL + macro,
        'Conv+Tail':    CONVENTIONAL + TAIL + macro,
        'Tail only':    TAIL + macro,
    }
    models = [('Logistic', 'logit'), ('RandomForest', 'rf')]
    if HAS_XGB:
        models.append(('XGBoost', 'xgb'))
    else:
        print("WARNING: XGBoost not installed — skipping it. "
              "Install with: pip install xgboost\n")

    rows = []
    print("=" * 74)
    print("CORE EXPERIMENT — walk-forward, pooled out-of-sample")
    print("Primary metric: PR-AUC (higher = better at catching rare crises)")
    print("=" * 74)
    print(f"{'Model':<14}{'Features':<14}{'PR-AUC':>9}{'ROC-AUC':>9}{'lift':>7}")
    print("-" * 74)
    for mname, kind in models:
        prs = {}
        for fname, feat in feature_sets.items():
            pr, roc, base = walk_forward(df, feat, kind, spw)
            prs[fname] = pr
            rows.append({'model': mname, 'features': fname,
                         'pr_auc': pr, 'roc_auc': roc, 'lift': pr / base})
            print(f"{mname:<14}{fname:<14}{pr:>9.4f}{roc:>9.4f}{pr/base:>6.2f}x")
        tail_effect = prs['Conv+Tail'] - prs['Conventional']
        pct = 100 * tail_effect / prs['Conventional']
        print(f"{'  tail effect:':<28}{tail_effect:>+9.4f}   ({pct:+.1f}% vs conventional)")
        print("-" * 74)

    pd.DataFrame(rows).to_csv(OUT / "model_results.csv", index=False)
    print(f"\nSaved: {OUT / 'model_results.csv'}")
    print(f"(base rate ~{100*base:.2f}%)")


if __name__ == "__main__":
    main()
