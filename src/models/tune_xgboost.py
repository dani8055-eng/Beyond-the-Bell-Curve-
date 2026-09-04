"""
Tune XGBoost for the currency-crisis task WITHOUT look-ahead leakage.

Standard hyperparameter search (random CV) shuffles data and would train on the
future to pick settings — the exact leak we avoid everywhere else. So tuning here
is TIME-AWARE:

  - Train period:      years  < 2010   (fit candidate models)
  - Validation period: 2010-2013        (score candidates, pick the best dials)
  - Test period:       2014-2016         (untouched during tuning)

The winning dials are chosen using only train+validation (all <= 2013), so the
2014-2016 test years never influence tuning.

Then we re-run the FULL walk-forward comparison (as in run_core_experiment.py)
using the tuned XGBoost, so you can see whether tuning changes the story vs the
default-XGBoost result (Conventional 0.025 -> Conv+Tail 0.027).

INPUT:  data/interim/modelling_table.csv
OUTPUT: outputs/xgb_tuning_results.csv  (grid scores)
"""

import warnings
warnings.filterwarnings('ignore')
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

INTERIM = Path("data/interim")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

CONVENTIONAL = ['dep_1m','dep_mean_3m','dep_vol_3m','dep_cum_3m','dep_mean_6m',
                'dep_vol_6m','dep_cum_6m','dep_mean_12m','dep_vol_12m','dep_cum_12m']
TAIL = ['dep_skew_6m','dep_kurt_6m','dep_max_6m','dep_es90_6m','dep_skew_12m',
        'dep_kurt_12m','dep_max_12m','dep_es90_12m','extreme_count_3m',
        'extreme_count_6m','extreme_count_12m']

# Time-based tuning split
TRAIN_END = 2009          # train: years <= 2009
VALID_YEARS = (2010, 2013)  # validation: 2010-2013 inclusive

# Small, sensible grid (tuned for rare, noisy data: shallower trees, strong reg)
GRID = {
    'max_depth':        [2, 3, 4],
    'learning_rate':    [0.03, 0.1],
    'n_estimators':     [150, 400],
    'min_child_weight': [1, 5],
    'reg_lambda':       [1.0, 5.0],
    'subsample':        [0.8],
    'colsample_bytree': [0.8],
}


def make_xgb(params, spw):
    return XGBClassifier(
        **params, scale_pos_weight=spw, eval_metric='aucpr',
        random_state=42, n_jobs=-1, verbosity=0,
    )


def tune(df, feat):
    """Grid search using time-based train/validation. Returns best params + table."""
    d = df.dropna(subset=feat + ['target']).reset_index(drop=True)
    tr = d[d['year'] <= TRAIN_END]
    va = d[(d['year'] >= VALID_YEARS[0]) & (d['year'] <= VALID_YEARS[1])]
    spw = (tr['target'] == 0).sum() / max(1, (tr['target'] == 1).sum())

    Xtr, ytr = tr[feat].values, tr['target'].values
    Xva, yva = va[feat].values, va['target'].values

    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"Searching {len(combos)} hyperparameter combinations "
          f"(train<= {TRAIN_END}, valid {VALID_YEARS[0]}-{VALID_YEARS[1]})...",
          flush=True)

    rows = []
    best = (-1, None)
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        m = make_xgb(params, spw)
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xva)[:, 1]
        pr = average_precision_score(yva, p)
        rows.append({**params, 'valid_pr_auc': pr})
        if pr > best[0]:
            best = (pr, params)
        if i % 8 == 0:
            print(f"  ...{i}/{len(combos)} done (best valid PR-AUC so far {best[0]:.4f})",
                  flush=True)

    return best[1], best[0], pd.DataFrame(rows).sort_values('valid_pr_auc', ascending=False)


def walk_forward_xgb(df, feat, params, first_test=2002, last=2016):
    d = df.dropna(subset=feat + ['target']).reset_index(drop=True)
    P, T = [], []
    for ty in range(first_test, last + 1):
        tr, te = d[d['year'] < ty], d[d['year'] == ty]
        if len(te) == 0 or tr['target'].sum() == 0:
            continue
        spw = (tr['target'] == 0).sum() / max(1, (tr['target'] == 1).sum())
        m = make_xgb(params, spw)
        m.fit(tr[feat].values, tr['target'].values)
        P.extend(m.predict_proba(te[feat].values)[:, 1]); T.extend(te['target'].values)
    P, T = np.array(P), np.array(T)
    return average_precision_score(T, P), roc_auc_score(T, P), T.mean()


def main():
    df = pd.read_csv(INTERIM / "modelling_table.csv")
    macro = [c for c in df.columns if c.startswith('macro_')]

    feat_conv = CONVENTIONAL + macro
    feat_full = CONVENTIONAL + TAIL + macro

    print("=" * 70, flush=True)
    print("TUNING XGBOOST (time-aware, no leakage)", flush=True)
    print("=" * 70, flush=True)

    # Tune on the FULL feature set (conventional + tail)
    best_params, best_valid, table = tune(df, feat_full)
    table.to_csv(OUT / "xgb_tuning_results.csv", index=False)
    print(f"\nBest params (by validation PR-AUC {best_valid:.4f}):", flush=True)
    for k, v in best_params.items():
        print(f"    {k}: {v}", flush=True)

    # Re-run full walk-forward with tuned params, both feature sets
    print("\n" + "=" * 70, flush=True)
    print("TUNED XGBOOST — walk-forward, pooled OOS (2002-2016)", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Features':<16}{'PR-AUC':>9}{'ROC-AUC':>9}{'lift':>7}", flush=True)
    print("-" * 70, flush=True)
    prs = {}
    for name, feat in [('Conventional', feat_conv), ('Conv+Tail', feat_full)]:
        pr, roc, base = walk_forward_xgb(df, feat, best_params)
        prs[name] = pr
        print(f"{name:<16}{pr:>9.4f}{roc:>9.4f}{pr/base:>6.2f}x", flush=True)
    eff = prs['Conv+Tail'] - prs['Conventional']
    print("-" * 70, flush=True)
    print(f"tail effect: {eff:+.4f} ({100*eff/prs['Conventional']:+.1f}%)", flush=True)
    print(f"\nCompare to DEFAULT XGBoost: Conventional 0.0254 -> Conv+Tail 0.0272", flush=True)
    print(f"(base rate ~{100*base:.2f}%)", flush=True)


if __name__ == "__main__":
    main()
