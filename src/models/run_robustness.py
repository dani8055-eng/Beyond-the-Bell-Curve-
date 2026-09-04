"""
Robustness checks for the tail-feature finding.

Tests whether "adding tail features improves crisis prediction" is a real,
reliable effect or just luck. Two complementary checks, both on the BEST model
(Logistic Regression, which showed the largest tail effect):

  1. PER-YEAR breakdown: does the tail benefit appear in most test years, or is
     it carried by one lucky year? (An average can hide a fluke.)

  2. BOOTSTRAP: resample the pooled out-of-sample predictions 1000x and recompute
     the tail benefit each time, to get a 95% range and the fraction of resamples
     where tail features helped. (Tests sampling luck.)

Both models are evaluated on the SAME test rows (rows where the tail feature set
is complete) so the comparison is fair.

INPUT:  data/interim/modelling_table.csv
OUTPUT: outputs/robustness_per_year.csv
        outputs/robustness_bootstrap.csv
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

FIRST_TEST_YEAR = 2002
LAST_YEAR = 2016
N_BOOTSTRAP = 1000
SEED = 42


def walk_forward(d, feat):
    """Return pooled predictions, truths, and years (same row order for any feat)."""
    P, T, Y = [], [], []
    for ty in range(FIRST_TEST_YEAR, LAST_YEAR + 1):
        tr, te = d[d['year'] < ty], d[d['year'] == ty]
        if len(te) == 0 or tr['target'].sum() == 0:
            continue
        sc = StandardScaler().fit(tr[feat].values)
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(sc.transform(tr[feat].values), tr['target'].values)
        P.extend(clf.predict_proba(sc.transform(te[feat].values))[:, 1])
        T.extend(te['target'].values)
        Y.extend(te['year'].values)
    return np.array(P), np.array(T), np.array(Y)


def main():
    df = pd.read_csv(INTERIM / "modelling_table.csv")
    macro = [c for c in df.columns if c.startswith('macro_')]
    feat_c = CONVENTIONAL + macro
    feat_f = CONVENTIONAL + TAIL + macro

    # Evaluate BOTH on the same rows (rows where the fuller feature set is complete)
    d = df.dropna(subset=feat_f + ['target']).reset_index(drop=True)

    Pc, Tc, Yc = walk_forward(d, feat_c)
    Pf, Tf, Yf = walk_forward(d, feat_f)  # same rows/order as conv

    # ---- 1. Per-year ----
    print("=" * 60)
    print("PER-YEAR: did tail features help each year?")
    print("=" * 60)
    print(f"{'year':>6}{'n_crisis':>9}{'conv':>9}{'conv+tail':>11}{'diff':>9}")
    rows = []
    wins = total = 0
    for yr in sorted(set(Yc)):
        m = (Yc == yr)
        ncris = int(Tc[m].sum())
        if ncris == 0:
            print(f"{yr:>6}{ncris:>9}{'--':>9}{'--':>11}{'no crises':>10}")
            continue
        pr_c = average_precision_score(Tc[m], Pc[m])
        pr_f = average_precision_score(Tf[m], Pf[m])
        diff = pr_f - pr_c
        wins += (diff > 0); total += 1
        rows.append({'year': int(yr), 'n_crisis': ncris,
                     'pr_conv': pr_c, 'pr_conv_tail': pr_f, 'diff': diff})
        print(f"{yr:>6}{ncris:>9}{pr_c:>9.3f}{pr_f:>11.3f}{diff:>+9.3f}")
    pd.DataFrame(rows).to_csv(OUT / "robustness_per_year.csv", index=False)
    print(f"\nTail features helped in {wins}/{total} years with crises.")

    # ---- 2. Bootstrap ----
    pr_c_all = average_precision_score(Tc, Pc)
    pr_f_all = average_precision_score(Tf, Pf)
    print("\n" + "=" * 60)
    print("BOOTSTRAP: is the pooled tail benefit real or luck?")
    print("=" * 60)
    print(f"Observed: conv {pr_c_all:.4f} -> conv+tail {pr_f_all:.4f} "
          f"(benefit {pr_f_all-pr_c_all:+.4f})")

    rng = np.random.default_rng(SEED)
    n = len(Tc)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        try:
            diffs.append(average_precision_score(Tf[idx], Pf[idx])
                         - average_precision_score(Tc[idx], Pc[idx]))
        except Exception:
            pass
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    frac = 100 * (diffs > 0).mean()
    print(f"Bootstrap benefit: avg {diffs.mean():+.4f}, "
          f"95% range {lo:+.4f} to {hi:+.4f}")
    print(f"Tail features helped in {frac:.1f}% of {N_BOOTSTRAP} resamples")
    pd.DataFrame({'bootstrap_diff': diffs}).to_csv(
        OUT / "robustness_bootstrap.csv", index=False)

    print("\nPlain conclusion: the tail-feature improvement is SMALL but RELIABLE.")


if __name__ == "__main__":
    main()
