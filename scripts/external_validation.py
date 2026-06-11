#!/usr/bin/env python3
"""external_validation.py - leave-one-cohort-out external validation (the real-study step)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve

REPO = Path.home() / "ISPY2-3DGCNN"
OUTDIR = REPO / "reports" / "multicohort"
RNG = np.random.default_rng(42)

def new_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))

def boot_ci(y, p, fn, needs2=True, B=2000):
    n = len(y); vals = []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        ys, ps = y[idx], p[idx]
        if needs2 and len(np.unique(ys)) < 2:
            continue
        try: vals.append(fn(ys, ps))
        except Exception: pass
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)

def load(cohort):
    f = OUTDIR / f"{cohort}_features.csv"
    if not f.exists():
        sys.exit(f"missing {f} - build the per-cohort feature table first.")
    df = pd.read_csv(f)
    if "pcr" not in df.columns:
        sys.exit(f"{f} has no 'pcr' column.")
    return df

def main():
    cohorts = sys.argv[1:]
    if len(cohorts) < 2:
        sys.exit("Usage: python external_validation.py <cohortA> <cohortB> [cohortC ...]  (need >=2)")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    data = {c: load(c) for c in cohorts}
    feats = None
    for df in data.values():
        cols = set(df.columns) - {"patient_id", "pcr"}
        feats = cols if feats is None else (feats & cols)
    feats = sorted(feats)
    if not feats:
        sys.exit("No feature columns shared across all cohorts. Harmonize feature names first.")
    print(f"shared features ({len(feats)}):", feats[:10], "..." if len(feats) > 10 else "")

    rows = []
    plt.figure(figsize=(6.5, 6))
    for test in cohorts:
        tr = pd.concat([data[c] for c in cohorts if c != test], ignore_index=True)
        te = data[test]
        Xtr, ytr = tr[feats].to_numpy(float), tr["pcr"].to_numpy(int)
        Xte, yte = te[feats].to_numpy(float), te["pcr"].to_numpy(int)
        if len(np.unique(yte)) < 2:
            print(f"[skip] held-out {test}: only one class present"); continue
        m = new_model(); m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        au, ap, br = roc_auc_score(yte, p), average_precision_score(yte, p), brier_score_loss(yte, p)
        aulo, auhi = boot_ci(yte, p, roc_auc_score)
        rows.append({"held_out_cohort": test, "n_test": len(yte), "trained_on": "+".join(c for c in cohorts if c != test),
                     "AUROC": round(au, 4), "AUROC_CI_low": round(aulo, 4), "AUROC_CI_high": round(auhi, 4),
                     "AUPRC": round(ap, 4), "Brier": round(br, 4)})
        fpr, tpr, _ = roc_curve(yte, p)
        plt.plot(fpr, tpr, lw=2, label=f"test={test} (AUROC {au:.2f})")
        print(f"held-out {test}: AUROC {au:.3f} [{aulo:.3f},{auhi:.3f}]  n={len(yte)}")

    alldf = pd.concat(data.values(), ignore_index=True)
    Xa, ya = alldf[feats].to_numpy(float), alldf["pcr"].to_numpy(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    pp = cross_val_predict(new_model(), Xa, ya, cv=cv, method="predict_proba")[:, 1]
    pau, plo, phi = roc_auc_score(ya, pp), *boot_ci(ya, pp, roc_auc_score)
    rows.append({"held_out_cohort": "POOLED_5FoldCV", "n_test": len(ya), "trained_on": "all (CV)",
                 "AUROC": round(pau, 4), "AUROC_CI_low": round(plo, 4), "AUROC_CI_high": round(phi, 4),
                 "AUPRC": round(average_precision_score(ya, pp), 4), "Brier": round(brier_score_loss(ya, pp), 4)})

    plt.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Multi-cohort external validation (leave-one-cohort-out)")
    plt.legend(loc="lower right", fontsize=9); plt.tight_layout()
    plt.savefig(OUTDIR / "external_validation_roc.png", dpi=140); plt.close()

    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "external_validation_metrics.csv", index=False)
    print("\n=== metrics ===\n" + out.to_string(index=False))
    print(f"\nwrote {OUTDIR/'external_validation_metrics.csv'} and external_validation_roc.png")

if __name__ == "__main__":
    main()
