#!/usr/bin/env python3
"""external_validation.py - leave-one-cohort-out validation with per-cohort harmonization."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
REPO=Path.home()/"ISPY2-3DGCNN"; OUTDIR=REPO/"reports"/"multicohort"; RNG=np.random.default_rng(42)

def boot_ci(y,p,B=2000):
    n=len(y); v=[]
    for _ in range(B):
        idx=RNG.integers(0,n,n); ys,ps=y[idx],p[idx]
        if len(np.unique(ys))<2: continue
        try: v.append(roc_auc_score(ys,ps))
        except Exception: pass
    return (float(np.percentile(v,2.5)),float(np.percentile(v,97.5))) if v else (np.nan,np.nan)

def load(c):
    f=OUTDIR/f"{c}_features.csv"
    if not f.exists(): sys.exit(f"missing {f}")
    df=pd.read_csv(f)
    if "pcr" not in df.columns: sys.exit(f"{f} has no 'pcr' column")
    return df

def main():
    cohorts=sys.argv[1:]
    if len(cohorts)<2: sys.exit("Usage: python external_validation.py <cohortA> <cohortB> [..]")
    data={c:load(c) for c in cohorts}
    feats=None
    for df in data.values():
        cols=set(df.columns)-{"patient_id","pcr"}
        feats=cols if feats is None else (feats & cols)
    feats=sorted(feats)
    if not feats: sys.exit("No shared feature columns across cohorts.")
    print(f"shared, harmonized features ({len(feats)}): {feats}")
    Z={}
    for c,df in data.items():
        d=df.dropna(subset=feats+["pcr"]).copy(); d["pcr"]=d["pcr"].astype(int)
        d[feats]=StandardScaler().fit_transform(d[feats].to_numpy(float)); Z[c]=d
        print(f"  {c}: {len(d)} labeled patients (pCR+={int(d['pcr'].sum())})")
    rows=[]; plt.figure(figsize=(6.5,6))
    for test in cohorts:
        tr=pd.concat([Z[c] for c in cohorts if c!=test],ignore_index=True); te=Z[test]
        m=LogisticRegression(max_iter=2000,class_weight="balanced").fit(tr[feats],tr["pcr"])
        p=m.predict_proba(te[feats])[:,1]; y=te["pcr"].to_numpy(int)
        if len(np.unique(y))<2: print(f"[skip] {test}: one class"); continue
        au=roc_auc_score(y,p); lo,hi=boot_ci(y,p)
        rows.append({"held_out":test,"n_test":len(y),"trained_on":"+".join(c for c in cohorts if c!=test),
                     "AUROC":round(au,4),"AUROC_CI_low":round(lo,4),"AUROC_CI_high":round(hi,4),
                     "AUPRC":round(average_precision_score(y,p),4),"Brier":round(brier_score_loss(y,p),4)})
        fpr,tpr,_=roc_curve(y,p); plt.plot(fpr,tpr,lw=2,label=f"test={test} (AUROC {au:.2f})")
        print(f"held-out {test}: AUROC {au:.3f} [{lo:.3f},{hi:.3f}]  n={len(y)}")
    plt.plot([0,1],[0,1],"--",color="grey"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Two-cohort external validation (train on one, test on the other)")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(OUTDIR/"external_validation_roc.png",dpi=140); plt.close()
    pd.DataFrame(rows).to_csv(OUTDIR/"external_validation_metrics.csv",index=False)
    print("\n"+pd.DataFrame(rows).to_string(index=False))
    print(f"\nwrote external_validation_metrics.csv and external_validation_roc.png")
if __name__ == "__main__": main()
