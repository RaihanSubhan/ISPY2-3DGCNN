#!/usr/bin/env python3
"""single_cohort_validate.py - within-cohort pCR validation (5-fold CV + bootstrap CI + permutation)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
REPO=Path.home()/"ISPY2-3DGCNN"; OUTDIR=REPO/"reports"/"multicohort"; RNG=np.random.default_rng(42)

def boot_ci(y,p,fn,B=3000,needs2=True):
    n=len(y); v=[]
    for _ in range(B):
        idx=RNG.integers(0,n,n); ys,ps=y[idx],p[idx]
        if needs2 and len(np.unique(ys))<2: continue
        try: v.append(fn(ys,ps))
        except Exception: pass
    return (float(np.percentile(v,2.5)),float(np.percentile(v,97.5))) if v else (np.nan,np.nan)

def perm_p(y,p,B=5000):
    obs=roc_auc_score(y,p); c=t=0
    for _ in range(B):
        ys=RNG.permutation(y)
        if len(np.unique(ys))<2: continue
        t+=1; c+= int(roc_auc_score(ys,p)>=obs)
    return (c+1)/(t+1)

def models():
    imp=lambda *steps: make_pipeline(SimpleImputer(strategy="median"), *steps)
    return {
      "LogReg": imp(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
      "SVM_RBF": imp(StandardScaler(), SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
      "RandomForest": imp(RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)),
    }

def main():
    if len(sys.argv) < 2: sys.exit("Usage: python single_cohort_validate.py <cohort>")
    name=sys.argv[1]; f=OUTDIR/f"{name}_features.csv"
    df=pd.read_csv(f)
    if "pcr" not in df.columns: sys.exit(f"{f} has no 'pcr' - run merge_labels.py first")
    df=df.dropna(subset=["pcr"]); df["pcr"]=df["pcr"].astype(int)
    feats=[c for c in df.columns if c not in ("patient_id","pcr") and df[c].nunique(dropna=True)>1]
    X=df[feats].to_numpy(float); y=df["pcr"].to_numpy(int)
    print(f"{name}: {len(df)} labeled patients | pCR+={int(y.sum())} pCR-={int((y==0).sum())} | {len(feats)} features")
    if len(np.unique(y))<2 or min((y==0).sum(),(y==1).sum())<3:
        sys.exit("Not enough per-class samples for 5-fold CV.")
    cv=StratifiedKFold(5,shuffle=True,random_state=42); rows=[]; best=None
    for nm,mdl in models().items():
        p=cross_val_predict(mdl,X,y,cv=cv,method="predict_proba")[:,1]
        au,ap,br=roc_auc_score(y,p),average_precision_score(y,p),brier_score_loss(y,p)
        lo,hi=boot_ci(y,p,roc_auc_score); pv=perm_p(y,p)
        rows.append({"model":nm,"n":len(y),"AUROC":round(au,4),"AUROC_CI_low":round(lo,4),
                     "AUROC_CI_high":round(hi,4),"AUPRC":round(ap,4),"Brier":round(br,4),"perm_p":round(pv,4)})
        print(f"  {nm:14s} AUROC {au:.3f} [{lo:.3f},{hi:.3f}]  perm_p={pv:.3f}")
        if best is None or au>best[1]: best=(nm,au,y,p)
    pd.DataFrame(rows).to_csv(OUTDIR/f"{name}_validation_metrics.csv",index=False)
    nm,au,yb,pb=best; fpr,tpr,_=roc_curve(yb,pb)
    plt.figure(figsize=(6,6)); plt.plot(fpr,tpr,lw=2,label=f"{nm} (AUROC {au:.2f})")
    plt.plot([0,1],[0,1],"--",color="grey"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(f"{name} within-cohort pCR ({len(yb)} patients, 5-fold CV)")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(OUTDIR/f"{name}_validation_roc.png",dpi=140); plt.close()
    print(f"\nwrote {name}_validation_metrics.csv and {name}_validation_roc.png")

if __name__ == "__main__": main()
