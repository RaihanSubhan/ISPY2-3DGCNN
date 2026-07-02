#!/usr/bin/env python3
"""metadata_multicohort.py - multi-cohort pCR model from BreastDCEDL harmonized metadata."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
REPO=Path.home()/"ISPY2-3DGCNN"; OUTDIR=REPO/"reports"/"multicohort"; RNG=np.random.default_rng(42)
CAND=["tum_vol","age","HR","HER2","TripleNeg","HER2pos","HRposHER2neg","menopause"]

def boot_ci(y,p,B=2000):
    n=len(y); v=[]
    for _ in range(B):
        i=RNG.integers(0,n,n); ys,ps=y[i],p[i]
        if len(np.unique(ys))<2: continue
        try: v.append(roc_auc_score(ys,ps))
        except Exception: pass
    return (float(np.percentile(v,2.5)),float(np.percentile(v,97.5))) if v else (np.nan,np.nan)

def model(): return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                                  LogisticRegression(max_iter=3000,class_weight="balanced"))

def evalset(Xtr,ytr,Xte,yte):
    m=model().fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]; lo,hi=boot_ci(yte,p)
    return p,{"AUROC":round(roc_auc_score(yte,p),4),"CI_low":round(lo,4),"CI_high":round(hi,4),
              "AUPRC":round(average_precision_score(yte,p),4),"Brier":round(brier_score_loss(yte,p),4)}

def main():
    src=sys.argv[1] if len(sys.argv)>1 else str(Path.home()/"BreastDCEDL"/"BreastDCEDL_metadata_min_crop.csv")
    OUTDIR.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(src)
    df=df[df["pCR"].notna()].copy(); df["pCR"]=df["pCR"].astype(int)
    feats=[c for c in CAND if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    print(f"labeled patients: {len(df)} | features: {feats}")
    print("by cohort:", df["dataset"].value_counts().to_dict())
    tflag=df["test"].astype(str).str.lower().isin(["1","1.0","true","test","yes"])
    tr,te=df[~tflag],df[tflag]
    Xtr,ytr=tr[feats].to_numpy(float),tr["pCR"].to_numpy(int)
    Xte,yte=te[feats].to_numpy(float),te["pCR"].to_numpy(int)
    rows=[]; plt.figure(figsize=(6.5,6))
    p,mt=evalset(Xtr,ytr,Xte,yte)
    rows.append({"evaluation":"official_test_split","n":len(yte),**mt})
    print(f"\n[official test split] train={len(ytr)} test={len(yte)}  AUROC {mt['AUROC']} [{mt['CI_low']},{mt['CI_high']}]")
    fpr,tpr,_=roc_curve(yte,p); plt.plot(fpr,tpr,lw=2.5,label=f"official test (AUROC {mt['AUROC']:.2f})")
    for c in te["dataset"].unique():
        sub=te[te["dataset"]==c]
        if sub["pCR"].nunique()<2: continue
        pc=model().fit(Xtr,ytr).predict_proba(sub[feats].to_numpy(float))[:,1]
        yc=sub["pCR"].to_numpy(int); lo,hi=boot_ci(yc,pc)
        rows.append({"evaluation":f"test_split::{c}","n":len(yc),"AUROC":round(roc_auc_score(yc,pc),4),
                     "CI_low":round(lo,4),"CI_high":round(hi,4),
                     "AUPRC":round(average_precision_score(yc,pc),4),"Brier":round(brier_score_loss(yc,pc),4)})
        print(f"   - {c}: n={len(yc)} AUROC {round(roc_auc_score(yc,pc),3)}")
    print("\n[leave-one-cohort-out external validation]")
    for c in df["dataset"].unique():
        tr2=df[df["dataset"]!=c]; te2=df[df["dataset"]==c]
        if te2["pCR"].nunique()<2: continue
        p2,m2=evalset(tr2[feats].to_numpy(float),tr2["pCR"].to_numpy(int),
                      te2[feats].to_numpy(float),te2["pCR"].to_numpy(int))
        rows.append({"evaluation":f"LOCO_test::{c}","n":len(te2),**m2})
        print(f"   train on others -> test {c}: n={len(te2)} AUROC {m2['AUROC']} [{m2['CI_low']},{m2['CI_high']}]")
        fpr,tpr,_=roc_curve(te2["pCR"].to_numpy(int),p2); plt.plot(fpr,tpr,lw=1.5,ls="--",label=f"LOCO {c} ({m2['AUROC']:.2f})")
    plt.plot([0,1],[0,1],color="grey",lw=1); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Multi-cohort pCR (BreastDCEDL metadata: tumor volume + clinical)")
    plt.legend(loc="lower right",fontsize=8); plt.tight_layout()
    plt.savefig(OUTDIR/"metadata_multicohort_roc.png",dpi=140); plt.close()
    pd.DataFrame(rows).to_csv(OUTDIR/"metadata_multicohort_metrics.csv",index=False)
    print(f"\nwrote metadata_multicohort_metrics.csv and metadata_multicohort_roc.png")

if __name__ == "__main__": main()
