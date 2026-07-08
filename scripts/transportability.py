#!/usr/bin/env python3
"""transportability.py - cross-cohort transportability + harmonization study for pCR."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import rankdata, norm
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
REPO=Path.home()/"ISPY2-3DGCNN"; OUT=REPO/"reports"/"multicohort"
SRC=Path.home()/"BreastDCEDL"/"BreastDCEDL_metadata_min_crop.csv"
FEATS=["tum_vol","age","HR","HER2","TripleNeg"]
def model(): return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                                  LogisticRegression(max_iter=3000,class_weight="balanced"))
def auroc(y,p): return roc_auc_score(y,p) if len(np.unique(y))>1 else np.nan
def pcz(df,feats):
    d=df.copy(); d[feats]=d[feats].astype(float)
    for c in d["dataset"].unique():
        m=d["dataset"]==c; d.loc[m,feats]=StandardScaler().fit_transform(d.loc[m,feats].to_numpy(float))
    return d
def pcrank(df,feats):
    d=df.copy(); d[feats]=d[feats].astype(float)
    for c in d["dataset"].unique():
        m=d["dataset"]==c
        for f in feats:
            x=d.loc[m,f].to_numpy(float); n=len(x); r=rankdata(x); d.loc[m,f]=norm.ppf((r-0.5)/n)
    return d
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(SRC); df=df[df["pCR"].notna()].copy(); df["pCR"]=df["pCR"].astype(int)
    feats=[f for f in FEATS if f in df.columns]; cohorts=sorted(df["dataset"].unique())
    cv=StratifiedKFold(5,shuffle=True,random_state=42)
    print(f"{len(df)} labeled | cohorts {cohorts} | features {feats}")
    M=pd.DataFrame(index=cohorts,columns=cohorts,dtype=float)
    for tr in cohorts:
        for te in cohorts:
            dtr=df[df["dataset"]==tr]; dte=df[df["dataset"]==te]
            if tr==te:
                p=cross_val_predict(model(),dtr[feats],dtr["pCR"],cv=cv,method="predict_proba")[:,1]
                M.loc[tr,te]=round(auroc(dtr["pCR"].to_numpy(int),p),3)
            else:
                m=model().fit(dtr[feats],dtr["pCR"]); p=m.predict_proba(dte[feats])[:,1]
                M.loc[tr,te]=round(auroc(dte["pCR"].to_numpy(int),p),3)
    M.to_csv(OUT/"transportability_matrix.csv")
    diag=np.nanmean([M.loc[c,c] for c in cohorts]); off=np.nanmean([M.loc[a,b] for a in cohorts for b in cohorts if a!=b])
    print("\nTransportability AUROC (rows=train, cols=test):\n"+M.to_string())
    print(f"mean IN-domain {diag:.3f}  vs  CROSS-domain {off:.3f}   (transport gap {diag-off:.3f})")
    fig,ax=plt.subplots(figsize=(5.4,4.6)); im=ax.imshow(M.to_numpy(float),cmap="RdYlGn",vmin=0.5,vmax=0.85)
    ax.set_xticks(range(len(cohorts))); ax.set_xticklabels(cohorts); ax.set_yticks(range(len(cohorts))); ax.set_yticklabels(cohorts)
    ax.set_xlabel("test cohort"); ax.set_ylabel("train cohort")
    for i in range(len(cohorts)):
        for j in range(len(cohorts)): ax.text(j,i,f"{M.iloc[i,j]:.2f}",ha="center",va="center",fontsize=12)
    ax.set_title("Cross-cohort transportability (AUROC)"); fig.colorbar(im,fraction=0.046); fig.tight_layout()
    fig.savefig(OUT/"transportability_heatmap.png",dpi=140); plt.close()
    rows=[]
    for i,a in enumerate(cohorts):
        for b in cohorts[i+1:]:
            sub=df[df["dataset"].isin([a,b])].copy(); y=(sub["dataset"]==b).astype(int).to_numpy()
            p=cross_val_predict(model(),sub[feats],y,cv=cv,method="predict_proba")[:,1]
            rows.append({"pair":f"{a} vs {b}","domain_AUROC":round(auroc(y,p),3)})
    ds=pd.DataFrame(rows); ds.to_csv(OUT/"domain_shift.csv",index=False)
    print("\nDomain shift (cohort-classifier AUROC; 0.5=identical, 1.0=totally separable):\n"+ds.to_string(index=False))
    pd.DataFrame([{**{"feature":f},**{f"mean_{c}":round(df[df['dataset']==c][f].mean(),3) for c in cohorts}} for f in feats]).to_csv(OUT/"feature_means_by_cohort.csv",index=False)
    def loco(d):
        res={}
        for te in cohorts:
            tr=d[d["dataset"]!=te]; td=d[d["dataset"]==te]
            m=model().fit(tr[feats],tr["pCR"]); p=m.predict_proba(td[feats])[:,1]
            res[te]=round(auroc(td["pCR"].to_numpy(int),p),3)
        res["mean"]=round(np.nanmean([v for k,v in res.items()]),3); return res
    H=pd.DataFrame({"raw":loco(df),"per-cohort z-score":loco(pcz(df,feats)),"per-cohort rank-norm":loco(pcrank(df,feats))}).T
    H.to_csv(OUT/"harmonization_comparison.csv")
    print("\nHarmonization effect on leave-one-cohort-out AUROC:\n"+H.to_string())
    print("\nwrote transportability_matrix.csv, transportability_heatmap.png, domain_shift.csv, harmonization_comparison.csv, feature_means_by_cohort.csv")
if __name__=="__main__": main()
