#!/usr/bin/env python3
"""component2_selective.py - trust-aware pCR: OOD detection + conformal selective prediction."""
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
REPO=Path.home()/"ISPY2-3DGCNN"; OUT=REPO/"reports"/"multicohort"
SRC=Path.home()/"BreastDCEDL"/"BreastDCEDL_metadata_min_crop.csv"
FEATS=["tum_vol","age","HR","HER2","TripleNeg"]; RNG=42
def maha_fit(X,eps=1e-4):
    mu=X.mean(0); C=np.cov(X,rowvar=False)+np.eye(X.shape[1])*eps
    return mu, np.linalg.pinv(C)
def maha(X,mu,Ci):
    d=X-mu; return np.sqrt(np.einsum('ij,jk,ik->i',d,Ci,d))
def risk_coverage(conf,correct):
    o=np.argsort(-conf); c=correct[o]
    return np.arange(1,len(c)+1)/len(c), np.cumsum(c)/np.arange(1,len(c)+1)
def calib_threshold(conf,correct,target_acc,min_n=15):
    for t in np.sort(np.unique(conf)):
        m=conf>=t
        if m.sum()>=min_n and correct[m].mean()>=target_acc: return float(t)
    return float(conf.max())
def main():
    TARGET_ACC=0.75
    df=pd.read_csv(SRC); df=df[df["pCR"].notna()].copy(); df["pCR"]=df["pCR"].astype(int)
    feats=[f for f in FEATS if f in df.columns]
    ispy=df[df["dataset"].isin(["spy1","spy2"])].copy(); duke=df[df["dataset"]=="duke"].copy()
    print(f"in-distribution (I-SPY1+I-SPY2): {len(ispy)} | OOD (Duke): {len(duke)}")
    imp=SimpleImputer(strategy="median").fit(ispy[feats].astype(float))
    Xispy=imp.transform(ispy[feats].astype(float)); Xduke=imp.transform(duke[feats].astype(float))
    yispy=ispy["pCR"].to_numpy(int); yduke=duke["pCR"].to_numpy(int)
    Xtr,Xtmp,ytr,ytmp=train_test_split(Xispy,yispy,test_size=0.4,stratify=yispy,random_state=RNG)
    Xca,Xte,yca,yte=train_test_split(Xtmp,ytmp,test_size=0.5,stratify=ytmp,random_state=RNG)
    sc=StandardScaler().fit(Xtr); Xtr,Xca,Xte,Xd=sc.transform(Xtr),sc.transform(Xca),sc.transform(Xte),sc.transform(Xduke)
    clf=LogisticRegression(max_iter=2000,class_weight="balanced").fit(Xtr,ytr)
    mu,Ci=maha_fit(Xtr)
    ood_te=maha(Xte,mu,Ci); ood_d=maha(Xd,mu,Ci)
    ood_auroc=roc_auc_score(np.r_[np.zeros(len(ood_te)),np.ones(len(ood_d))],np.r_[ood_te,ood_d])
    tau_ood=np.percentile(maha(Xtr,mu,Ci),95)
    def conf_pred(X):
        p=clf.predict_proba(X)[:,1]; return np.maximum(p,1-p),(p>=0.5).astype(int)
    conf_ca,yhat_ca=conf_pred(Xca); conf_te,yhat_te=conf_pred(Xte); conf_d,yhat_d=conf_pred(Xd)
    tau=calib_threshold(conf_ca,(yhat_ca==yca).astype(int),TARGET_ACC)
    acc_te_full=(yhat_te==yte).mean()
    keep_te=(conf_te>=tau); cov_te=keep_te.mean(); acc_kept=(yhat_te[keep_te]==yte[keep_te]).mean() if keep_te.sum() else np.nan
    accept_d=(conf_d>=tau)&(maha(Xd,mu,Ci)<=tau_ood)
    duke_abstain=1-accept_d.mean(); acc_d_kept=(yhat_d[accept_d]==yduke[accept_d]).mean() if accept_d.sum() else np.nan
    duke_flagged_ood=(maha(Xd,mu,Ci)>tau_ood).mean()
    rows=[
      {"metric":"OOD detection AUROC (I-SPY test vs Duke)","value":round(ood_auroc,3)},
      {"metric":"Duke patients flagged OOD (frac)","value":round(duke_flagged_ood,3)},
      {"metric":"I-SPY test accuracy (no abstention)","value":round(acc_te_full,3)},
      {"metric":"I-SPY test coverage at guarantee","value":round(cov_te,3)},
      {"metric":"I-SPY test accuracy on accepted","value":round(acc_kept,3)},
      {"metric":"Duke abstention rate","value":round(duke_abstain,3)},
      {"metric":"Duke accuracy on accepted","value":round(acc_d_kept,3) if accept_d.sum() else "n/a (all abstained)"},
    ]
    R=pd.DataFrame(rows); OUT.mkdir(parents=True,exist_ok=True); R.to_csv(OUT/"component2_selective.csv",index=False)
    print("\n"+R.to_string(index=False))
    fig,ax=plt.subplots(1,2,figsize=(11,4.4))
    ax[0].hist(ood_te,bins=25,alpha=0.7,label="I-SPY test (in-dist)",color="#2f8f5b",density=True)
    ax[0].hist(ood_d,bins=25,alpha=0.7,label="Duke (routine care)",color="#b23a5b",density=True)
    ax[0].axvline(tau_ood,color="k",ls="--",lw=1,label="OOD threshold (95th pct)")
    ax[0].set_xlabel("distance to training distribution (Mahalanobis)"); ax[0].set_ylabel("density")
    ax[0].set_title(f"OOD detection (AUROC {ood_auroc:.2f})"); ax[0].legend(fontsize=8)
    cov,acc=risk_coverage(conf_te,(yhat_te==yte).astype(int))
    ax[1].plot(cov,acc,lw=2,color="#2f5f9e"); ax[1].axhline(TARGET_ACC,color="grey",ls="--",lw=1,label=f"target {TARGET_ACC:.0%}")
    ax[1].scatter([cov_te],[acc_kept],color="#b23a5b",zorder=5,label=f"operating point ({cov_te:.0%} cov)")
    ax[1].set_xlabel("coverage (fraction predicted on)"); ax[1].set_ylabel("accuracy on accepted (I-SPY test)")
    ax[1].set_title("Risk-coverage: accuracy rises as model abstains"); ax[1].legend(fontsize=8); ax[1].set_ylim(0.4,1.02)
    fig.tight_layout(); fig.savefig(OUT/"component2_selective.png",dpi=140); plt.close()
    print("\nwrote component2_selective.csv and component2_selective.png")
if __name__=="__main__": main()
