#!/usr/bin/env python3
"""extract_features_nifti.py - radiomics-style features from BreastDCEDL I-SPY1 NIfTI + masks."""
import os, sys, glob, re
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
from scipy.stats import skew, kurtosis

HOME=Path.home(); REPO=HOME/"ISPY2-3DGCNN"; OUTDIR=REPO/"reports"/"multicohort"
BASE=HOME/"breastdcedl_data"/"spy1_unpacked"/"BreastDCEDL_spy1"
DCE_DIR=BASE/"spt1_dce"; MASK_DIR=BASE/"spy1_mask"
META=HOME/"BreastDCEDL"/"BreastDCEDL_metadata_min_crop.csv"

def load(p):
    im=nib.load(str(p)); return np.asanyarray(im.dataobj).astype(np.float32), tuple(float(z) for z in im.header.get_zooms()[:3])
def numkey(s):
    r=re.findall(r"\d+",str(s)); return str(int(max(r,key=len))) if r else str(s)
def pick_post_pre(pid):
    fs=glob.glob(str(DCE_DIR/f"{pid}_spy1_vis1_acq*.nii.gz"))
    if not fs: return None,None
    idx={int(re.search(r'acq(\d+)',f).group(1)):f for f in fs}
    return (idx.get(1) or idx.get(max(idx))), idx.get(0)
def shape_feats(mask,zoom):
    f={}; v=int(mask.sum()); voxvol=float(np.prod(zoom))
    f["volume_mm3"]=round(v*voxvol,2); f["n_tumor_slices"]=int((mask.reshape(mask.shape[0],-1).sum(1)>0).sum())
    if v==0:
        for k in["bbox_max","bbox_mid","bbox_min","elongation","surface_area","sphericity"]: f[k]=np.nan
        return f
    c=np.argwhere(mask); ext=sorted((c.max(0)-c.min(0)+1).tolist(),reverse=True)
    while len(ext)<3: ext.append(0)
    f["bbox_max"],f["bbox_mid"],f["bbox_min"]=ext[0],ext[1],ext[2]
    f["elongation"]=round(ext[2]/ext[0],4) if ext[0] else np.nan
    try:
        from skimage.measure import marching_cubes, mesh_surface_area
        vt,fc,_,_=marching_cubes(mask.astype(float),level=0.5,spacing=zoom)
        sa=float(mesh_surface_area(vt,fc)); f["surface_area"]=round(sa,2)
        f["sphericity"]=round((np.pi**(1/3)*(6*v*voxvol)**(2/3))/sa,4) if sa>0 else np.nan
    except Exception:
        f["surface_area"]=np.nan; f["sphericity"]=np.nan
    return f
def firstorder(vals,pref):
    ks=["mean","std","median","p10","p90","skew","kurtosis","energy","entropy","iqr"]
    if len(vals)==0: return {f"{pref}_{k}":np.nan for k in ks}
    f={f"{pref}_mean":round(float(np.mean(vals)),4),f"{pref}_std":round(float(np.std(vals)),4),
       f"{pref}_median":round(float(np.median(vals)),4),f"{pref}_p10":round(float(np.percentile(vals,10)),4),
       f"{pref}_p90":round(float(np.percentile(vals,90)),4),
       f"{pref}_skew":round(float(skew(vals)),4) if len(vals)>2 else np.nan,
       f"{pref}_kurtosis":round(float(kurtosis(vals)),4) if len(vals)>3 else np.nan,
       f"{pref}_energy":round(float(np.sum(vals.astype(np.float64)**2)/len(vals)),2),
       f"{pref}_iqr":round(float(np.percentile(vals,75)-np.percentile(vals,25)),4)}
    h,_=np.histogram(vals,bins=32); pr=h/h.sum(); pr=pr[pr>0]
    f[f"{pref}_entropy"]=round(float(-(pr*np.log2(pr)).sum()),4)
    return f
def main():
    test="--test" in sys.argv; OUTDIR.mkdir(parents=True,exist_ok=True)
    masks=sorted(glob.glob(str(MASK_DIR/"ISPY1_*_mask.nii.gz")))
    if not masks: sys.exit(f"no masks under {MASK_DIR}")
    meta=pd.read_csv(META); meta=meta[meta["dataset"].astype(str).str.lower().str.contains("spy1")]
    lab=dict(zip(meta["pid"].map(numkey), meta["pCR"]))
    print(f"{len(masks)} I-SPY1 masks | {len(lab)} spy1 labels in metadata")
    if test: masks=masks[:1]
    rows=[]
    for i,mp in enumerate(masks):
        pid=os.path.basename(mp).replace("_spy1_vis1_mask.nii.gz","")
        post,pre=pick_post_pre(pid)
        if post is None:
            if test: print(f"  no DCE image for {pid}")
            continue
        mask,zoom=load(mp); mask=mask>0; pim,_=load(post)
        if pim.shape!=mask.shape:
            if test: print(f"  shape mismatch {pid}: img{pim.shape} vs mask{mask.shape}")
            continue
        rec={"patient_id":pid}; rec.update(shape_feats(mask,zoom)); rec.update(firstorder(pim[mask],"post"))
        if pre:
            prim,_=load(pre)
            if prim.shape==mask.shape:
                pv=prim[mask]; qv=pim[mask]; enh=(qv-pv)/(np.abs(pv)+1e-6)
                rec["enh_mean"]=round(float(np.mean(enh)),4); rec["enh_p90"]=round(float(np.percentile(enh,90)),4)
                rec["enh_frac_gt1"]=round(float(np.mean(enh>1.0)),4)
        rec["pcr"]=lab.get(numkey(pid),np.nan); rows.append(rec)
        if test:
            for k,v in rec.items(): print(f"  {k:22s} {v}")
            print(f"\n  image {pim.shape} spacing {zoom} | tumor voxels {int(mask.sum())}")
        elif (i+1)%25==0: print(f"  ... {i+1}/{len(masks)}",flush=True)
    if test: print("\nTEST done - if features look sane, run without --test."); return
    df=pd.DataFrame(rows); df.to_csv(OUTDIR/"ispy1_features.csv",index=False)
    n=int(df["pcr"].notna().sum())
    print(f"\nwrote ispy1_features.csv: {len(df)} patients, {df.shape[1]-2} features, {n} labeled")
    if n: k=df.dropna(subset=['pcr']); print(f"pCR+={int((k['pcr']==1).sum())} pCR-={int((k['pcr']==0).sum())}")
if __name__=="__main__": main()
