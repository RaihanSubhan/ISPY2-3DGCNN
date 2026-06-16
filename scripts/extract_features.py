#!/usr/bin/env python3
"""extract_features.py - harmonized tumor features per patient from a cohort's segmentations."""
import sys, os, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom

REPO = Path.home() / "ISPY2-3DGCNN"
OUTDIR = REPO / "reports" / "multicohort"

# tumor SEG series description per cohort (from inspect_patient.py)
TUMOR_SEG = {
    "ispy1": "PE Segmentation thresh=70",   # functional tumor volume segmentation
    "duke":  "Segmentation",                 # Duke expert tumor segmentation
}

def read_seg_mask(folder):
    """Return (3D bool mask, voxel_mm3 or nan) from a SEG series folder."""
    seg_ds = None
    for fn in sorted(os.listdir(folder)):
        fp = os.path.join(folder, fn)
        if not os.path.isfile(fp):
            continue
        try:
            ds = pydicom.dcmread(fp, force=True)
        except Exception:
            continue
        if str(getattr(ds, "Modality", "")) == "SEG":
            seg_ds = ds
            break
    if seg_ds is None:
        return None, np.nan
    arr = np.asarray(seg_ds.pixel_array)
    mask = arr > 0
    if mask.ndim == 2:
        mask = mask[None, ...]
    vox = np.nan
    try:
        pm = seg_ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
        ps = [float(x) for x in pm.PixelSpacing]
        th = float(getattr(pm, "SliceThickness", getattr(pm, "SpacingBetweenSlices", 1.0)))
        vox = ps[0] * ps[1] * th
    except Exception:
        pass
    return mask, vox

def shape_features(mask, vox):
    feats = {}
    v = int(mask.sum())
    feats["tumor_volume_voxels"] = v
    feats["tumor_volume_mm3"] = round(v * vox, 2) if not np.isnan(vox) else np.nan
    feats["n_tumor_slices"] = int((mask.reshape(mask.shape[0], -1).sum(1) > 0).sum())
    if v == 0:
        for k in ["bbox_max", "bbox_mid", "bbox_min", "elongation", "surface_area", "sphericity"]:
            feats[k] = np.nan
        return feats
    coords = np.argwhere(mask)
    ext = sorted((coords.max(0) - coords.min(0) + 1).tolist(), reverse=True)
    while len(ext) < 3:
        ext.append(0)
    feats["bbox_max"], feats["bbox_mid"], feats["bbox_min"] = ext[0], ext[1], ext[2]
    feats["elongation"] = round(ext[2] / ext[0], 4) if ext[0] else np.nan
    try:
        from skimage.measure import marching_cubes, mesh_surface_area
        verts, faces, _, _ = marching_cubes(mask.astype(float), level=0.5)
        sa = float(mesh_surface_area(verts, faces))
        feats["surface_area"] = round(sa, 2)
        feats["sphericity"] = round((np.pi ** (1/3) * (6 * v) ** (2/3)) / sa, 4) if sa > 0 else np.nan
    except Exception:
        feats["surface_area"] = np.nan
        feats["sphericity"] = np.nan
    return feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.name not in TUMOR_SEG:
        sys.exit(f"{args.name}: tumor-SEG rule only defined for {list(TUMOR_SEG)} "
                 f"(I-SPY2 reuses your existing features - merged separately).")
    seg_desc = TUMOR_SEG[args.name]
    inv = OUTDIR / f"{args.name}_inventory.csv"
    if not inv.exists():
        sys.exit(f"missing {inv}")
    df = pd.read_csv(inv, dtype=str).fillna("")
    seg = df[df["series_desc"] == seg_desc].copy()
    if seg.empty:
        sys.exit(f"No SEG series named '{seg_desc}' in {args.name}.")
    patients = sorted(seg["patient_id"].unique())
    if args.test:
        patients = patients[:1]
        print(f"TEST MODE: patient {patients[0]} | tumor SEG = '{seg_desc}'\n")

    rows = []
    for i, pid in enumerate(patients):
        psegs = seg[seg["patient_id"] == pid].sort_values("study_date")
        mask, vox = read_seg_mask(psegs.iloc[0]["path"])
        if mask is None:
            if args.test:
                print(f"  !! could not read a SEG object under {psegs.iloc[0]['path']}")
            continue
        f = {"patient_id": pid}
        f.update(shape_features(mask, vox))
        if len(psegs) > 1:  # I-SPY1 multi-timepoint -> FTV change
            m2, _ = read_seg_mask(psegs.iloc[-1]["path"])
            if m2 is not None and mask.sum() > 0:
                f["ftv_ratio_final_baseline"] = round(int(m2.sum()) / int(mask.sum()), 4)
        rows.append(f)
        if args.test:
            print("  features:")
            for k, val in f.items():
                print(f"    {k:28s} {val}")
            print(f"\n  decoded mask shape: {mask.shape} | voxel_mm3: {vox}")
        elif (i + 1) % 25 == 0:
            print(f"  ... {i+1}/{len(patients)} patients", flush=True)

    if args.test:
        print("\nTEST OK - if FTV/shape numbers look sane, re-run WITHOUT --test for the full cohort.")
        return
    out = pd.DataFrame(rows)
    dst = OUTDIR / f"{args.name}_features.csv"
    out.to_csv(dst, index=False)
    print(f"\nwrote {dst}: {len(out)} patients, {out.shape[1]-1} features")
    print("columns:", [c for c in out.columns if c != "patient_id"])

if __name__ == "__main__":
    main()
