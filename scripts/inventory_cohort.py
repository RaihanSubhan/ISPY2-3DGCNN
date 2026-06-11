#!/usr/bin/env python3
"""inventory_cohort.py - index a downloaded TCIA cohort (DICOM tree) into a CSV."""
import sys
import os
from pathlib import Path
import pydicom
import pandas as pd

REPO = Path.home() / "ISPY2-3DGCNN"
OUTDIR = REPO / "reports" / "multicohort"

def first_dicom(folder):
    for fn in sorted(os.listdir(folder)):
        fp = os.path.join(folder, fn)
        if os.path.isfile(fp) and (fn.lower().endswith(".dcm") or "." not in fn):
            try:
                return pydicom.dcmread(fp, stop_before_pixels=True, force=True), fp
            except Exception:
                continue
    return None, None

def count_slices(folder):
    return sum(
        1 for fn in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, fn)) and (fn.lower().endswith(".dcm") or "." not in fn)
    )

def main():
    if len(sys.argv) < 3:
        sys.exit("Usage: python inventory_cohort.py <COHORT_ROOT> <name>")
    root = os.path.abspath(sys.argv[1])
    name = sys.argv[2]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    series_dirs = []
    for dirpath, _, files in os.walk(root):
        if any(f.lower().endswith(".dcm") or ("." not in f and os.path.isfile(os.path.join(dirpath, f))) for f in files):
            series_dirs.append(dirpath)
    rows = []
    for i, d in enumerate(series_dirs):
        ds, fp = first_dicom(d)
        if ds is None:
            continue
        g = lambda k, default="": str(getattr(ds, k, default))
        rows.append({
            "cohort": name,
            "patient_id": g("PatientID"),
            "study_uid": g("StudyInstanceUID"),
            "study_date": g("StudyDate"),
            "series_uid": g("SeriesInstanceUID"),
            "series_desc": g("SeriesDescription"),
            "modality": g("Modality"),
            "n_slices": count_slices(d),
            "path": d,
        })
        if (i + 1) % 500 == 0:
            print(f"  ... scanned {i + 1}/{len(series_dirs)} series", flush=True)
    if not rows:
        sys.exit(f"No DICOM series found under {root}. Check the path.")
    df = pd.DataFrame(rows)
    out = OUTDIR / f"{name}_inventory.csv"
    df.to_csv(out, index=False)
    print(f"\n=== {name} inventory ===")
    print(f"patients : {df['patient_id'].nunique()}")
    print(f"studies  : {df['study_uid'].nunique()}")
    print(f"series   : {len(df)}")
    print("modalities:", df["modality"].value_counts().to_dict())
    print("top series descriptions:")
    print(df["series_desc"].value_counts().head(12).to_string())
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
