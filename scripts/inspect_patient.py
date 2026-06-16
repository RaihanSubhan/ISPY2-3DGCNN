#!/usr/bin/env python3
"""inspect_patient.py - drill into one patient of a cohort to reveal its structure."""
import sys, argparse
from pathlib import Path
import pandas as pd

REPO = Path.home() / "ISPY2-3DGCNN"
INV = REPO / "reports" / "multicohort"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--patient", default=None)
    args = ap.parse_args()
    f = INV / f"{args.name}_inventory.csv"
    if not f.exists():
        sys.exit(f"missing {f} - run inventory_cohort.py for {args.name} first.")
    df = pd.read_csv(f, dtype=str).fillna("")
    df["n_slices"] = pd.to_numeric(df["n_slices"], errors="coerce").fillna(0).astype(int)

    per = df.groupby("patient_id").size()
    print(f"=== {args.name}: {df['patient_id'].nunique()} patients, {len(df)} series ===")
    print(f"series per patient: min={per.min()}  median={int(per.median())}  max={per.max()}")
    print("modalities:", df["modality"].value_counts().to_dict())

    pid = args.patient or per.idxmax()
    sub = df[df["patient_id"] == pid].sort_values(["study_date", "series_desc"])
    print(f"\n--- example patient {pid}: {len(sub)} series across {sub['study_uid'].nunique()} studies ---")
    for _, r in sub.iterrows():
        print(f"  [{r['study_date']:>8}] {r['modality']:4s} n={r['n_slices']:>4}  {r['series_desc']}")

    mr = sub[sub["modality"] == "MR"]; seg = sub[sub["modality"] == "SEG"]
    print(f"\nMR series: {len(mr)}   SEG series: {len(seg)}")
    print("unique MR descriptions :", sorted(mr['series_desc'].unique())[:25])
    print("unique SEG descriptions:", sorted(seg['series_desc'].unique())[:25])
    print(f"\nexample MR  path: {mr['path'].iloc[0] if len(mr) else 'none'}")
    print(f"example SEG path: {seg['path'].iloc[0] if len(seg) else 'none'}")

if __name__ == "__main__":
    main()
