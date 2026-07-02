#!/usr/bin/env python3
"""build_ispy2_features.py - map existing I-SPY2 FTV support-data into the shared schema."""
import sys
from pathlib import Path
import pandas as pd
REPO = Path.home()/"ISPY2-3DGCNN"; OUTDIR = REPO/"reports"/"multicohort"
SRC = REPO/"reports"/"tables"/"phase8c_ftv_modeling_dataset.csv"

def main():
    if not SRC.exists(): sys.exit(f"missing {SRC}")
    df = pd.read_csv(SRC)
    out = pd.DataFrame({
        "patient_id": df["patient_folder"].astype(str),
        "tumor_volume_mm3": pd.to_numeric(df["VOLUME_TUM_BLU_V10"], errors="coerce"),
        "sphericity": pd.to_numeric(df["SPHERICITY_T0"], errors="coerce"),
        "pcr": pd.to_numeric(df["label"], errors="coerce"),
    }).dropna(subset=["pcr"])
    out["pcr"] = out["pcr"].astype(int)
    dst = OUTDIR/"ispy2_features.csv"; OUTDIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    print(f"wrote {dst}: {len(out)} patients | pCR+={int(out['pcr'].sum())} pCR-={int((out['pcr']==0).sum())}")
    print("columns:", [c for c in out.columns if c!='patient_id'])
if __name__ == "__main__": main()
