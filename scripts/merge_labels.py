#!/usr/bin/env python3
"""merge_labels.py - join a cohort's features with its pCR labels (robust ID match)."""
import sys, re
from pathlib import Path
import pandas as pd
REPO = Path.home()/"ISPY2-3DGCNN"; OUTDIR = REPO/"reports"/"multicohort"

def norm_id(s):
    runs = re.findall(r"\d+", str(s))
    if not runs: return str(s).strip().lower()
    key = max(runs, key=len)
    try: return str(int(key))
    except Exception: return key

def main():
    if len(sys.argv) < 2: sys.exit("Usage: python merge_labels.py <cohort>")
    name = sys.argv[1]
    feat = OUTDIR/f"{name}_features.csv"; lab = OUTDIR/f"{name}_labels.csv"
    if not feat.exists(): sys.exit(f"missing {feat}")
    if not lab.exists(): sys.exit(f"missing {lab} - run load_labels.py first")
    F = pd.read_csv(feat); L = pd.read_csv(lab)
    if "pcr" in F.columns: F = F.drop(columns=["pcr"])
    F["_k"] = F["patient_id"].map(norm_id); L["_k"] = L["patient_id"].map(norm_id)
    L = L.dropna(subset=["pcr"]).drop_duplicates("_k")
    merged = F.merge(L[["_k","pcr"]], on="_k", how="left").drop(columns=["_k"])
    n = int(merged["pcr"].notna().sum())
    print(f"{name}: {len(F)} feature rows, {len(L)} label rows -> matched pcr for {n}")
    print(f"  feature ID example: {F['patient_id'].iloc[0]!r} -> key {norm_id(F['patient_id'].iloc[0])!r}")
    print(f"  label   ID example: {L['patient_id'].iloc[0]!r} -> key {norm_id(L['patient_id'].iloc[0])!r}")
    if n == 0: sys.exit("NO MATCHES - ID formats don't align. Paste a few IDs from each file.")
    merged.to_csv(feat, index=False)
    k = merged.dropna(subset=["pcr"])
    print(f"  pCR+={int((k['pcr']==1).sum())}  pCR-={int((k['pcr']==0).sum())}")
    print(f"wrote {feat} (now has 'pcr'; {n} labeled patients)")

if __name__ == "__main__": main()
