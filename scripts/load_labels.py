#!/usr/bin/env python3
"""load_labels.py - read a cohort's clinical spreadsheet and extract pCR labels."""
import sys
import os
import argparse
from pathlib import Path
import pandas as pd

REPO = Path.home() / "ISPY2-3DGCNN"
OUTDIR = REPO / "reports" / "multicohort"

ID_HINTS = ["patient", "subject", "ispy", "mrn", "deid", "de-id", "de_id", "case", "id"]
PCR_HINTS = ["pcr", "pathologic complete", "pathologic_complete", "path_response", "response"]

POS = {"1", "1.0", "yes", "y", "true", "pcr", "pcr=1", "complete", "responder", "cr"}
NEG = {"0", "0.0", "no", "n", "false", "non-pcr", "nonpcr", "non_pcr", "pcr=0", "incomplete", "non-responder", "residual"}

def pick(cols, hints):
    low = {c: str(c).strip().lower() for c in cols}
    for c, cl in low.items():
        if cl in hints:
            return c
    for h in hints:
        for c, cl in low.items():
            if h in cl:
                return c
    return None

def to01(v):
    s = str(v).strip().lower()
    if s in POS:
        return 1
    if s in NEG:
        return 0
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("name")
    ap.add_argument("--id", default=None)
    ap.add_argument("--pcr", default=None)
    ap.add_argument("--sheet", default=0)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if args.file.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(args.file, sheet_name=args.sheet)
    else:
        df = pd.read_csv(args.file)
    df.columns = [str(c).strip() for c in df.columns]

    print(f"=== {args.name}: {len(df)} rows, {len(df.columns)} columns ===")
    print("columns:", list(df.columns))

    id_col = args.id or pick(df.columns, ID_HINTS)
    pcr_col = args.pcr or pick(df.columns, PCR_HINTS)
    print(f"\ndetected ID column : {id_col!r}")
    print(f"detected pCR column: {pcr_col!r}")
    if id_col is None or pcr_col is None:
        sys.exit("Could not auto-detect columns. Re-run with --id and --pcr (see column list above).")

    print(f"\nraw values in {pcr_col!r}:")
    print(df[pcr_col].value_counts(dropna=False).to_string())

    out = pd.DataFrame({"patient_id": df[id_col].astype(str).str.strip(),
                        "pcr": df[pcr_col].map(to01)})
    unmapped = out["pcr"].isna().sum()
    out = out.dropna(subset=["pcr"]).drop_duplicates(subset=["patient_id"])
    out["pcr"] = out["pcr"].astype(int)

    dst = OUTDIR / f"{args.name}_labels.csv"
    out.to_csv(dst, index=False)
    print(f"\nmapped labels: {len(out)} patients  |  pCR+={int(out['pcr'].sum())}  pCR-={int((out['pcr']==0).sum())}")
    if unmapped:
        print(f"WARNING: {unmapped} rows had a pCR value that didn't map to 0/1 - inspect and extend POS/NEG if needed.")
    print(f"wrote {dst}")

if __name__ == "__main__":
    main()
