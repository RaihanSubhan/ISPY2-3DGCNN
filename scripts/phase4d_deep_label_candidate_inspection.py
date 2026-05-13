from pathlib import Path
import re
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path.home() / "ISPY2-3DGCNN"
REPORTS = REPO / "reports"
TABLES = REPORTS / "tables"
FIGS = REPORTS / "figures"
VIEWS = REPORTS / "table_views"

TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
VIEWS.mkdir(parents=True, exist_ok=True)

candidate_path = TABLES / "phase4c_label_candidate_review.csv"
backup_path = TABLES / "phase4b_label_merge_candidates.csv"

if candidate_path.exists():
    cand = pd.read_csv(candidate_path, dtype=str).fillna("")
elif backup_path.exists():
    cand = pd.read_csv(backup_path, dtype=str).fillna("")
else:
    raise SystemExit("No label candidate table found. Run Phase 4C first.")

def read_table(path):
    path = Path(path)
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, dtype=str).fillna("")
        if path.suffix.lower() == ".tsv":
            return pd.read_csv(path, sep="\t", dtype=str).fillna("")
        if path.suffix.lower() == ".txt":
            return pd.read_csv(path, sep=None, engine="python", dtype=str).fillna("")
        if path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(path, dtype=str).fillna("")
    except Exception as e:
        return None
    return None

def clean_text(x, n=120):
    x = str(x).replace("\n", " ").replace("\r", " ").replace("|", "\\|")
    if len(x) > n:
        x = x[:n-3] + "..."
    return x

def label_like_columns(cols):
    keys = [
        "pcr", "pathologic", "complete", "response", "responder",
        "rcb", "outcome", "residual", "burden", "surgery",
        "hr", "her2", "er", "pr", "tnbc", "subtype", "arm", "treatment"
    ]
    bad = ["series", "study", "uid", "path", "folder", "file", "description"]
    out = []
    for c in cols:
        low = str(c).lower()
        if any(k in low for k in keys) and not any(b in low for b in bad):
            out.append(c)
    return out

for col in ["usable_labeled_patients", "patient_overlap", "n_positive", "n_negative"]:
    if col in cand.columns:
        cand[col] = pd.to_numeric(cand[col], errors="coerce").fillna(0)

sort_cols = [c for c in ["usable_labeled_patients", "patient_overlap", "n_positive", "n_negative"] if c in cand.columns]
if sort_cols:
    cand = cand.sort_values(sort_cols, ascending=False)

inspection_rows = []
preview_rows = []

for idx, row in cand.head(30).iterrows():
    fpath = row.get("file_path", "")
    p = Path(fpath)

    item = {
        "candidate_rank": len(inspection_rows) + 1,
        "file_path": fpath,
        "file_exists": p.exists(),
        "patient_column_from_phase4": row.get("patient_column", ""),
        "label_column_from_phase4": row.get("label_column", ""),
        "patient_overlap_from_phase4": row.get("patient_overlap", ""),
        "usable_labeled_patients_from_phase4": row.get("usable_labeled_patients", ""),
        "n_positive_from_phase4": row.get("n_positive", ""),
        "n_negative_from_phase4": row.get("n_negative", ""),
        "status_from_phase4": row.get("status", ""),
    }

    if not p.exists():
        item["read_status"] = "file_missing"
        inspection_rows.append(item)
        continue

    df = read_table(p)
    if df is None or df.empty:
        item["read_status"] = "read_failed_or_empty"
        inspection_rows.append(item)
        continue

    cols = list(df.columns)
    lab_cols = label_like_columns(cols)

    item["read_status"] = "read_ok"
    item["n_rows"] = len(df)
    item["n_columns"] = len(cols)
    item["label_like_columns_found"] = "; ".join(lab_cols[:30])
    item["all_columns_preview"] = "; ".join([str(c) for c in cols[:40]])

    value_summaries = []
    for c in lab_cols[:8]:
        vc = df[c].astype(str).replace("", "blank").value_counts().head(8)
        vc_text = ", ".join([f"{clean_text(k, 30)}:{int(v)}" for k, v in vc.items()])
        value_summaries.append(f"{c} => {vc_text}")
    item["label_value_preview"] = " || ".join(value_summaries)

    inspection_rows.append(item)

    patient_col = row.get("patient_column", "")
    if patient_col not in df.columns:
        patient_col = ""
        for c in cols:
            low = str(c).lower()
            if "patient" in low or "subject" in low or low in ["id", "patientid"]:
                patient_col = c
                break

    preview_cols = []
    if patient_col:
        preview_cols.append(patient_col)

    for c in lab_cols[:8]:
        if c not in preview_cols:
            preview_cols.append(c)

    for c in cols:
        low = str(c).lower()
        if any(k in low for k in ["subtype", "arm", "treatment", "her2", "er", "pr", "tnbc"]) and c not in preview_cols:
            preview_cols.append(c)

    preview_cols = preview_cols[:12]

    if preview_cols:
        small = df[preview_cols].head(12).copy()
        for ridx, prow in small.iterrows():
            out = {
                "candidate_rank": item["candidate_rank"],
                "source_file": fpath,
                "preview_row": ridx,
            }
            for c in preview_cols:
                out[c] = clean_text(prow[c], 90)
            preview_rows.append(out)

inspection = pd.DataFrame(inspection_rows)
preview = pd.DataFrame(preview_rows)

inspection.to_csv(TABLES / "phase4d_label_candidate_deep_inspection.csv", index=False)
preview.to_csv(TABLES / "phase4d_label_candidate_preview.csv", index=False)

# Markdown preview
lines = []
lines.append("# Phase 4D Label Candidate Preview")
lines.append("")
lines.append("This file previews candidate clinical or response-label tables.")
lines.append("")
if len(preview) > 0:
    small = preview.head(80).iloc[:, :12].fillna("")
    lines.append("| " + " | ".join(clean_text(c, 50) for c in small.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(small.columns)) + " |")
    for _, r in small.iterrows():
        lines.append("| " + " | ".join(clean_text(v, 90) for v in r.tolist()) + " |")
else:
    lines.append("No preview rows were created.")

(VIEWS / "phase4d_label_candidate_preview.md").write_text("\n".join(lines))

# Figure
plt.figure(figsize=(10, 6))
if len(inspection) > 0 and "patient_overlap_from_phase4" in inspection.columns:
    plot_df = inspection.head(15).copy()
    y = plot_df["candidate_rank"].astype(str) + ": " + plot_df["label_column_from_phase4"].astype(str).str[:35]
    x = pd.to_numeric(plot_df["patient_overlap_from_phase4"], errors="coerce").fillna(0)
    plt.barh(y, x)
    plt.xlabel("Patient overlap from Phase 4")
    plt.title("Phase 4D Candidate Label Source Overlap")
else:
    plt.text(0.5, 0.5, "No candidate rows found", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "90_phase4d_candidate_overlap.png", dpi=220)
plt.close()

n_candidates = len(inspection)
n_read_ok = int((inspection["read_status"] == "read_ok").sum()) if len(inspection) else 0
n_with_label_like = int(inspection["label_like_columns_found"].astype(str).str.len().gt(0).sum()) if len(inspection) else 0

summary = [
    "# Phase 4D Label Source Deep Inspection Summary",
    "",
    "This phase inspects candidate pCR or response-label files more deeply.",
    "",
    "## Main outputs",
    "",
    "- reports/tables/phase4d_label_candidate_deep_inspection.csv",
    "- reports/tables/phase4d_label_candidate_preview.csv",
    "- reports/table_views/phase4d_label_candidate_preview.md",
    "- reports/figures/90_phase4d_candidate_overlap.png",
    "",
    "## Main counts",
    "",
    "- Candidate rows inspected: " + str(n_candidates),
    "- Candidate files read successfully: " + str(n_read_ok),
    "- Candidates with label-like columns: " + str(n_with_label_like),
    "",
    "## What to do next",
    "",
    "Open `reports/tables/phase4d_label_candidate_deep_inspection.csv` and `reports/table_views/phase4d_label_candidate_preview.md` in GitHub.",
    "",
    "Look for a real outcome column such as pCR, pathologic complete response, RCB, responder, or response.",
    "",
    "If a true pCR column is visible, we can map it into `reports/tables/phase4c_pcr_label_template.csv` and rerun Phase 4B.",
    "",
    "If no true pCR column is visible, then the official clinical support data still needs to be downloaded or obtained.",
]

(REPORTS / "Phase4D_Label_Source_Deep_Inspection_Summary.md").write_text("\n".join(summary))

# Dashboard
dash = REPORTS / "Dashboard.md"
old = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
add = "\n\n## Phase 4D table view\n\n- [Label candidate preview](table_views/phase4d_label_candidate_preview.md)\n"
if "Phase 4D table view" not in old:
    dash.write_text(old.rstrip() + add)

print("Phase 4D complete.")
print("Candidate rows inspected:", n_candidates)
print("Files read successfully:", n_read_ok)
print("Candidates with label-like columns:", n_with_label_like)
