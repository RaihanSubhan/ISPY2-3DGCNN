#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/ISPY2-3DGCNN"

source /home/mdsubhan01/miniforge3/etc/profile.d/conda.sh
conda activate 3dgcnn

MAX_CASES="${1:-2688}"
MAX_OVERLAYS="${2:-40}"

mkdir -p reports/logs reports/tables reports/figures reports/table_views

echo "Phase 7F started at: $(date)"
echo "MAX_CASES=$MAX_CASES"
echo "MAX_OVERLAYS=$MAX_OVERLAYS"

echo
echo "Step 1: clean old untracked segmentation QC images only"
git clean -f reports/figures/segmentation_qc_case_*.png 2>/dev/null || true

echo
echo "Step 2: rerun Phase 7C on expanded SEG cohort"
python scripts/phase7c_fractional_seg_threshold_recovery.py \
  --max-cases "$MAX_CASES" \
  --max-overlays "$MAX_OVERLAYS" \
  > reports/logs/phase7f_phase7c.log 2>&1

echo "Phase 7C finished at: $(date)"
tail -n 20 reports/logs/phase7f_phase7c.log || true

echo
echo "Step 3: rerun Phase 7D using all recovered pass cases"
python scripts/phase7d_rebuild_features_from_recovered_tumor_masks.py \
  --max-cases 100000 \
  > reports/logs/phase7f_phase7d.log 2>&1

echo "Phase 7D finished at: $(date)"
tail -n 20 reports/logs/phase7f_phase7d.log || true

echo
echo "Step 4: rerun Phase 7E pCR feasibility"
python scripts/phase7e_recovered_tumor_pcr_feasibility.py \
  > reports/logs/phase7f_phase7e.log 2>&1

echo "Phase 7E finished at: $(date)"
tail -n 30 reports/logs/phase7f_phase7e.log || true

echo
echo "Step 5: write Phase 7F summary"

python - <<'PY'
from pathlib import Path
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

for p in [REPORTS, TABLES, FIGS, VIEWS]:
    p.mkdir(parents=True, exist_ok=True)

def read_csv(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path, dtype=str).fillna("")
    return pd.DataFrame()

c7 = read_csv(TABLES / "phase7c_fractional_tumor_mask_candidate_table.csv")
d7 = read_csv(TABLES / "phase7d_recovered_tumor_vascular_features.csv")
e7 = read_csv(TABLES / "phase7e_recovered_tumor_modeling_dataset.csv")
perf = read_csv(TABLES / "phase7e_recovered_tumor_model_performance.csv")

c7_total = len(c7)
c7_pass = int((c7.get("recovery_status", pd.Series(dtype=str)) == "fractional_tumor_candidate_pass").sum()) if len(c7) else 0
c7_no = int((c7.get("recovery_status", pd.Series(dtype=str)) == "no_threshold_candidate").sum()) if len(c7) else 0
c7_review = int(c7.get("recovery_status", pd.Series(dtype=str)).astype(str).str.contains("candidate_large|candidate_shape|candidate_no|review").sum()) if len(c7) else 0
c7_failed = int(c7.get("recovery_status", pd.Series(dtype=str)).astype(str).str.contains("failed").sum()) if len(c7) else 0

d7_total = len(d7)
d7_success = int((d7.get("feature_status", pd.Series(dtype=str)) == "success").sum()) if len(d7) else 0
d7_patients = d7["patient_folder"].nunique() if len(d7) and "patient_folder" in d7.columns else 0

e7_patients = len(e7)
if len(e7) and "label" in e7.columns:
    labels = pd.to_numeric(e7["label"], errors="coerce")
    e7_pos = int((labels == 1).sum())
    e7_neg = int((labels == 0).sum())
else:
    e7_pos = 0
    e7_neg = 0

ml_status = "unknown"
best_model = ""
best_auroc = ""
best_auprc = ""

if len(perf):
    if "reason" in perf.columns:
        ml_status = "not_run"
    elif "AUROC" in perf.columns:
        ml_status = "run"
        perf2 = perf.copy()
        perf2["AUROC_num"] = pd.to_numeric(perf2["AUROC"], errors="coerce")
        perf2["AUPRC_num"] = pd.to_numeric(perf2["AUPRC"], errors="coerce")
        perf2 = perf2.sort_values(["AUROC_num", "AUPRC_num"], ascending=[False, False])
        best_model = str(perf2.iloc[0].get("model", ""))
        best_auroc = str(perf2.iloc[0].get("AUROC", ""))
        best_auprc = str(perf2.iloc[0].get("AUPRC", ""))
else:
    ml_status = "no_performance_file"

summary_df = pd.DataFrame([
    {"stage": "Phase 7C", "metric": "SEG cases tested", "value": c7_total},
    {"stage": "Phase 7C", "metric": "fractional tumor candidate pass", "value": c7_pass},
    {"stage": "Phase 7C", "metric": "no threshold candidate", "value": c7_no},
    {"stage": "Phase 7C", "metric": "review candidates", "value": c7_review},
    {"stage": "Phase 7C", "metric": "failed", "value": c7_failed},
    {"stage": "Phase 7D", "metric": "feature rows", "value": d7_total},
    {"stage": "Phase 7D", "metric": "feature success rows", "value": d7_success},
    {"stage": "Phase 7D", "metric": "unique patients", "value": d7_patients},
    {"stage": "Phase 7E", "metric": "labeled modeling patients", "value": e7_patients},
    {"stage": "Phase 7E", "metric": "pCR positive", "value": e7_pos},
    {"stage": "Phase 7E", "metric": "non-pCR negative", "value": e7_neg},
    {"stage": "Phase 7E", "metric": "ML status", "value": ml_status},
    {"stage": "Phase 7E", "metric": "best model", "value": best_model},
    {"stage": "Phase 7E", "metric": "best AUROC", "value": best_auroc},
    {"stage": "Phase 7E", "metric": "best AUPRC", "value": best_auprc},
])

summary_df.to_csv(TABLES / "phase7f_expanded_recovered_cohort_summary.csv", index=False)

# Figure 220
plt.figure(figsize=(10, 5))
plot = summary_df[summary_df["metric"].isin([
    "SEG cases tested",
    "fractional tumor candidate pass",
    "feature success rows",
    "labeled modeling patients",
    "pCR positive",
    "non-pCR negative"
])].copy()
plot["value_num"] = pd.to_numeric(plot["value"], errors="coerce").fillna(0)
plt.barh(plot["metric"], plot["value_num"])
plt.title("Phase 7F Expanded Recovered Tumor Cohort Summary")
plt.xlabel("Count")
plt.tight_layout()
plt.savefig(FIGS / "220_phase7f_expansion_summary.png", dpi=220)
plt.close()

# Figure 221
plt.figure(figsize=(7, 5))
plt.bar(["pCR", "non-pCR"], [e7_pos, e7_neg])
plt.title("Phase 7F Label Counts in Recovered Tumor Cohort")
plt.ylabel("Patients")
plt.tight_layout()
plt.savefig(FIGS / "221_phase7f_label_counts.png", dpi=220)
plt.close()

# Figure 222
plt.figure(figsize=(8, 5))
if len(c7) and "selected_frame_fraction" in c7.columns:
    vals = pd.to_numeric(c7["selected_frame_fraction"], errors="coerce").dropna()
    if len(vals):
        vals.plot(kind="hist", bins=30)
        plt.title("Phase 7F Recovered Mask Fraction")
        plt.xlabel("Mask pixels / image pixels")
        plt.ylabel("Cases")
    else:
        plt.text(0.5, 0.5, "No mask fraction values", ha="center", va="center")
        plt.axis("off")
else:
    plt.text(0.5, 0.5, "No Phase 7C fraction table found", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "222_phase7f_mask_fraction_distribution.png", dpi=220)
plt.close()

# Markdown table view
def make_md(csv_path, md_path, title, max_rows=80, max_cols=8):
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    small = df.head(max_rows).iloc[:, :max_cols]

    def clean(x):
        x = str(x).replace("|", "\\|").replace("\n", " ")
        if len(x) > 90:
            x = x[:87] + "..."
        return x

    lines = ["# " + title, "", "Rows: " + str(len(df)), "Columns: " + str(len(df.columns)), ""]
    lines.append("| " + " | ".join(clean(c) for c in small.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(small.columns)) + " |")
    for _, r in small.iterrows():
        lines.append("| " + " | ".join(clean(v) for v in r.tolist()) + " |")
    md_path.write_text("\n".join(lines))

make_md(
    TABLES / "phase7f_expanded_recovered_cohort_summary.csv",
    VIEWS / "phase7f_expanded_recovered_cohort_summary.md",
    "Phase 7F Expanded Recovered Cohort Summary"
)

report = []
report.append("# Phase 7F Expanded Recovered Tumor Cohort Report")
report.append("")
report.append("This phase reruns Phase 7C, Phase 7D, and Phase 7E on a larger SEG cohort.")
report.append("")
report.append("## Main outputs")
report.append("")
report.append("- reports/tables/phase7f_expanded_recovered_cohort_summary.csv")
report.append("- reports/table_views/phase7f_expanded_recovered_cohort_summary.md")
report.append("- reports/figures/220_phase7f_expansion_summary.png")
report.append("- reports/figures/221_phase7f_label_counts.png")
report.append("- reports/figures/222_phase7f_mask_fraction_distribution.png")
report.append("")
report.append("## Main counts")
report.append("")
for _, r in summary_df.iterrows():
    report.append(f"- {r['stage']} | {r['metric']}: {r['value']}")
report.append("")
report.append("## Interpretation")
report.append("")
if ml_status == "run":
    report.append("The expanded recovered tumor cohort is now large enough for a feasibility ML model.")
    report.append(f"Best model: {best_model}. AUROC: {best_auroc}. AUPRC: {best_auprc}.")
elif e7_patients >= 6 and e7_pos >= 2 and e7_neg >= 2:
    report.append("The expanded cohort has enough labeled patients, but model output should be checked.")
else:
    report.append("The expanded recovered tumor cohort is still too small for reliable supervised ML.")
    report.append("If this remains true after all SEG cases are tested, the project should switch to official support-data ROI, FTV, PE, or SER maps instead of relying only on DICOM SEG fractional recovery.")
report.append("")
report.append("## Next step")
report.append("")
if ml_status == "run":
    report.append("Run Phase 7G to create final recovered-mask model interpretation and compare it with Phase 5A.")
else:
    report.append("If counts are still low, run Phase 8A to search support-data ROI/FTV/PE/SER tumor sources.")

(REPORTS / "Phase7F_Expanded_Recovered_Tumor_Cohort_Report.md").write_text("\n".join(report))

# Dashboard update
dash = REPORTS / "Dashboard.md"
old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
dash_add = "\n\n## Phase 7F expanded recovered tumor cohort\n\n- [Expanded recovered cohort report](Phase7F_Expanded_Recovered_Tumor_Cohort_Report.md)\n- [Expanded recovered cohort summary](table_views/phase7f_expanded_recovered_cohort_summary.md)\n"
if "Phase 7F expanded recovered tumor cohort" not in old_dash:
    dash.write_text(old_dash.rstrip() + dash_add)

# README update
readme = REPO / "README.md"
old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
readme_add = "\n\n### Phase 7F: expanded recovered tumor cohort\n\nMain outputs:\n\n- reports/Phase7F_Expanded_Recovered_Tumor_Cohort_Report.md\n- reports/tables/phase7f_expanded_recovered_cohort_summary.csv\n- reports/figures/220_phase7f_expansion_summary.png\n"
if "Phase 7F: expanded recovered tumor cohort" not in old_readme:
    readme.write_text(old_readme.rstrip() + readme_add)

print("Phase 7F report written.")
print("SEG cases tested:", c7_total)
print("Fractional pass cases:", c7_pass)
print("Feature success rows:", d7_success)
print("Labeled modeling patients:", e7_patients)
print("pCR:", e7_pos)
print("non-pCR:", e7_neg)
print("ML status:", ml_status)
PY

echo
echo "Phase 7F completed at: $(date)"
