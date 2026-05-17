from pathlib import Path
import numpy as np
import pandas as pd

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

BASELINE = TABLES / "phase5a_model_calibration_metrics.csv"
FTV_PERF = TABLES / "phase8c_ftv_model_performance.csv"
DELTA = TABLES / "phase8d_metric_delta_summary.csv"
CLAIMS = TABLES / "phase8d_safe_claims_and_limitations.csv"
FTV_SOURCE = TABLES / "phase8c_ftv_source_selection.csv"

def load_csv(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

baseline = load_csv(BASELINE)
ftv = load_csv(FTV_PERF)
delta = load_csv(DELTA)
claims_in = load_csv(CLAIMS)
source = load_csv(FTV_SOURCE)

def best_model(df):
    if df.empty:
        return {}
    d = df.copy()
    for c in ["AUROC", "AUPRC", "Brier_score"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values(["AUROC", "AUPRC", "Brier_score"], ascending=[False, False, True])
    return d.iloc[0].to_dict()

baseline_best = best_model(baseline)
ftv_best = best_model(ftv)

baseline_auroc = float(baseline_best.get("AUROC", np.nan))
baseline_auprc = float(baseline_best.get("AUPRC", np.nan))
baseline_brier = float(baseline_best.get("Brier_score", np.nan))

ftv_auroc = float(ftv_best.get("AUROC", np.nan))
ftv_auprc = float(ftv_best.get("AUPRC", np.nan))
ftv_brier = float(ftv_best.get("Brier_score", np.nan))

auroc_delta = ftv_auroc - baseline_auroc
auprc_delta = ftv_auprc - baseline_auprc
brier_delta = ftv_brier - baseline_brier

# -------------------------
# Final project status table
# -------------------------
status_rows = [
    {
        "phase": "Phase 1",
        "name": "DICOM inventory and visit tracking",
        "status": "complete",
        "main_result": "Built cohort-level ISPY2 inventory and visit tables.",
        "paper_use": "Methods: data inventory and cohort construction."
    },
    {
        "phase": "Phase 2",
        "name": "SEG-to-MR linking",
        "status": "complete but diagnostic",
        "main_result": "Linked SEG files to MR series, but high-confidence tumor masks were not confirmed.",
        "paper_use": "Methods and limitation."
    },
    {
        "phase": "Phase 7A-7F",
        "name": "DICOM SEG diagnosis and recovered-mask path",
        "status": "diagnostic, not main result",
        "main_result": "DICOM SEG files were mostly full-slice masks. Fractional thresholding recovered limited tumor candidates.",
        "paper_use": "Important negative result and rationale for support-data path."
    },
    {
        "phase": "Phase 8A-8B",
        "name": "Support-data tumor source search",
        "status": "complete",
        "main_result": "Identified official MRI/FTV support-data source as stronger tumor-feature path.",
        "paper_use": "Methods: tumor feature source selection."
    },
    {
        "phase": "Phase 8C",
        "name": "FTV support-data modeling",
        "status": "complete pilot",
        "main_result": "SVM_RBF achieved AUROC 0.6667 and AUPRC 0.7528 on 13 labeled patients.",
        "paper_use": "Pilot results."
    },
    {
        "phase": "Phase 8D",
        "name": "FTV vs baseline comparison",
        "status": "complete",
        "main_result": "FTV path improved AUROC by 0.1205 over Phase 5A baseline.",
        "paper_use": "Main pilot comparison result."
    },
    {
        "phase": "Future work",
        "name": "Temporal graph / 3DGCNN direction",
        "status": "planned",
        "main_result": "Use stable FTV/PE/SER patient-visit features first, then temporal graph learning.",
        "paper_use": "Future work and novelty."
    }
]

status_df = pd.DataFrame(status_rows)
status_df.to_csv(TABLES / "phase8e_final_project_status.csv", index=False)

# -------------------------
# Manuscript outline table
# -------------------------
outline_rows = [
    {
        "section": "Title",
        "content": "Pilot FTV-Based pCR Prediction and Tumor-Source Validation in I-SPY2 Breast MRI",
        "purpose": "Make the project focus clear."
    },
    {
        "section": "Background",
        "content": "Breast MRI can monitor treatment response during neoadjuvant therapy, but reliable tumor-source selection is needed before model building.",
        "purpose": "Explain why this work matters."
    },
    {
        "section": "Problem",
        "content": "Initial DICOM SEG masks were not reliable tumor-only masks because they often represented full-slice analysis masks.",
        "purpose": "Explain the key technical challenge."
    },
    {
        "section": "Hypothesis",
        "content": "Official MRI/FTV support-data features provide a cleaner pilot signal for pCR prediction than weak DICOM SEG-derived proxy features.",
        "purpose": "State the testable idea."
    },
    {
        "section": "Methods",
        "content": "Download ISPY2 on Cradle, build DICOM inventory, diagnose SEG masks, search support data, build FTV feature table, merge pCR labels, and train baseline ML.",
        "purpose": "Summarize the pipeline."
    },
    {
        "section": "Models",
        "content": "Logistic regression, SVM, random forest, gradient boosting, and other classical ML models were compared.",
        "purpose": "Explain model choice."
    },
    {
        "section": "Main result",
        "content": "FTV SVM_RBF model achieved AUROC 0.6667, AUPRC 0.7528, and Brier 0.2405 in a small pilot cohort.",
        "purpose": "Show the key result."
    },
    {
        "section": "Limitations",
        "content": "Only 13 labeled FTV patients were used. This is not clinical validation.",
        "purpose": "Keep claims honest."
    },
    {
        "section": "Future work",
        "content": "Build temporal graph learning using longitudinal FTV, PE/SER, and tumor-change features once more stable patient-visit features are available.",
        "purpose": "Connect to 3DGCNN / temporal graph direction."
    }
]

outline_df = pd.DataFrame(outline_rows)
outline_df.to_csv(TABLES / "phase8e_manuscript_outline.csv", index=False)

# -------------------------
# Future temporal graph plan
# -------------------------
future_rows = [
    {
        "step": 1,
        "task": "Use official support-data features as tumor source",
        "reason": "DICOM SEG tumor masks were not reliable enough.",
        "output": "Clean patient-visit FTV/PE/SER feature table."
    },
    {
        "step": 2,
        "task": "Create patient-level longitudinal visit graph",
        "reason": "Treatment response is temporal, not static.",
        "output": "Nodes = visits, edges = time order."
    },
    {
        "step": 3,
        "task": "Add tumor-region feature graph when PE/SER or ROI maps are available",
        "reason": "Spatial tumor heterogeneity should be represented by regions or supervoxels.",
        "output": "Nodes = tumor habitats or supervoxels."
    },
    {
        "step": 4,
        "task": "Train temporal graph model",
        "reason": "Graph model can learn change patterns across visits.",
        "output": "pCR probability and response trajectory."
    },
    {
        "step": 5,
        "task": "Compare with classical ML",
        "reason": "Must prove temporal graph model is better than FTV baseline.",
        "output": "AUROC, AUPRC, Brier, calibration, and feature/attention explanation."
    }
]

future_df = pd.DataFrame(future_rows)
future_df.to_csv(TABLES / "phase8e_future_temporal_graph_plan.csv", index=False)

# -------------------------
# Safe claims
# -------------------------
if not claims_in.empty:
    safe_claims = claims_in.copy()
else:
    safe_claims = pd.DataFrame()

extra_claims = pd.DataFrame([
    {
        "claim_type": "safe_claim",
        "claim": "The project developed a reproducible Cradle-to-GitHub pipeline for ISPY2 data handling, tumor-source diagnosis, pCR label mapping, and pilot modeling.",
        "reason": "All phases and outputs are tracked in the repository."
    },
    {
        "claim_type": "safe_claim",
        "claim": "The official FTV/MRI support-data route produced a stronger pilot pCR signal than the earlier internal proxy route.",
        "reason": "Phase 8C AUROC exceeded Phase 5A AUROC."
    },
    {
        "claim_type": "avoid_claim",
        "claim": "The model is ready for clinical use.",
        "reason": "Only 13 labeled patients were used in Phase 8C."
    },
    {
        "claim_type": "avoid_claim",
        "claim": "The DICOM SEG masks are confirmed tumor masks.",
        "reason": "Phase 7B showed the DICOM SEG masks were mostly full-slice masks."
    }
])

safe_claims = pd.concat([safe_claims, extra_claims], ignore_index=True)
safe_claims.to_csv(TABLES / "phase8e_safe_claims_for_paper.csv", index=False)

# -------------------------
# Figures
# -------------------------
# Figure 270 pipeline
fig, ax = plt.subplots(figsize=(12, 7))
ax.axis("off")
pipeline_text = (
    "Final project logic\n\n"
    "ISPY2 raw DICOM on Cradle\n"
    "↓\n"
    "Phase 1-2: Inventory + SEG/MR linking\n"
    "↓\n"
    "Phase 7: SEG diagnosis showed DICOM SEG is weak for tumor-only masks\n"
    "↓\n"
    "Phase 8: Support-data source selection\n"
    "↓\n"
    "Official FTV / multi-feature MRI table\n"
    "↓\n"
    "pCR labels + classical ML\n"
    "↓\n"
    f"Best pilot result: SVM_RBF AUROC={ftv_auroc:.3f}, AUPRC={ftv_auprc:.3f}\n"
    "↓\n"
    "Future: temporal graph model with longitudinal FTV/PE/SER features"
)
ax.text(0.5, 0.5, pipeline_text, ha="center", va="center", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "270_phase8e_project_pipeline.png", dpi=220)
plt.close()

# Figure 271 metric comparison
fig, ax = plt.subplots(figsize=(9, 5))
metrics = ["AUROC", "AUPRC", "Brier score"]
baseline_vals = [baseline_auroc, baseline_auprc, baseline_brier]
ftv_vals = [ftv_auroc, ftv_auprc, ftv_brier]
x = np.arange(len(metrics))
width = 0.35
ax.bar(x - width/2, baseline_vals, width, label="Phase 5A baseline")
ax.bar(x + width/2, ftv_vals, width, label="Phase 8C FTV")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("Score")
ax.set_title("Key pilot metric comparison")
ax.legend()
plt.tight_layout()
plt.savefig(FIGS / "271_phase8e_key_metric_comparison.png", dpi=220)
plt.close()

# Figure 272 future temporal graph
fig, ax = plt.subplots(figsize=(12, 7))
ax.axis("off")
graph_text = (
    "Future temporal graph plan\n\n"
    "Patient\n"
    " ├── Visit T0: FTV, PE/SER, shape, subtype\n"
    " ├── Visit T1: FTV change, PE/SER change\n"
    " ├── Visit T2: response trajectory\n"
    " └── Outcome: pCR / non-pCR\n\n"
    "Graph design\n"
    "  Nodes = visits or tumor habitats\n"
    "  Edges = temporal order and feature similarity\n"
    "  Model = temporal GNN / 3D graph model\n\n"
    "Goal\n"
    "  Predict pCR and explain which tumor-change pattern matters"
)
ax.text(0.5, 0.5, graph_text, ha="center", va="center", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "272_phase8e_future_temporal_graph_plan.png", dpi=220)
plt.close()

# -------------------------
# Markdown helper
# -------------------------
def make_md(csv_path, md_path, title, max_rows=80, max_cols=8):
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    small = df.head(max_rows).iloc[:, :max_cols]

    def clean(x):
        x = str(x).replace("|", "\\|").replace("\n", " ")
        if len(x) > 90:
            x = x[:87] + "..."
        return x

    lines = []
    lines.append("# " + title)
    lines.append("")
    lines.append("Rows: " + str(len(df)))
    lines.append("Columns: " + str(len(df.columns)))
    lines.append("")
    lines.append("| " + " | ".join(clean(c) for c in small.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(small.columns)) + " |")

    for _, row in small.iterrows():
        lines.append("| " + " | ".join(clean(v) for v in row.tolist()) + " |")

    md_path.write_text("\n".join(lines))

make_md(TABLES / "phase8e_final_project_status.csv", VIEWS / "phase8e_final_project_status.md", "Phase 8E Final Project Status")
make_md(TABLES / "phase8e_manuscript_outline.csv", VIEWS / "phase8e_manuscript_outline.md", "Phase 8E Manuscript Outline")
make_md(TABLES / "phase8e_future_temporal_graph_plan.csv", VIEWS / "phase8e_future_temporal_graph_plan.md", "Phase 8E Future Temporal Graph Plan")
make_md(TABLES / "phase8e_safe_claims_for_paper.csv", VIEWS / "phase8e_safe_claims_for_paper.md", "Phase 8E Safe Claims for Paper")

# -------------------------
# Final report
# -------------------------
report = []
report.append("# Phase 8E Final Research Proposal and Manuscript Plan")
report.append("")
report.append("## Final research direction")
report.append("")
report.append("The strongest current direction is not to continue with weak DICOM SEG masks. The stronger path is to use official MRI / FTV support-data features for pCR prediction, then extend the work toward temporal graph learning once stable longitudinal tumor features are available.")
report.append("")
report.append("## Working title")
report.append("")
report.append("Pilot FTV-Based Prediction of Pathologic Complete Response in I-SPY2 Breast MRI with a Future Temporal Graph Learning Framework")
report.append("")
report.append("## What has been done")
report.append("")
report.append("A full Cradle-to-GitHub pipeline was built. The project downloaded ISPY2 data, created a DICOM inventory, linked SEG and MR data, diagnosed segmentation problems, recovered limited fractional tumor masks, found official MRI/FTV support data, mapped pCR labels, and trained pilot ML models.")
report.append("")
report.append("## Key technical lesson")
report.append("")
report.append("The DICOM SEG files were not reliable tumor-only masks for the main modeling pipeline. Phase 7 showed that many masks behaved like full-slice analysis masks. This changed the project direction toward official support-data FTV / MRI features.")
report.append("")
report.append("## Main result")
report.append("")
report.append(f"The Phase 5A baseline best model was {baseline_best.get('model', '')}, with AUROC {baseline_auroc:.4f}, AUPRC {baseline_auprc:.4f}, and Brier score {baseline_brier:.4f}.")
report.append("")
report.append(f"The Phase 8C FTV best model was {ftv_best.get('model', '')}, with AUROC {ftv_auroc:.4f}, AUPRC {ftv_auprc:.4f}, and Brier score {ftv_brier:.4f}.")
report.append("")
report.append(f"The FTV path improved AUROC by {auroc_delta:.4f}. It improved AUPRC by {auprc_delta:.4f}. It reduced Brier score by {abs(brier_delta):.4f}.")
report.append("")
report.append("## Safe conclusion")
report.append("")
report.append("Official MRI / FTV support-data features produced a stronger pilot pCR prediction signal than the earlier internal proxy biomarker pipeline.")
report.append("")
report.append("## Main limitation")
report.append("")
report.append("This is still a pilot result. The Phase 8C FTV model used only 13 labeled patients. It should not be described as clinically validated.")
report.append("")
report.append("## Recommended manuscript structure")
report.append("")
for _, r in outline_df.iterrows():
    report.append(f"### {r['section']}")
    report.append(str(r["content"]))
    report.append("")
report.append("## Future temporal graph plan")
report.append("")
report.append("The next scientific step is to build a longitudinal patient-visit graph. Each patient can be represented by serial visits. Each visit can include FTV, PE/SER, tumor shape, and treatment-response features. Edges can encode temporal order and similarity of tumor-change patterns. A temporal graph model can then be tested against the FTV SVM baseline.")
report.append("")
report.append("## Next practical action")
report.append("")
report.append("Use this Phase 8E report as the base for a Word proposal and PowerPoint presentation. Then plan a new technical phase focused on longitudinal FTV / PE-SER graph modeling.")

(REPORTS / "Phase8E_Final_Research_Proposal_and_Manuscript_Plan.md").write_text("\n".join(report))

# Dashboard update
dash = REPORTS / "Dashboard.md"
old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
dash_add = (
    "\n\n## Phase 8E final research plan\n\n"
    "- [Final research proposal and manuscript plan](Phase8E_Final_Research_Proposal_and_Manuscript_Plan.md)\n"
    "- [Final project status](table_views/phase8e_final_project_status.md)\n"
    "- [Manuscript outline](table_views/phase8e_manuscript_outline.md)\n"
    "- [Future temporal graph plan](table_views/phase8e_future_temporal_graph_plan.md)\n"
)
if "Phase 8E final research plan" not in old_dash:
    dash.write_text(old_dash.rstrip() + dash_add)

# README update
readme = REPO / "README.md"
old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
readme_add = (
    "\n\n### Phase 8E: final research proposal and manuscript plan\n\n"
    "Main outputs:\n\n"
    "- reports/Phase8E_Final_Research_Proposal_and_Manuscript_Plan.md\n"
    "- reports/tables/phase8e_final_project_status.csv\n"
    "- reports/figures/270_phase8e_project_pipeline.png\n"
)
if "Phase 8E: final research proposal and manuscript plan" not in old_readme:
    readme.write_text(old_readme.rstrip() + readme_add)

print("Phase 8E complete.")
print("Phase 5A baseline AUROC:", round(baseline_auroc, 4))
print("Phase 8C FTV AUROC:", round(ftv_auroc, 4))
print("AUROC delta:", round(auroc_delta, 4))
print("AUPRC delta:", round(auprc_delta, 4))
print("Brier delta:", round(brier_delta, 4))
