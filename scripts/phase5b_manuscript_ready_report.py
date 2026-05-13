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

metrics_path = TABLES / "phase5a_model_calibration_metrics.csv"
importance_path = TABLES / "phase5a_feature_importance.csv"
recommend_path = TABLES / "phase5a_model_recommendation.csv"
model_data_path = TABLES / "phase4b_modeling_dataset.csv"

if not metrics_path.exists():
    raise SystemExit("Missing phase5a_model_calibration_metrics.csv. Run Phase 5A first.")

if not model_data_path.exists():
    raise SystemExit("Missing phase4b_modeling_dataset.csv. Run Phase 4B first.")

metrics = pd.read_csv(metrics_path)
model_data = pd.read_csv(model_data_path)

importance = pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame()
recommend = pd.read_csv(recommend_path) if recommend_path.exists() else pd.DataFrame()

# Basic cohort counts
n_patients = len(model_data)
label_col = "label"
n_pos = int((pd.to_numeric(model_data[label_col], errors="coerce") == 1).sum())
n_neg = int((pd.to_numeric(model_data[label_col], errors="coerce") == 0).sum())
pcr_rate = n_pos / n_patients if n_patients else np.nan

# Best model
metrics_sorted = metrics.sort_values(["AUROC", "AUPRC", "Brier_score"], ascending=[False, False, True])
best = metrics_sorted.iloc[0]

best_model = str(best["model"])
best_auroc = float(best["AUROC"])
best_auprc = float(best["AUPRC"])
best_brier = float(best["Brier_score"])
best_bal_acc = float(best["balanced_accuracy"]) if "balanced_accuracy" in best else np.nan
best_f1 = float(best["F1"]) if "F1" in best else np.nan

# Key results table
key_rows = [
    {
        "section": "Cohort",
        "item": "Labeled patients",
        "value": n_patients,
        "interpretation": "Small pilot cohort for feasibility analysis."
    },
    {
        "section": "Cohort",
        "item": "pCR positive",
        "value": n_pos,
        "interpretation": "Patients labeled as pCR or responder."
    },
    {
        "section": "Cohort",
        "item": "non-pCR negative",
        "value": n_neg,
        "interpretation": "Patients labeled as non-pCR or non-responder."
    },
    {
        "section": "Cohort",
        "item": "pCR rate",
        "value": round(pcr_rate, 4),
        "interpretation": "This is also the baseline rate for AUPRC comparison."
    },
    {
        "section": "Best model",
        "item": "Recommended pilot model",
        "value": best_model,
        "interpretation": "Selected by AUROC, then AUPRC, then Brier score."
    },
    {
        "section": "Performance",
        "item": "AUROC",
        "value": round(best_auroc, 4),
        "interpretation": "Discrimination is weak in this small pilot cohort."
    },
    {
        "section": "Performance",
        "item": "AUPRC",
        "value": round(best_auprc, 4),
        "interpretation": "Compare with pCR rate baseline."
    },
    {
        "section": "Performance",
        "item": "Brier score",
        "value": round(best_brier, 4),
        "interpretation": "Probability calibration error. Lower is better."
    },
]

key_df = pd.DataFrame(key_rows)
key_df.to_csv(TABLES / "phase5b_key_results_summary.csv", index=False)

# Limitations and next steps table
lim_rows = [
    {
        "issue": "Small labeled cohort",
        "current_state": f"Only {n_patients} labeled patients were used.",
        "risk": "Model performance may be unstable.",
        "next_step": "Expand geometry-ready feature extraction to more patients."
    },
    {
        "issue": "Pilot-level biomarkers",
        "current_state": "Current features are geometry QC and volume-proxy based.",
        "risk": "They may not fully represent vascular-perfusion biology.",
        "next_step": "Compute verified tumor, peritumor, vesselness, radial signal, and longitudinal delta features."
    },
    {
        "issue": "No external validation",
        "current_state": "All evaluation is internal cross-validation.",
        "risk": "Generalization is unknown.",
        "next_step": "Use a held-out cohort or external breast MRI dataset later."
    },
    {
        "issue": "Deep learning not yet justified",
        "current_state": "Only 33 labels are available.",
        "risk": "CNN, GNN, or RL models may overfit.",
        "next_step": "Use interpretable ML first, then add deep models after cohort expansion."
    },
    {
        "issue": "XGBoost unavailable",
        "current_state": "Phase 5A reported XGBoost as not available.",
        "risk": "XGBoost was not compared yet.",
        "next_step": "Install XGBoost later and rerun Phase 5A after feature expansion."
    },
]

lim_df = pd.DataFrame(lim_rows)
lim_df.to_csv(TABLES / "phase5b_limitations_and_next_steps.csv", index=False)

# Markdown table helper
def make_md(csv_path, md_path, title, max_rows=50, max_cols=10):
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

make_md(
    TABLES / "phase5b_key_results_summary.csv",
    VIEWS / "phase5b_key_results_summary.md",
    "Phase 5B Key Results Summary"
)

make_md(
    TABLES / "phase5b_limitations_and_next_steps.csv",
    VIEWS / "phase5b_limitations_and_next_steps.md",
    "Phase 5B Limitations and Next Steps"
)

# Figure 120: pipeline summary
plt.figure(figsize=(12, 7))
plt.axis("off")
pipeline_text = (
    "ISPY2 4D Vascular-Perfusion Tumor Atlas\\n\\n"
    "Phase 1: DICOM inventory + visit tracking\\n"
    "↓\\n"
    "Phase 2: SEG-to-MR linking\\n"
    "↓\\n"
    "Phase 3: geometry QC + biomarker-ready table\\n"
    "↓\\n"
    "Phase 4: pCR label acquisition + baseline ML\\n"
    "↓\\n"
    "Phase 5: calibration, interpretation, and manuscript-ready reporting\\n\\n"
    f"Current pilot cohort: n={n_patients}, pCR={n_pos}, non-pCR={n_neg}\\n"
    f"Best pilot model: {best_model}, AUROC={best_auroc:.3f}, AUPRC={best_auprc:.3f}"
)
plt.text(0.5, 0.5, pipeline_text, ha="center", va="center", fontsize=14)
plt.tight_layout()
plt.savefig(FIGS / "120_phase5b_pipeline_summary.png", dpi=220)
plt.close()

# Figure 121: model performance summary
plt.figure(figsize=(10, 6))
plot_df = metrics_sorted.copy()
plot_df = plot_df[["model", "AUROC", "AUPRC"]].set_index("model")
plot_df.plot(kind="bar")
plt.title("Phase 5B Model Performance Summary")
plt.ylabel("Score")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(FIGS / "121_phase5b_model_performance_summary.png", dpi=220)
plt.close()

# Figure 122: top feature importance
plt.figure(figsize=(10, 7))
if not importance.empty and "permutation_importance_mean" in importance.columns:
    imp = importance.copy()
    imp["permutation_importance_mean"] = pd.to_numeric(imp["permutation_importance_mean"], errors="coerce")
    imp = imp.dropna(subset=["permutation_importance_mean"])
    imp = imp.sort_values("permutation_importance_mean", ascending=False).head(12).sort_values("permutation_importance_mean")

    if len(imp) > 0:
        plt.barh(imp["feature"], imp["permutation_importance_mean"])
        plt.xlabel("Permutation importance")
        plt.title("Top Pilot Biomarker Features")
    else:
        plt.text(0.5, 0.5, "No valid feature importance values", ha="center", va="center")
        plt.axis("off")
else:
    plt.text(0.5, 0.5, "Feature importance table not available", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "122_phase5b_top_feature_importance.png", dpi=220)
plt.close()

# Manuscript-ready Markdown report
report = []
report.append("# Phase 5B Manuscript-Ready Results Report")
report.append("")
report.append("## Working title")
report.append("")
report.append("A 4D Vascular-Perfusion Tumor Atlas for Early Prediction of Pathologic Complete Response in Breast Cancer Using I-SPY2 MRI")
report.append("")
report.append("## Study aim")
report.append("")
report.append("This study aims to build a longitudinal imaging atlas that links patient visits, DICOM SEG tumor masks, MR imaging, geometry-verified tumor biomarkers, and pCR outcome labels.")
report.append("")
report.append("## Current cohort")
report.append("")
report.append(f"The current supervised pilot cohort contains {n_patients} labeled patients. Among them, {n_pos} are pCR positive and {n_neg} are non-pCR negative. The pCR rate is {pcr_rate:.3f}.")
report.append("")
report.append("## Modeling method")
report.append("")
report.append("Baseline supervised models were trained using cross-validated probability estimates. The tested models included logistic regression, SVM, KNN, random forest, and gradient boosting. XGBoost was not available in the current environment.")
report.append("")
report.append("## Main result")
report.append("")
report.append(f"The recommended pilot model was **{best_model}**. Its AUROC was **{best_auroc:.4f}**, AUPRC was **{best_auprc:.4f}**, and Brier score was **{best_brier:.4f}**.")
report.append("")
report.append("## Interpretation")
report.append("")
report.append("The current AUROC shows weak discrimination. This does not mean the project failed. It means the current feature table and labeled cohort are still at a feasibility stage. The result supports a complete working pipeline, from raw ISPY2 DICOM data to GitHub-tracked model outputs, but it should not be presented as final clinical performance.")
report.append("")
report.append("## Why logistic regression is acceptable here")
report.append("")
report.append("Logistic regression is suitable as the current pilot model because the labeled sample size is small. With only 33 labeled patients, more flexible models may overfit. Logistic regression gives a stable and interpretable baseline before moving to XGBoost, temporal models, or deep learning.")
report.append("")
report.append("## Main limitations")
report.append("")
report.append("1. The labeled cohort is small.")
report.append("2. The current biomarkers are still proxy and QC-driven features.")
report.append("3. The model has no external validation.")
report.append("4. The DCE vascular-perfusion component still needs stronger pharmacokinetic and vesselness features.")
report.append("5. The current result should be treated as method development, not clinical deployment.")
report.append("")
report.append("## Recommended next phase")
report.append("")
report.append("Phase 6 should improve the scientific strength of the project by expanding the biomarker feature set. The next technical target is to compute verified tumor and peritumor vascular features across more geometry-ready cases. These should include tumor volume, shape, peritumor rim features, Frangi vesselness, radial signal, and longitudinal change features.")
report.append("")
report.append("## Manuscript claim that is safe")
report.append("")
report.append("A safe claim is: the study developed a reproducible HPC and GitHub-tracked pipeline for ISPY2 DICOM inventory, SEG-MR linking, geometry QC, pCR label mapping, and pilot ML modeling.")
report.append("")
report.append("## Manuscript claim to avoid")
report.append("")
report.append("Do not claim that the model is clinically accurate yet. The current AUROC is not strong enough for that claim.")
report.append("")
report.append("## Files generated")
report.append("")
report.append("- `reports/tables/phase5b_key_results_summary.csv`")
report.append("- `reports/tables/phase5b_limitations_and_next_steps.csv`")
report.append("- `reports/figures/120_phase5b_pipeline_summary.png`")
report.append("- `reports/figures/121_phase5b_model_performance_summary.png`")
report.append("- `reports/figures/122_phase5b_top_feature_importance.png`")

(REPORTS / "Phase5B_Manuscript_Ready_Results_Report.md").write_text("\n".join(report))

# Dashboard update
dash_path = REPORTS / "Dashboard.md"
old_dash = dash_path.read_text() if dash_path.exists() else "# ISPY2 4D Atlas Dashboard\n"
dash_add = (
    "\n\n## Phase 5B manuscript-ready outputs\n\n"
    "- [Manuscript-ready results report](Phase5B_Manuscript_Ready_Results_Report.md)\n"
    "- [Key results summary](table_views/phase5b_key_results_summary.md)\n"
    "- [Limitations and next steps](table_views/phase5b_limitations_and_next_steps.md)\n"
)
if "Phase 5B manuscript-ready outputs" not in old_dash:
    dash_path.write_text(old_dash.rstrip() + dash_add)

# README update
readme_path = REPO / "README.md"
old_readme = readme_path.read_text() if readme_path.exists() else "# ISPY2-3DGCNN\n"
readme_add = (
    "\n\n### Phase 5B: manuscript-ready pilot report\n\n"
    "Main outputs:\n\n"
    "- reports/Phase5B_Manuscript_Ready_Results_Report.md\n"
    "- reports/tables/phase5b_key_results_summary.csv\n"
    "- reports/figures/120_phase5b_pipeline_summary.png\n"
)
if "Phase 5B: manuscript-ready pilot report" not in old_readme:
    readme_path.write_text(old_readme.rstrip() + readme_add)

print("Phase 5B complete.")
print("Best pilot model:", best_model)
print("AUROC:", round(best_auroc, 4))
print("AUPRC:", round(best_auprc, 4))
print("Brier score:", round(best_brier, 4))
print("Patients:", n_patients)
