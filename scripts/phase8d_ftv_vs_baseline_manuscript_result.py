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

BASELINE_PATH = TABLES / "phase5a_model_calibration_metrics.csv"
FTV_PATH = TABLES / "phase8c_ftv_model_performance.csv"
FTV_SOURCE_PATH = TABLES / "phase8c_ftv_source_selection.csv"
FTV_IMPORTANCE_PATH = TABLES / "phase8c_ftv_feature_importance.csv"
FTV_DATASET_PATH = TABLES / "phase8c_ftv_modeling_dataset.csv"

if not BASELINE_PATH.exists():
    raise SystemExit("Missing Phase 5A baseline metrics. Run Phase 5A first.")

if not FTV_PATH.exists():
    raise SystemExit("Missing Phase 8C FTV model performance. Run Phase 8C first.")

baseline = pd.read_csv(BASELINE_PATH)
ftv = pd.read_csv(FTV_PATH)

def best_model(df):
    d = df.copy()
    for c in ["AUROC", "AUPRC", "Brier_score"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values(["AUROC", "AUPRC", "Brier_score"], ascending=[False, False, True])
    return d.iloc[0].to_dict(), d

baseline_best, baseline_sorted = best_model(baseline)
ftv_best, ftv_sorted = best_model(ftv)

comparison = pd.DataFrame([
    {
        "analysis": "Phase 5A baseline biomarker model",
        "best_model": baseline_best.get("model", ""),
        "n_patients": baseline_best.get("n_patients", ""),
        "positive_pCR": baseline_best.get("positive_labels", baseline_best.get("positive_pCR", "")),
        "negative_non_pCR": baseline_best.get("negative_labels", baseline_best.get("negative_non_pCR", "")),
        "n_features": baseline_best.get("n_features", ""),
        "AUROC": baseline_best.get("AUROC", ""),
        "AUPRC": baseline_best.get("AUPRC", ""),
        "Brier_score": baseline_best.get("Brier_score", ""),
        "interpretation": "Earlier internal baseline using proxy biomarker table."
    },
    {
        "analysis": "Phase 8C FTV support-data model",
        "best_model": ftv_best.get("model", ""),
        "n_patients": ftv_best.get("n_patients", ""),
        "positive_pCR": ftv_best.get("positive_pCR", ""),
        "negative_non_pCR": ftv_best.get("negative_non_pCR", ""),
        "n_features": ftv_best.get("n_features", ""),
        "AUROC": ftv_best.get("AUROC", ""),
        "AUPRC": ftv_best.get("AUPRC", ""),
        "Brier_score": ftv_best.get("Brier_score", ""),
        "interpretation": "Support-data model using official MRI / FTV features."
    }
])

comparison.to_csv(TABLES / "phase8d_baseline_vs_ftv_model_comparison.csv", index=False)

delta = pd.DataFrame([{
    "metric": "AUROC",
    "phase5a_baseline": float(baseline_best.get("AUROC", np.nan)),
    "phase8c_ftv": float(ftv_best.get("AUROC", np.nan)),
    "difference_phase8c_minus_phase5a": float(ftv_best.get("AUROC", np.nan)) - float(baseline_best.get("AUROC", np.nan))
},
{
    "metric": "AUPRC",
    "phase5a_baseline": float(baseline_best.get("AUPRC", np.nan)),
    "phase8c_ftv": float(ftv_best.get("AUPRC", np.nan)),
    "difference_phase8c_minus_phase5a": float(ftv_best.get("AUPRC", np.nan)) - float(baseline_best.get("AUPRC", np.nan))
},
{
    "metric": "Brier_score",
    "phase5a_baseline": float(baseline_best.get("Brier_score", np.nan)),
    "phase8c_ftv": float(ftv_best.get("Brier_score", np.nan)),
    "difference_phase8c_minus_phase5a": float(ftv_best.get("Brier_score", np.nan)) - float(baseline_best.get("Brier_score", np.nan))
}])

delta.to_csv(TABLES / "phase8d_metric_delta_summary.csv", index=False)

# Study claim table
claims = pd.DataFrame([
    {
        "claim_type": "safe_claim",
        "claim": "A reproducible Cradle and GitHub pipeline was built for ISPY2 inventory, label mapping, feature extraction, and pilot ML.",
        "reason": "This is supported by the generated phase reports and committed outputs."
    },
    {
        "claim_type": "safe_claim",
        "claim": "The official FTV / multi-feature MRI support-data path is stronger than the recovered DICOM SEG path.",
        "reason": "Phase 8C created a usable pCR modeling set, while Phase 7E/7F could not support ML from recovered SEG masks."
    },
    {
        "claim_type": "safe_claim",
        "claim": "The Phase 8C FTV support-data model improved AUROC over the Phase 5A baseline in this pilot experiment.",
        "reason": "Phase 8C best AUROC is higher than Phase 5A best AUROC."
    },
    {
        "claim_type": "avoid_claim",
        "claim": "The model is clinically validated.",
        "reason": "The cohort is small, only 13 labeled FTV patients were used for Phase 8C."
    },
    {
        "claim_type": "avoid_claim",
        "claim": "DICOM SEG tumor masks are reliable tumor masks.",
        "reason": "Phase 7B showed current DICOM SEG masks were mostly full-slice masks, and only a limited number were recoverable by thresholding."
    },
    {
        "claim_type": "future_claim",
        "claim": "A temporal graph model may improve response prediction after stable patient-visit tumor features are built.",
        "reason": "This is a future direction, not yet tested in the current repo."
    }
])

claims.to_csv(TABLES / "phase8d_safe_claims_and_limitations.csv", index=False)

# Optional feature importance
if FTV_IMPORTANCE_PATH.exists():
    importance = pd.read_csv(FTV_IMPORTANCE_PATH)
else:
    importance = pd.DataFrame()

# Figure 260: AUROC/AUPRC/Brier comparison
fig, ax = plt.subplots(figsize=(9, 5))
plot_df = delta.copy()
plot_df = plot_df[plot_df["metric"].isin(["AUROC", "AUPRC", "Brier_score"])]
x = np.arange(len(plot_df))
width = 0.35

ax.bar(x - width/2, plot_df["phase5a_baseline"], width, label="Phase 5A baseline")
ax.bar(x + width/2, plot_df["phase8c_ftv"], width, label="Phase 8C FTV")
ax.set_xticks(x)
ax.set_xticklabels(plot_df["metric"])
ax.set_title("Phase 8D: Baseline vs FTV Support-Data Model")
ax.set_ylabel("Score")
ax.legend()
plt.tight_layout()
plt.savefig(FIGS / "260_phase8d_baseline_vs_ftv_metrics.png", dpi=220)
plt.close()

# Figure 261: cohort size comparison
fig, ax = plt.subplots(figsize=(8, 5))
cohort_labels = ["Phase 5A baseline", "Phase 8C FTV"]
cohort_sizes = [
    float(baseline_best.get("n_patients", 0)),
    float(ftv_best.get("n_patients", 0))
]
ax.bar(cohort_labels, cohort_sizes)
ax.set_ylabel("Patients")
ax.set_title("Phase 8D Cohort Size Comparison")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(FIGS / "261_phase8d_cohort_size_comparison.png", dpi=220)
plt.close()

# Figure 262: all model AUROC comparison
all_rows = []
for _, r in baseline_sorted.iterrows():
    all_rows.append({
        "analysis": "Phase 5A",
        "model": r.get("model", ""),
        "AUROC": r.get("AUROC", np.nan)
    })
for _, r in ftv_sorted.iterrows():
    all_rows.append({
        "analysis": "Phase 8C",
        "model": r.get("model", ""),
        "AUROC": r.get("AUROC", np.nan)
    })

all_model = pd.DataFrame(all_rows)
all_model["AUROC"] = pd.to_numeric(all_model["AUROC"], errors="coerce")
all_model.to_csv(TABLES / "phase8d_all_model_auroc_comparison.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 6))
if len(all_model) > 0:
    labels = all_model["analysis"] + " | " + all_model["model"]
    ax.barh(labels, all_model["AUROC"])
    ax.set_xlabel("AUROC")
    ax.set_title("Phase 8D All Model AUROC Comparison")
else:
    ax.text(0.5, 0.5, "No model comparison data", ha="center", va="center")
    ax.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "262_phase8d_all_model_auroc.png", dpi=220)
plt.close()

# Figure 263: top FTV feature importance
fig, ax = plt.subplots(figsize=(10, 7))
if not importance.empty and "permutation_importance_mean" in importance.columns:
    imp = importance.copy()
    imp["permutation_importance_mean"] = pd.to_numeric(imp["permutation_importance_mean"], errors="coerce")
    imp = imp.dropna(subset=["permutation_importance_mean"])
    imp = imp.sort_values("permutation_importance_mean", ascending=False).head(12).sort_values("permutation_importance_mean")
    if len(imp) > 0:
        ax.barh(imp["feature"], imp["permutation_importance_mean"])
        ax.set_xlabel("Permutation importance")
        ax.set_title("Phase 8D Top FTV Feature Importance")
    else:
        ax.text(0.5, 0.5, "No valid feature importance values", ha="center", va="center")
        ax.axis("off")
else:
    ax.text(0.5, 0.5, "Feature importance table not available", ha="center", va="center")
    ax.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "263_phase8d_ftv_feature_importance.png", dpi=220)
plt.close()

# Figure 264: project decision flow
fig, ax = plt.subplots(figsize=(12, 7))
ax.axis("off")
flow = (
    "Project decision after segmentation diagnosis\n\n"
    "DICOM SEG path\n"
    "  Phase 7B: full-slice masks found\n"
    "  Phase 7C/7D: limited recovered masks\n"
    "  Phase 7E/7F: too few labeled recovered-mask patients\n\n"
    "Support-data FTV path\n"
    "  Phase 8A/8B: official support source found\n"
    "  Phase 8C: FTV model ran successfully\n"
    f"  Best model: {ftv_best.get('model', '')}, AUROC={float(ftv_best.get('AUROC', np.nan)):.3f}\n\n"
    "Next: manuscript-ready pilot claim + future temporal graph plan"
)
ax.text(0.5, 0.5, flow, ha="center", va="center", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "264_phase8d_decision_flow.png", dpi=220)
plt.close()

# Markdown helper
def make_md(csv_path, md_path, title, max_rows=80, max_cols=10):
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
    TABLES / "phase8d_baseline_vs_ftv_model_comparison.csv",
    VIEWS / "phase8d_baseline_vs_ftv_model_comparison.md",
    "Phase 8D Baseline vs FTV Model Comparison"
)

make_md(
    TABLES / "phase8d_metric_delta_summary.csv",
    VIEWS / "phase8d_metric_delta_summary.md",
    "Phase 8D Metric Delta Summary"
)

make_md(
    TABLES / "phase8d_safe_claims_and_limitations.csv",
    VIEWS / "phase8d_safe_claims_and_limitations.md",
    "Phase 8D Safe Claims and Limitations"
)

make_md(
    TABLES / "phase8d_all_model_auroc_comparison.csv",
    VIEWS / "phase8d_all_model_auroc_comparison.md",
    "Phase 8D All Model AUROC Comparison"
)

# Final Phase 8D report
auroc_delta = float(delta[delta["metric"] == "AUROC"]["difference_phase8c_minus_phase5a"].iloc[0])
auprc_delta = float(delta[delta["metric"] == "AUPRC"]["difference_phase8c_minus_phase5a"].iloc[0])
brier_delta = float(delta[delta["metric"] == "Brier_score"]["difference_phase8c_minus_phase5a"].iloc[0])

report = []
report.append("# Phase 8D FTV vs Baseline Manuscript-Ready Result")
report.append("")
report.append("## Summary")
report.append("")
report.append("This phase compares the earlier Phase 5A baseline model against the Phase 8C FTV support-data model.")
report.append("")
report.append("## Main result")
report.append("")
report.append(f"The Phase 5A best model was **{baseline_best.get('model', '')}**, with AUROC = {float(baseline_best.get('AUROC', np.nan)):.4f}, AUPRC = {float(baseline_best.get('AUPRC', np.nan)):.4f}, and Brier score = {float(baseline_best.get('Brier_score', np.nan)):.4f}.")
report.append("")
report.append(f"The Phase 8C FTV best model was **{ftv_best.get('model', '')}**, with AUROC = {float(ftv_best.get('AUROC', np.nan)):.4f}, AUPRC = {float(ftv_best.get('AUPRC', np.nan)):.4f}, and Brier score = {float(ftv_best.get('Brier_score', np.nan)):.4f}.")
report.append("")
report.append(f"The FTV path improved AUROC by **{auroc_delta:.4f}** and AUPRC by **{auprc_delta:.4f}** compared with the Phase 5A baseline. The Brier score changed by **{brier_delta:.4f}**; lower Brier is better.")
report.append("")
report.append("## Scientific interpretation")
report.append("")
report.append("The result supports moving the project away from weak DICOM SEG recovery and toward official support-data FTV / MRI features. The FTV path gives a cleaner tumor-response signal and a working pCR prediction pipeline.")
report.append("")
report.append("## Important limitation")
report.append("")
report.append("This is still a pilot result. The Phase 8C FTV model used only 13 labeled patients. Therefore, this should not be claimed as final clinical performance.")
report.append("")
report.append("## Safe manuscript claim")
report.append("")
report.append("A safe claim is: official MRI / FTV support features produced a stronger pilot pCR prediction signal than the earlier internal proxy biomarker pipeline.")
report.append("")
report.append("## Claim to avoid")
report.append("")
report.append("Do not claim that the model is clinically validated or ready for decision-making.")
report.append("")
report.append("## Next step")
report.append("")
report.append("Phase 8E should prepare the final research proposal/manuscript structure, including background, methods, results, limitations, and future temporal graph model plan.")
report.append("")
report.append("## Main outputs")
report.append("")
report.append("- reports/tables/phase8d_baseline_vs_ftv_model_comparison.csv")
report.append("- reports/tables/phase8d_metric_delta_summary.csv")
report.append("- reports/tables/phase8d_safe_claims_and_limitations.csv")
report.append("- reports/figures/260_phase8d_baseline_vs_ftv_metrics.png")
report.append("- reports/figures/261_phase8d_cohort_size_comparison.png")
report.append("- reports/figures/262_phase8d_all_model_auroc.png")
report.append("- reports/figures/263_phase8d_ftv_feature_importance.png")
report.append("- reports/figures/264_phase8d_decision_flow.png")

(REPORTS / "Phase8D_FTV_vs_Baseline_Manuscript_Result.md").write_text("\n".join(report))

# Dashboard update
dash = REPORTS / "Dashboard.md"
old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
dash_add = "\n\n## Phase 8D FTV vs baseline result\n\n- [FTV vs baseline manuscript result](Phase8D_FTV_vs_Baseline_Manuscript_Result.md)\n- [Baseline vs FTV comparison](table_views/phase8d_baseline_vs_ftv_model_comparison.md)\n- [Metric delta summary](table_views/phase8d_metric_delta_summary.md)\n"
if "Phase 8D FTV vs baseline result" not in old_dash:
    dash.write_text(old_dash.rstrip() + dash_add)

# README update
readme = REPO / "README.md"
old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
readme_add = "\n\n### Phase 8D: FTV vs baseline manuscript-ready result\n\nMain outputs:\n\n- reports/Phase8D_FTV_vs_Baseline_Manuscript_Result.md\n- reports/tables/phase8d_baseline_vs_ftv_model_comparison.csv\n- reports/figures/260_phase8d_baseline_vs_ftv_metrics.png\n"
if "Phase 8D: FTV vs baseline manuscript-ready result" not in old_readme:
    readme.write_text(old_readme.rstrip() + readme_add)

print("Phase 8D complete.")
print("Phase 5A best model:", baseline_best.get("model", ""))
print("Phase 5A AUROC:", round(float(baseline_best.get("AUROC", np.nan)), 4))
print("Phase 8C best model:", ftv_best.get("model", ""))
print("Phase 8C AUROC:", round(float(ftv_best.get("AUROC", np.nan)), 4))
print("AUROC delta:", round(auroc_delta, 4))
print("AUPRC delta:", round(auprc_delta, 4))
print("Brier delta:", round(brier_delta, 4))
