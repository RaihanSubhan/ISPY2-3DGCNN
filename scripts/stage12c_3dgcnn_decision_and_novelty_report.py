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

MODEL_COMPARE = TABLES / "stage12b_3dgcnn_vs_existing_models.csv"
GNN_PERF = TABLES / "stage12b_3dgcnn_model_performance.csv"
GNN_IMPORTANCE = TABLES / "stage12b_3dgcnn_feature_importance.csv"
PHASE8C_PERF = TABLES / "phase8c_ftv_model_performance.csv"
PHASE9C_PLAN = TABLES / "phase9c_next_research_plan.csv"

if not MODEL_COMPARE.exists():
    raise SystemExit("Missing reports/tables/stage12b_3dgcnn_vs_existing_models.csv. Run Stage 12B first.")

compare = pd.read_csv(MODEL_COMPARE, dtype=str).fillna("")

for col in ["AUROC", "AUPRC", "Brier_score"]:
    compare[col] = pd.to_numeric(compare[col], errors="coerce")

# Rank models: AUROC high, AUPRC high, Brier low.
ranked = compare.sort_values(
    ["AUROC", "AUPRC", "Brier_score"],
    ascending=[False, False, True]
).reset_index(drop=True)

ranked["rank"] = np.arange(1, len(ranked) + 1)
ranked = ranked[["rank"] + [c for c in ranked.columns if c != "rank"]]
ranked.to_csv(TABLES / "stage12c_model_ranking.csv", index=False)

best = ranked.iloc[0].to_dict()
gnn_row = ranked[ranked["analysis"].str.contains("3DGCNN", case=False, na=False)]
if len(gnn_row) > 0:
    gnn = gnn_row.iloc[0].to_dict()
else:
    gnn = {}

best_auroc = float(best.get("AUROC", np.nan))
best_auprc = float(best.get("AUPRC", np.nan))
best_brier = float(best.get("Brier_score", np.nan))

gnn_auroc = float(gnn.get("AUROC", np.nan))
gnn_auprc = float(gnn.get("AUPRC", np.nan))
gnn_brier = float(gnn.get("Brier_score", np.nan))

# Feature importance
if GNN_IMPORTANCE.exists():
    imp = pd.read_csv(GNN_IMPORTANCE, dtype=str).fillna("")
    if "importance_auroc_drop" in imp.columns:
        imp["importance_auroc_drop"] = pd.to_numeric(imp["importance_auroc_drop"], errors="coerce")
        imp = imp.sort_values("importance_auroc_drop", ascending=False)
else:
    imp = pd.DataFrame()

imp.to_csv(TABLES / "stage12c_3dgcnn_feature_importance_ranked.csv", index=False)

# Novelty positioning table
novelty = pd.DataFrame([
    {
        "novelty_axis": "Tumor-source validation",
        "what_you_did": "Diagnosed DICOM SEG masks and showed the raw SEG path was not reliable enough for main modeling.",
        "why_it_matters": "Many imaging ML projects assume tumor masks are valid. Your work checks the tumor source first.",
        "paper_role": "Core contribution"
    },
    {
        "novelty_axis": "Official MRI/FTV support-data modeling",
        "what_you_did": "Selected official MRI/FTV support features and trained pCR prediction models.",
        "why_it_matters": "FTV features gave the strongest pilot signal.",
        "paper_role": "Main result"
    },
    {
        "novelty_axis": "Graph-ready longitudinal representation",
        "what_you_did": "Built patient-visit graphs with visit nodes and temporal edges.",
        "why_it_matters": "This creates the bridge toward future temporal 3DGCNN/GNN models.",
        "paper_role": "Future-work bridge"
    },
    {
        "novelty_axis": "Exploratory temporal 3DGCNN",
        "what_you_did": "Implemented a lightweight graph convolution model over patient visits.",
        "why_it_matters": "It proves the model framework can run on Cradle, but it does not beat FTV SVM yet.",
        "paper_role": "Prototype, not main claim"
    },
    {
        "novelty_axis": "3D visual explanation",
        "what_you_did": "Created breast-pair, contrast-transport proxy, 3D tumor surface, vessel schematic, and multivisit growth visuals.",
        "why_it_matters": "These help explain the biological intuition and future graph idea.",
        "paper_role": "Methods visualization"
    }
])
novelty.to_csv(TABLES / "stage12c_novel_paper_positioning.csv", index=False)

# Claims table
claims = pd.DataFrame([
    {
        "claim_type": "safe_main_claim",
        "claim": "Official MRI/FTV support-data features produced the strongest current pilot pCR prediction signal.",
        "evidence": "Phase 8C FTV SVM_RBF ranked first with AUROC 0.6667 and AUPRC 0.7528."
    },
    {
        "claim_type": "safe_secondary_claim",
        "claim": "A temporal 3DGCNN-style graph model was implemented as an exploratory prototype.",
        "evidence": "Stage 12B ran on 13 patient graphs with temporal visit edges."
    },
    {
        "claim_type": "safe_secondary_claim",
        "claim": "The 3DGCNN is not yet the main model.",
        "evidence": "Stage 12B 3DGCNN underperformed Phase 8C FTV SVM."
    },
    {
        "claim_type": "avoid_claim",
        "claim": "The 3DGCNN is clinically validated.",
        "evidence": "Only 13 patient graphs were available."
    },
    {
        "claim_type": "avoid_claim",
        "claim": "The vessel/flow animations measure blood, milk, or fluid flow.",
        "evidence": "They are conceptual contrast-transport proxies only."
    },
    {
        "claim_type": "future_claim",
        "claim": "A larger longitudinal FTV/PE/SER graph model may improve pCR prediction.",
        "evidence": "This should be tested after more patient graphs or stronger tumor-region features are available."
    }
])
claims.to_csv(TABLES / "stage12c_publication_claims.csv", index=False)

# Next experiment plan
plan = pd.DataFrame([
    {
        "priority": 1,
        "next_experiment": "Keep Phase 8C FTV SVM as main paper result",
        "why": "It ranks best by AUROC, AUPRC, and Brier score.",
        "deliverable": "Main results table and figure."
    },
    {
        "priority": 2,
        "next_experiment": "Report Stage 12B 3DGCNN as exploratory prototype",
        "why": "It demonstrates feasibility but does not outperform the FTV model.",
        "deliverable": "Methods/future-work subsection."
    },
    {
        "priority": 3,
        "next_experiment": "Create larger patient-visit graph cohort",
        "why": "A true 3DGCNN paper needs more than 13 patient graphs.",
        "deliverable": "Expanded graph dataset."
    },
    {
        "priority": 4,
        "next_experiment": "Add richer PE/SER or voxel/region features",
        "why": "The current graph nodes are visit-level features, not true 3D tumor-region nodes.",
        "deliverable": "Tumor-region graph nodes."
    },
    {
        "priority": 5,
        "next_experiment": "Retest GNN only after cohort expansion",
        "why": "Training deep graph models on 13 graphs risks overfitting.",
        "deliverable": "Stage 13 or future paper."
    }
])
plan.to_csv(TABLES / "stage12c_next_3dgcnn_experiment_plan.csv", index=False)

# -------------------------------------------------------
# Figures
# -------------------------------------------------------
plt.figure(figsize=(10, 5))
plot = ranked.copy()
plot["label"] = plot["analysis"] + "\n" + plot["best_model"]
plt.barh(plot["label"], plot["AUROC"])
plt.xlabel("AUROC")
plt.title("Stage 12C Model Ranking by AUROC")
plt.tight_layout()
plt.savefig(FIGS / "520_stage12c_model_ranking.png", dpi=220)
plt.close()

plt.figure(figsize=(9, 5))
if not imp.empty and "importance_auroc_drop" in imp.columns:
    tmp = imp.head(12).sort_values("importance_auroc_drop")
    plt.barh(tmp["feature"], tmp["importance_auroc_drop"])
    plt.xlabel("AUROC drop after permutation")
    plt.title("Stage 12C 3DGCNN Node Feature Importance")
else:
    plt.text(0.5, 0.5, "No feature importance table available", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "521_stage12c_3dgcnn_feature_importance.png", dpi=220)
plt.close()

plt.figure(figsize=(11, 6))
plt.axis("off")
text = (
    "Novel paper positioning\n\n"
    "1. Tumor-source validation\n"
    "   DICOM SEG path was diagnosed before modeling.\n\n"
    "2. Reliable support-data modeling\n"
    "   Official MRI/FTV features gave the strongest pilot pCR signal.\n\n"
    "3. Future graph learning bridge\n"
    "   Patient visits became graph nodes with temporal edges.\n\n"
    "4. 3DGCNN prototype\n"
    "   Implemented successfully, but not main model yet.\n\n"
    "Main current result: Phase 8C FTV SVM_RBF"
)
plt.text(0.5, 0.5, text, ha="center", va="center", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "522_stage12c_novelty_map.png", dpi=220)
plt.close()

plt.figure(figsize=(11, 6))
plt.axis("off")
story = (
    "Paper story flow\n\n"
    "ISPY2 MRI on Cradle\n"
    "↓\n"
    "Tumor-source validation\n"
    "↓\n"
    "DICOM SEG path found weak for main modeling\n"
    "↓\n"
    "Official MRI/FTV support features selected\n"
    "↓\n"
    "FTV SVM gives best pilot pCR result\n"
    "↓\n"
    "Longitudinal graph dataset built\n"
    "↓\n"
    "3DGCNN prototype tested\n"
    "↓\n"
    "3DGCNN remains future-work model"
)
plt.text(0.5, 0.5, story, ha="center", va="center", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "523_stage12c_paper_story_flow.png", dpi=220)
plt.close()

# Markdown helper
def make_md(csv_path, md_path, title, max_rows=80, max_cols=10):
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    small = df.head(max_rows).iloc[:, :max_cols]

    def clean(x):
        x = str(x).replace("|", "\\|").replace("\n", " ")
        return x[:87] + "..." if len(x) > 90 else x

    lines = []
    lines.append("# " + title)
    lines.append("")
    lines.append("Rows: " + str(len(df)))
    lines.append("Columns: " + str(len(df.columns)))
    lines.append("")
    lines.append("| " + " | ".join(clean(c) for c in small.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(small.columns)) + " |")

    for _, r in small.iterrows():
        lines.append("| " + " | ".join(clean(v) for v in r.tolist()) + " |")

    md_path.write_text("\n".join(lines))

make_md(TABLES / "stage12c_model_ranking.csv", VIEWS / "stage12c_model_ranking.md", "Stage 12C Model Ranking")
make_md(TABLES / "stage12c_3dgcnn_feature_importance_ranked.csv", VIEWS / "stage12c_3dgcnn_feature_importance_ranked.md", "Stage 12C 3DGCNN Feature Importance")
make_md(TABLES / "stage12c_novel_paper_positioning.csv", VIEWS / "stage12c_novel_paper_positioning.md", "Stage 12C Novel Paper Positioning")
make_md(TABLES / "stage12c_publication_claims.csv", VIEWS / "stage12c_publication_claims.md", "Stage 12C Publication Claims")
make_md(TABLES / "stage12c_next_3dgcnn_experiment_plan.csv", VIEWS / "stage12c_next_3dgcnn_experiment_plan.md", "Stage 12C Next 3DGCNN Experiment Plan")

# Report
report = []
report.append("# Stage 12C 3DGCNN Decision and Novelty Report")
report.append("")
report.append("This stage interprets the Stage 12B exploratory temporal 3DGCNN result and positions the novel paper idea.")
report.append("")
report.append("## Main conclusion")
report.append("")
report.append("The exploratory temporal 3DGCNN ran successfully, but it did not outperform the Phase 8C FTV SVM model. Therefore, the FTV SVM remains the current main pilot result, and the 3DGCNN should be reported as an exploratory prototype and future-work direction.")
report.append("")
report.append("## Model ranking")
report.append("")
for _, row in ranked.iterrows():
    report.append(f"- Rank {row['rank']}: {row['analysis']} / {row['best_model']} with AUROC {float(row['AUROC']):.4f}, AUPRC {float(row['AUPRC']):.4f}, Brier {float(row['Brier_score']):.4f}.")
report.append("")
report.append("## Novel paper idea")
report.append("")
report.append("A strong paper should focus on tumor-source validation and FTV-based pCR prediction, with an exploratory temporal 3DGCNN prototype. This is stronger than claiming the 3DGCNN is already the best model.")
report.append("")
report.append("## Why this is novel")
report.append("")
report.append("The project does not blindly apply deep learning. It first validates the tumor source, rejects weak DICOM SEG masks as the main path, uses official MRI/FTV support features for the primary pilot model, and then builds a graph-ready longitudinal framework for future 3DGCNN work.")
report.append("")
report.append("## Safe manuscript claim")
report.append("")
report.append("Official MRI/FTV support features gave the strongest current pilot pCR prediction result. A temporal 3DGCNN prototype was implemented but remains exploratory because of small sample size.")
report.append("")
report.append("## Claim to avoid")
report.append("")
report.append("Do not claim that the 3DGCNN is clinically validated or that it is currently better than the FTV SVM model.")
report.append("")
report.append("## Next step")
report.append("")
report.append("Use Stage 12C to update the paper outline. The title should mention FTV-based pCR prediction, tumor-source validation, and exploratory temporal 3DGCNN.")

(REPORTS / "Stage12C_3DGCNN_Decision_and_Novelty_Report.md").write_text("\n".join(report))

# Dashboard and README
dash = REPORTS / "Dashboard.md"
old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
dash_add = (
    "\n\n## Stage 12C 3DGCNN decision and novelty\n\n"
    "- [3DGCNN decision and novelty report](Stage12C_3DGCNN_Decision_and_Novelty_Report.md)\n"
    "- [Model ranking](table_views/stage12c_model_ranking.md)\n"
    "- [Novel paper positioning](table_views/stage12c_novel_paper_positioning.md)\n"
)
if "Stage 12C 3DGCNN decision and novelty" not in old_dash:
    dash.write_text(old_dash.rstrip() + dash_add)

readme = REPO / "README.md"
old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
readme_add = (
    "\n\n### Stage 12C: 3DGCNN decision and novelty report\n\n"
    "Main outputs:\n\n"
    "- reports/Stage12C_3DGCNN_Decision_and_Novelty_Report.md\n"
    "- reports/tables/stage12c_model_ranking.csv\n"
    "- reports/figures/522_stage12c_novelty_map.png\n"
)
if "Stage 12C: 3DGCNN decision and novelty report" not in old_readme:
    readme.write_text(old_readme.rstrip() + readme_add)

print("Stage 12C complete.")
print("Best current model:", best.get("analysis", ""), "/", best.get("best_model", ""))
print("Best AUROC:", round(best_auroc, 4))
print("3DGCNN AUROC:", round(gnn_auroc, 4))
print("Decision: FTV SVM remains main model; 3DGCNN is exploratory/future work.")
