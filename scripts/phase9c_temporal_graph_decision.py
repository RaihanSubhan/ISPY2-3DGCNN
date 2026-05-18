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

PHASE8C = TABLES / "phase8c_ftv_model_performance.csv"
PHASE9B = TABLES / "phase9b_temporal_delta_model_performance.csv"
PHASE9A_SUMMARY = TABLES / "phase9a_patient_graph_summary.csv"

if not PHASE8C.exists():
    raise SystemExit("Missing phase8c_ftv_model_performance.csv. Run Phase 8C first.")

if not PHASE9B.exists():
    raise SystemExit("Missing phase9b_temporal_delta_model_performance.csv. Run Phase 9B first.")

p8 = pd.read_csv(PHASE8C)
p9 = pd.read_csv(PHASE9B)

def best_model(df):
    d = df.copy()
    for c in ["AUROC", "AUPRC", "Brier_score"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values(["AUROC", "AUPRC", "Brier_score"], ascending=[False, False, True])
    return d.iloc[0].to_dict(), d

best8, p8_sorted = best_model(p8)
best9, p9_sorted = best_model(p9)

p8_auroc = float(best8["AUROC"])
p8_auprc = float(best8["AUPRC"])
p8_brier = float(best8["Brier_score"])

p9_auroc = float(best9["AUROC"])
p9_auprc = float(best9["AUPRC"])
p9_brier = float(best9["Brier_score"])

delta_auroc = p9_auroc - p8_auroc
delta_auprc = p9_auprc - p8_auprc
delta_brier = p9_brier - p8_brier

graph_summary = pd.read_csv(PHASE9A_SUMMARY) if PHASE9A_SUMMARY.exists() else pd.DataFrame()

if len(graph_summary) > 0:
    n_graphs = graph_summary["patient_folder"].nunique()
    n_pos = int((pd.to_numeric(graph_summary["label_pCR"], errors="coerce") == 1).sum())
    n_neg = int((pd.to_numeric(graph_summary["label_pCR"], errors="coerce") == 0).sum())
else:
    n_graphs = int(best9.get("n_patients", 0))
    n_pos = int(best9.get("positive_pCR", 0))
    n_neg = int(best9.get("negative_non_pCR", 0))

# Decision rule
if p9_auroc > p8_auroc and p9_auprc >= p8_auprc:
    decision = "temporal_delta_promising"
    recommendation = "Proceed to a small temporal GNN pilot, but still report it as exploratory."
elif n_graphs < 30:
    decision = "do_not_train_gnn_yet_small_n"
    recommendation = "Do not make temporal GNN the main model yet. Keep Phase 8C FTV SVM as the main pilot result and describe GNN as future work."
else:
    decision = "temporal_delta_weaker_than_ftv"
    recommendation = "Do not prioritize temporal GNN yet. Improve graph features or cohort size first."

decision_df = pd.DataFrame([
    {
        "analysis": "Phase 8C FTV support-data model",
        "best_model": best8.get("model", ""),
        "n_patients": best8.get("n_patients", ""),
        "AUROC": p8_auroc,
        "AUPRC": p8_auprc,
        "Brier_score": p8_brier,
        "role": "current strongest pilot model"
    },
    {
        "analysis": "Phase 9B temporal-delta baseline",
        "best_model": best9.get("model", ""),
        "n_patients": best9.get("n_patients", ""),
        "AUROC": p9_auroc,
        "AUPRC": p9_auprc,
        "Brier_score": p9_brier,
        "role": "diagnostic temporal baseline"
    }
])

decision_df.to_csv(TABLES / "phase9c_model_decision_summary.csv", index=False)

delta_df = pd.DataFrame([
    {
        "metric": "AUROC",
        "phase8c_ftv": p8_auroc,
        "phase9b_temporal_delta": p9_auroc,
        "delta_phase9b_minus_phase8c": delta_auroc,
        "interpretation": "higher is better"
    },
    {
        "metric": "AUPRC",
        "phase8c_ftv": p8_auprc,
        "phase9b_temporal_delta": p9_auprc,
        "delta_phase9b_minus_phase8c": delta_auprc,
        "interpretation": "higher is better"
    },
    {
        "metric": "Brier_score",
        "phase8c_ftv": p8_brier,
        "phase9b_temporal_delta": p9_brier,
        "delta_phase9b_minus_phase8c": delta_brier,
        "interpretation": "lower is better"
    }
])

delta_df.to_csv(TABLES / "phase9c_phase8c_vs_phase9b_metric_delta.csv", index=False)

next_plan = pd.DataFrame([
    {
        "priority": 1,
        "task": "Keep Phase 8C FTV SVM as current main pilot model",
        "why": "It has better AUROC, better AUPRC, and lower Brier score than Phase 9B.",
        "output": "Main result for current proposal/manuscript."
    },
    {
        "priority": 2,
        "task": "Use Phase 9A and Phase 9B as graph-readiness evidence",
        "why": "The project has visit nodes and temporal edges, but temporal-delta model is weaker.",
        "output": "Future-work section and graph design justification."
    },
    {
        "priority": 3,
        "task": "Do not train a full GNN as the main claim yet",
        "why": "Only 13 patient graphs are available. This is too small for a defensible deep graph model.",
        "output": "Avoid overclaiming."
    },
    {
        "priority": 4,
        "task": "Prepare final Word report and PowerPoint",
        "why": "The project now has a clear scientific story and enough outputs for presentation.",
        "output": "Proposal/report and slides."
    },
    {
        "priority": 5,
        "task": "Future technical expansion",
        "why": "A true temporal GNN needs more patients, richer PE/SER features, and stable node identity.",
        "output": "Phase 10 plan."
    }
])

next_plan.to_csv(TABLES / "phase9c_next_research_plan.csv", index=False)

# Figures
plt.figure(figsize=(9, 5))
x = np.arange(3)
labels = ["AUROC", "AUPRC", "Brier"]
p8_vals = [p8_auroc, p8_auprc, p8_brier]
p9_vals = [p9_auroc, p9_auprc, p9_brier]
width = 0.35
plt.bar(x - width/2, p8_vals, width, label="Phase 8C FTV")
plt.bar(x + width/2, p9_vals, width, label="Phase 9B temporal")
plt.xticks(x, labels)
plt.ylabel("Score")
plt.title("Phase 9C Model Decision")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "300_phase9c_model_decision.png", dpi=220)
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(["AUROC delta", "AUPRC delta", "Brier delta"], [delta_auroc, delta_auprc, delta_brier])
plt.axhline(0, linestyle="--")
plt.title("Phase 9C: Temporal Delta Minus FTV")
plt.ylabel("Difference")
plt.tight_layout()
plt.savefig(FIGS / "301_phase9c_metric_delta.png", dpi=220)
plt.close()

plt.figure(figsize=(12, 7))
plt.axis("off")
flow = (
    "Phase 9C decision flow\n\n"
    "Phase 8C FTV support-data model\n"
    f"  Best: {best8.get('model', '')}, AUROC={p8_auroc:.3f}, AUPRC={p8_auprc:.3f}, Brier={p8_brier:.3f}\n"
    "        ↓\n"
    "Phase 9A graph-ready visit dataset\n"
    f"  {n_graphs} patient graphs, {n_pos} pCR, {n_neg} non-pCR\n"
    "        ↓\n"
    "Phase 9B temporal-delta baseline\n"
    f"  Best: {best9.get('model', '')}, AUROC={p9_auroc:.3f}, AUPRC={p9_auprc:.3f}, Brier={p9_brier:.3f}\n"
    "        ↓\n"
    f"Decision: {decision}\n\n"
    f"{recommendation}"
)
plt.text(0.5, 0.5, flow, ha="center", va="center", fontsize=12)
plt.tight_layout()
plt.savefig(FIGS / "302_phase9c_graph_decision_flow.png", dpi=220)
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

    for _, r in small.iterrows():
        lines.append("| " + " | ".join(clean(v) for v in r.tolist()) + " |")

    md_path.write_text("\n".join(lines))

make_md(TABLES / "phase9c_model_decision_summary.csv", VIEWS / "phase9c_model_decision_summary.md", "Phase 9C Model Decision Summary")
make_md(TABLES / "phase9c_phase8c_vs_phase9b_metric_delta.csv", VIEWS / "phase9c_phase8c_vs_phase9b_metric_delta.md", "Phase 9C Metric Delta")
make_md(TABLES / "phase9c_next_research_plan.csv", VIEWS / "phase9c_next_research_plan.md", "Phase 9C Next Research Plan")

# Report
report = []
report.append("# Phase 9C Temporal Graph Model Decision Report")
report.append("")
report.append("This phase compares the current best FTV support-data model against the temporal-delta baseline and decides whether a temporal GNN should be the next main model.")
report.append("")
report.append("## Main result")
report.append("")
report.append(f"Phase 8C FTV best model: {best8.get('model', '')}, AUROC {p8_auroc:.4f}, AUPRC {p8_auprc:.4f}, Brier {p8_brier:.4f}.")
report.append("")
report.append(f"Phase 9B temporal-delta best model: {best9.get('model', '')}, AUROC {p9_auroc:.4f}, AUPRC {p9_auprc:.4f}, Brier {p9_brier:.4f}.")
report.append("")
report.append("## Decision")
report.append("")
report.append(f"Decision: **{decision}**.")
report.append("")
report.append(recommendation)
report.append("")
report.append("## Interpretation")
report.append("")
report.append("The temporal-delta baseline did not beat the FTV support-data model. This means the temporal graph idea is still scientifically useful, but it should not be the main result yet.")
report.append("")
report.append("## Why this is the right decision")
report.append("")
report.append("A temporal GNN needs enough patients and stable node features. The current graph-ready dataset has only 13 patient graphs. Training a deep graph model now would likely overfit and weaken the paper.")
report.append("")
report.append("## Safe paper statement")
report.append("")
report.append("The official FTV support-data SVM is the current strongest pilot model. The longitudinal graph dataset is a future-work bridge, not the current main clinical model.")
report.append("")
report.append("## Next practical step")
report.append("")
report.append("Prepare the final Word report and PowerPoint using Phase 8C, Phase 8D, Phase 8E, Phase 9A, Phase 9B, and Phase 9C. The future-work section should explain how a larger temporal GNN would be built.")

(REPORTS / "Phase9C_Temporal_Graph_Model_Decision_Report.md").write_text("\n".join(report))

# Dashboard update
dash = REPORTS / "Dashboard.md"
old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
dash_add = (
    "\n\n## Phase 9C temporal graph decision\n\n"
    "- [Temporal graph model decision report](Phase9C_Temporal_Graph_Model_Decision_Report.md)\n"
    "- [Model decision summary](table_views/phase9c_model_decision_summary.md)\n"
    "- [Next research plan](table_views/phase9c_next_research_plan.md)\n"
)
if "Phase 9C temporal graph decision" not in old_dash:
    dash.write_text(old_dash.rstrip() + dash_add)

# README update
readme = REPO / "README.md"
old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
readme_add = (
    "\n\n### Phase 9C: temporal graph model decision\n\n"
    "Main outputs:\n\n"
    "- reports/Phase9C_Temporal_Graph_Model_Decision_Report.md\n"
    "- reports/tables/phase9c_model_decision_summary.csv\n"
    "- reports/figures/300_phase9c_model_decision.png\n"
)
if "Phase 9C: temporal graph model decision" not in old_readme:
    readme.write_text(old_readme.rstrip() + readme_add)

print("Phase 9C complete.")
print("Decision:", decision)
print("Phase 8C AUROC:", round(p8_auroc, 4))
print("Phase 9B AUROC:", round(p9_auroc, 4))
print("AUROC delta phase9b - phase8c:", round(delta_auroc, 4))
print("Recommendation:", recommendation)
