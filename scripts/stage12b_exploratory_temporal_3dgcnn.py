from pathlib import Path
import math
import random
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path.home() / "ISPY2-3DGCNN"
REPORTS = REPO / "reports"
TABLES = REPORTS / "tables"
FIGS = REPORTS / "figures"
VIEWS = REPORTS / "table_views"

for p in [REPORTS, TABLES, FIGS, VIEWS]:
    p.mkdir(parents=True, exist_ok=True)

NODES_PATH = TABLES / "phase9a_longitudinal_visit_nodes.csv"
EDGES_PATH = TABLES / "phase9a_temporal_edges.csv"
PHASE8C_PERF = TABLES / "phase8c_ftv_model_performance.csv"
PHASE9B_PERF = TABLES / "phase9b_temporal_delta_model_performance.csv"

if not NODES_PATH.exists():
    raise SystemExit("Missing reports/tables/phase9a_longitudinal_visit_nodes.csv. Run Phase 9A first.")

if not EDGES_PATH.exists():
    raise SystemExit("Missing reports/tables/phase9a_temporal_edges.csv. Run Phase 9A first.")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

nodes = pd.read_csv(NODES_PATH, dtype=str).fillna("")
edges = pd.read_csv(EDGES_PATH, dtype=str).fillna("")

# ---------------------------------------------------------
# Feature columns for patient-visit graph nodes
# ---------------------------------------------------------
node_feature_cols = [
    "visit_index",
    "ftv_volume",
    "sphericity",
    "longest_diameter",
    "bpe_5slice_mean",
    "has_volume",
    "has_sphericity",
    "has_longest_diameter",
    "has_bpe",
]

needed_cols = ["node_id", "patient_folder", "visit_label", "visit_index", "label_pCR"]
for col in needed_cols:
    if col not in nodes.columns:
        raise SystemExit(f"Missing column in phase9a_longitudinal_visit_nodes.csv: {col}")

for col in node_feature_cols:
    if col not in nodes.columns:
        nodes[col] = ""

for col in node_feature_cols:
    nodes[col] = pd.to_numeric(nodes[col], errors="coerce")

nodes["label_pCR"] = pd.to_numeric(nodes["label_pCR"], errors="coerce").astype(int)
nodes["visit_index"] = pd.to_numeric(nodes["visit_index"], errors="coerce").fillna(0)

# Keep patients with labels and at least two visits.
patient_rows = []
for patient, sub in nodes.groupby("patient_folder"):
    sub = sub.sort_values("visit_index").copy()
    label_vals = sub["label_pCR"].dropna().unique()
    if len(label_vals) == 0:
        continue

    patient_rows.append({
        "patient_folder": patient,
        "label": int(label_vals[0]),
        "n_nodes": len(sub),
        "visit_labels": ";".join(sub["visit_label"].astype(str).tolist()),
    })

patient_manifest = pd.DataFrame(patient_rows)
patient_manifest.to_csv(TABLES / "stage12b_patient_graph_manifest.csv", index=False)

if len(patient_manifest) < 6:
    raise SystemExit("Not enough patient graphs for exploratory 3DGCNN.")

# ---------------------------------------------------------
# Build graph tensors
# ---------------------------------------------------------
all_node_features = nodes[node_feature_cols].copy()
feature_medians = all_node_features.median(numeric_only=True)
all_node_features = all_node_features.fillna(feature_medians)

scaler = StandardScaler()
scaler.fit(all_node_features.values.astype(float))

graphs = []
labels = []
patients = []

for patient, sub in nodes.groupby("patient_folder"):
    if patient not in set(patient_manifest["patient_folder"]):
        continue

    sub = sub.sort_values("visit_index").reset_index(drop=True)
    label = int(sub["label_pCR"].iloc[0])

    x = sub[node_feature_cols].copy()
    x = x.fillna(feature_medians)
    x = scaler.transform(x.values.astype(float))
    x = torch.tensor(x, dtype=torch.float32)

    # Build temporal edges within patient graph.
    n = x.shape[0]
    edge_pairs = []

    # temporal edges both directions
    for i in range(n - 1):
        edge_pairs.append([i, i + 1])
        edge_pairs.append([i + 1, i])

    # self-loops
    for i in range(n):
        edge_pairs.append([i, i])

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

    graphs.append({
        "patient_folder": patient,
        "x": x,
        "edge_index": edge_index,
        "label": label,
        "n_nodes": n,
    })
    labels.append(label)
    patients.append(patient)

labels_np = np.array(labels, dtype=int)
n_pos = int((labels_np == 1).sum())
n_neg = int((labels_np == 0).sum())
min_class = min(n_pos, n_neg)

# ---------------------------------------------------------
# Simple GCN layer without requiring torch_geometric
# This is a small graph convolution:
# H' = D^-1 A H W
# ---------------------------------------------------------
class SimpleGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        n = x.shape[0]
        adj = torch.zeros((n, n), dtype=x.dtype, device=x.device)
        src = edge_index[0]
        dst = edge_index[1]
        adj[dst, src] = 1.0

        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        agg = adj @ x / deg
        return self.lin(agg)


class Temporal3DGCNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=16):
        super().__init__()
        self.gcn1 = SimpleGraphConv(in_dim, hidden_dim)
        self.gcn2 = SimpleGraphConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward_graph(self, x, edge_index):
        h = self.gcn1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.gcn2(h, edge_index)
        h = F.relu(h)

        # graph pooling: mean over visit nodes
        hg = h.mean(dim=0)
        logit = self.classifier(hg).squeeze()
        return logit


def train_one_model(train_idx, test_idx, epochs=250, lr=0.01, weight_decay=1e-3):
    model = Temporal3DGCNN(in_dim=len(node_feature_cols), hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_labels = labels_np[train_idx]
    pos = max(int((train_labels == 1).sum()), 1)
    neg = max(int((train_labels == 0).sum()), 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        losses = []

        for i in train_idx:
            g = graphs[i]
            logit = model.forward_graph(g["x"], g["edge_index"])
            y = torch.tensor(float(g["label"]), dtype=torch.float32)
            loss = criterion(logit.view(1), y.view(1))
            losses.append(loss)

        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        optimizer.step()

    model.eval()
    probs = []
    with torch.no_grad():
        for i in test_idx:
            g = graphs[i]
            logit = model.forward_graph(g["x"], g["edge_index"])
            prob = torch.sigmoid(logit).item()
            probs.append(prob)

    return model, probs


# ---------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------
if min_class < 2:
    ml_possible = False
else:
    ml_possible = True

prediction_rows = []
performance_rows = []
feature_importance_rows = []

if ml_possible:
    n_splits = min(5, min_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    probs = np.zeros(len(graphs), dtype=float)

    for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(labels_np)), labels_np), start=1):
        model, fold_probs = train_one_model(train_idx, test_idx)
        for idx, prob in zip(test_idx, fold_probs):
            probs[idx] = prob
            prediction_rows.append({
                "fold": fold,
                "patient_folder": patients[idx],
                "true_label": int(labels_np[idx]),
                "predicted_probability": float(prob),
                "predicted_label": int(prob >= 0.5),
            })

    preds = (probs >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels_np, preds, labels=[0, 1]).ravel()

    performance_rows.append({
        "model": "Exploratory_Temporal_3DGCNN",
        "n_patients": len(graphs),
        "positive_pCR": n_pos,
        "negative_non_pCR": n_neg,
        "n_node_features": len(node_feature_cols),
        "n_splits": n_splits,
        "AUROC": float(roc_auc_score(labels_np, probs)),
        "AUPRC": float(average_precision_score(labels_np, probs)),
        "Brier_score": float(brier_score_loss(labels_np, probs)),
        "accuracy": float(accuracy_score(labels_np, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_np, preds)),
        "F1": float(f1_score(labels_np, preds, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    })

    # Simple permutation-style feature importance on graph node features
    base_auroc = roc_auc_score(labels_np, probs)
    rng = np.random.default_rng(SEED)

    for j, col in enumerate(node_feature_cols):
        perm_probs_all = []

        for rep in range(20):
            perm_graphs = []
            shuffled_values = []

            # Collect all node feature values for this feature
            for g in graphs:
                shuffled_values.extend(g["x"][:, j].numpy().tolist())
            shuffled_values = np.array(shuffled_values)
            rng.shuffle(shuffled_values)

            cursor = 0
            for g in graphs:
                x_perm = g["x"].clone()
                n_nodes = x_perm.shape[0]
                x_perm[:, j] = torch.tensor(shuffled_values[cursor:cursor+n_nodes], dtype=torch.float32)
                cursor += n_nodes
                perm_graphs.append({
                    "patient_folder": g["patient_folder"],
                    "x": x_perm,
                    "edge_index": g["edge_index"],
                    "label": g["label"],
                    "n_nodes": g["n_nodes"],
                })

            # train/evaluate again is expensive; for tiny data, do one fresh CV
            perm_probs = np.zeros(len(graphs), dtype=float)
            for train_idx, test_idx in cv.split(np.zeros(len(labels_np)), labels_np):
                old_graphs = graphs
                graphs_backup = globals()["graphs"]
                globals()["graphs"] = perm_graphs
                try:
                    _, fold_probs = train_one_model(train_idx, test_idx, epochs=150)
                finally:
                    globals()["graphs"] = graphs_backup

                for idx, prob in zip(test_idx, fold_probs):
                    perm_probs[idx] = prob

            try:
                perm_auroc = roc_auc_score(labels_np, perm_probs)
                perm_probs_all.append(perm_auroc)
            except Exception:
                pass

        if len(perm_probs_all) > 0:
            importance = base_auroc - float(np.mean(perm_probs_all))
            std = float(np.std(perm_probs_all))
        else:
            importance = np.nan
            std = np.nan

        feature_importance_rows.append({
            "feature": col,
            "importance_auroc_drop": importance,
            "permuted_auroc_std": std,
        })

else:
    performance_rows.append({
        "model": "Exploratory_Temporal_3DGCNN",
        "reason": "ML not run",
        "n_patients": len(graphs),
        "positive_pCR": n_pos,
        "negative_non_pCR": n_neg,
        "rule": "Need at least two patients in each class.",
    })

performance = pd.DataFrame(performance_rows)
predictions = pd.DataFrame(prediction_rows)
importance = pd.DataFrame(feature_importance_rows)

performance.to_csv(TABLES / "stage12b_3dgcnn_model_performance.csv", index=False)
predictions.to_csv(TABLES / "stage12b_3dgcnn_oof_predictions.csv", index=False)
importance.to_csv(TABLES / "stage12b_3dgcnn_feature_importance.csv", index=False)

# ---------------------------------------------------------
# Compare with Phase 8C and Phase 9B
# ---------------------------------------------------------
def best_perf(path, label):
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    if "AUROC" not in df.columns:
        return {}
    d = df.copy()
    for c in ["AUROC", "AUPRC", "Brier_score"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["AUROC"])
    if d.empty:
        return {}
    d = d.sort_values(["AUROC", "AUPRC", "Brier_score"], ascending=[False, False, True])
    r = d.iloc[0].to_dict()
    return {
        "analysis": label,
        "best_model": r.get("model", ""),
        "n_patients": r.get("n_patients", ""),
        "AUROC": r.get("AUROC", ""),
        "AUPRC": r.get("AUPRC", ""),
        "Brier_score": r.get("Brier_score", ""),
    }

rows = []
b8 = best_perf(PHASE8C_PERF, "Phase 8C FTV SVM baseline")
b9 = best_perf(PHASE9B_PERF, "Phase 9B temporal-delta baseline")
b12 = best_perf(TABLES / "stage12b_3dgcnn_model_performance.csv", "Stage 12B exploratory temporal 3DGCNN")

for r in [b8, b9, b12]:
    if r:
        rows.append(r)

comparison = pd.DataFrame(rows)
comparison.to_csv(TABLES / "stage12b_3dgcnn_vs_existing_models.csv", index=False)

# ---------------------------------------------------------
# Figures
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
pd.Series(labels_np).map({0: "non-pCR", 1: "pCR"}).value_counts().plot(kind="bar")
plt.title("Stage 12B Graph Label Distribution")
plt.ylabel("Patients")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(FIGS / "510_stage12b_graph_label_distribution.png", dpi=220)
plt.close()

plt.figure(figsize=(9, 5))
if not comparison.empty:
    comp = comparison.copy()
    comp["AUROC"] = pd.to_numeric(comp["AUROC"], errors="coerce")
    comp.sort_values("AUROC").plot(x="analysis", y="AUROC", kind="barh", legend=False)
    plt.xlabel("AUROC")
    plt.title("Stage 12B 3DGCNN vs Existing Models")
else:
    plt.text(0.5, 0.5, "No comparison data", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "511_stage12b_model_comparison_auroc.png", dpi=220)
plt.close()

plt.figure(figsize=(9, 5))
if not predictions.empty:
    plt.scatter(predictions["true_label"], predictions["predicted_probability"])
    plt.xticks([0, 1], ["non-pCR", "pCR"])
    plt.ylabel("Predicted probability")
    plt.title("Stage 12B Exploratory 3DGCNN Predictions")
else:
    plt.text(0.5, 0.5, "No predictions available", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "512_stage12b_prediction_scatter.png", dpi=220)
plt.close()

plt.figure(figsize=(10, 6))
if not importance.empty and "importance_auroc_drop" in importance.columns:
    imp = importance.copy()
    imp["importance_auroc_drop"] = pd.to_numeric(imp["importance_auroc_drop"], errors="coerce")
    imp = imp.sort_values("importance_auroc_drop", ascending=False)
    plt.barh(imp["feature"], imp["importance_auroc_drop"])
    plt.xlabel("AUROC drop after permutation")
    plt.title("Stage 12B Node Feature Importance")
else:
    plt.text(0.5, 0.5, "No feature importance available", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "513_stage12b_feature_importance.png", dpi=220)
plt.close()

plt.figure(figsize=(11, 6))
plt.axis("off")
text = (
    "Stage 12B exploratory temporal 3DGCNN\n\n"
    "Graph design:\n"
    "Patient = one graph\n"
    "Visit = one node\n"
    "Edges = temporal order T0 → T1 → T2 → T3\n\n"
    "Node features:\n"
    "FTV volume, sphericity, longest diameter, BPE, visit index\n\n"
    "Important:\n"
    "This is a prototype, not the main clinical model.\n"
    "Main pilot result remains Phase 8C FTV SVM_RBF."
)
plt.text(0.5, 0.5, text, ha="center", va="center", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "514_stage12b_3dgcnn_schema.png", dpi=220)
plt.close()

# ---------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------
def make_md(csv_path, md_path, title, max_rows=80, max_cols=12):
    if not Path(csv_path).exists():
        return
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

    for _, row in small.iterrows():
        lines.append("| " + " | ".join(clean(v) for v in row.tolist()) + " |")

    md_path.write_text("\n".join(lines))

make_md(TABLES / "stage12b_patient_graph_manifest.csv", VIEWS / "stage12b_patient_graph_manifest.md", "Stage 12B Patient Graph Manifest")
make_md(TABLES / "stage12b_3dgcnn_model_performance.csv", VIEWS / "stage12b_3dgcnn_model_performance.md", "Stage 12B 3DGCNN Model Performance")
make_md(TABLES / "stage12b_3dgcnn_oof_predictions.csv", VIEWS / "stage12b_3dgcnn_oof_predictions.md", "Stage 12B 3DGCNN OOF Predictions")
make_md(TABLES / "stage12b_3dgcnn_feature_importance.csv", VIEWS / "stage12b_3dgcnn_feature_importance.md", "Stage 12B 3DGCNN Feature Importance")
make_md(TABLES / "stage12b_3dgcnn_vs_existing_models.csv", VIEWS / "stage12b_3dgcnn_vs_existing_models.md", "Stage 12B 3DGCNN vs Existing Models")

# ---------------------------------------------------------
# Report
# ---------------------------------------------------------
if ml_possible and not performance.empty and "AUROC" in performance.columns:
    best = performance.iloc[0]
    result_text = (
        f"The exploratory temporal 3DGCNN ran on {len(graphs)} patient graphs. "
        f"AUROC = {float(best['AUROC']):.4f}, AUPRC = {float(best['AUPRC']):.4f}, "
        f"Brier score = {float(best['Brier_score']):.4f}."
    )
else:
    result_text = "The exploratory temporal 3DGCNN could not run due to sample/class limitations."

report = []
report.append("# Stage 12B Exploratory Temporal 3DGCNN Report")
report.append("")
report.append("This stage creates a real exploratory temporal graph convolutional model using the Phase 9A graph-ready FTV dataset.")
report.append("")
report.append("## Why this was created")
report.append("")
report.append("The attached 3D MRI papers motivate patient-specific 3D modeling and validated segmentation. This code tests a lightweight 3DGCNN-style graph model for breast MRI visits, while keeping the Phase 8C FTV SVM as the current main pilot result.")
report.append("")
report.append("## Graph design")
report.append("")
report.append("- One patient = one graph")
report.append("- One visit = one node")
report.append("- Temporal edges connect consecutive visits")
report.append("- Node features include FTV volume, sphericity, longest diameter, BPE, visit index, and feature availability flags")
report.append("")
report.append("## Main counts")
report.append("")
report.append("- Patient graphs: " + str(len(graphs)))
report.append("- pCR positive graphs: " + str(n_pos))
report.append("- non-pCR graphs: " + str(n_neg))
report.append("- Node features: " + str(len(node_feature_cols)))
report.append("")
report.append("## Modeling result")
report.append("")
report.append(result_text)
report.append("")
report.append("## Important interpretation")
report.append("")
report.append("This 3DGCNN is exploratory. It should not replace the Phase 8C FTV SVM as the main model unless it clearly outperforms it on a larger cohort.")
report.append("")
report.append("## Main outputs")
report.append("")
report.append("- reports/tables/stage12b_3dgcnn_model_performance.csv")
report.append("- reports/tables/stage12b_3dgcnn_oof_predictions.csv")
report.append("- reports/tables/stage12b_3dgcnn_vs_existing_models.csv")
report.append("- reports/figures/510_stage12b_graph_label_distribution.png")
report.append("- reports/figures/511_stage12b_model_comparison_auroc.png")
report.append("- reports/figures/514_stage12b_3dgcnn_schema.png")
report.append("")
report.append("## Next step")
report.append("")
report.append("If the 3DGCNN underperforms the FTV SVM, keep it as a method-prototype figure. If it improves results, use it as an exploratory future-work experiment, not clinical validation.")

(REPORTS / "Stage12B_Exploratory_Temporal_3DGCNN_Report.md").write_text("\n".join(report))

# Dashboard / README updates
dash = REPORTS / "Dashboard.md"
old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
dash_add = (
    "\n\n## Stage 12B exploratory temporal 3DGCNN\n\n"
    "- [Exploratory temporal 3DGCNN report](Stage12B_Exploratory_Temporal_3DGCNN_Report.md)\n"
    "- [3DGCNN model performance](table_views/stage12b_3dgcnn_model_performance.md)\n"
    "- [3DGCNN vs existing models](table_views/stage12b_3dgcnn_vs_existing_models.md)\n"
)
if "Stage 12B exploratory temporal 3DGCNN" not in old_dash:
    dash.write_text(old_dash.rstrip() + dash_add)

readme = REPO / "README.md"
old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
readme_add = (
    "\n\n### Stage 12B: exploratory temporal 3DGCNN\n\n"
    "Main outputs:\n\n"
    "- reports/Stage12B_Exploratory_Temporal_3DGCNN_Report.md\n"
    "- reports/tables/stage12b_3dgcnn_model_performance.csv\n"
    "- reports/figures/514_stage12b_3dgcnn_schema.png\n"
)
if "Stage 12B: exploratory temporal 3DGCNN" not in old_readme:
    readme.write_text(old_readme.rstrip() + readme_add)

print("Stage 12B complete.")
print("Patient graphs:", len(graphs))
print("pCR positive:", n_pos)
print("non-pCR:", n_neg)
print("ML possible:", "yes" if ml_possible else "no")
if ml_possible:
    print(performance.to_string(index=False))
