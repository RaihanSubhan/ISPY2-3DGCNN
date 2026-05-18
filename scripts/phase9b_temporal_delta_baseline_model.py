from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance

REPO = Path.home() / "ISPY2-3DGCNN"
REPORTS = REPO / "reports"
TABLES = REPORTS / "tables"
FIGS = REPORTS / "figures"
VIEWS = REPORTS / "table_views"

for p in [REPORTS, TABLES, FIGS, VIEWS]:
    p.mkdir(parents=True, exist_ok=True)

DELTA_PATH = TABLES / "phase9a_temporal_delta_features.csv"
NODES_PATH = TABLES / "phase9a_longitudinal_visit_nodes.csv"
PHASE8C_PERF = TABLES / "phase8c_ftv_model_performance.csv"

if not NODES_PATH.exists():
    raise SystemExit("Missing phase9a_longitudinal_visit_nodes.csv. Run Phase 9A first.")

nodes = pd.read_csv(NODES_PATH, dtype=str).fillna("")

# -------------------------------------------------------
# 1. Load delta table, or rebuild it from visit nodes
# -------------------------------------------------------
if DELTA_PATH.exists():
    try:
        deltas = pd.read_csv(DELTA_PATH, dtype=str).fillna("")
    except Exception:
        deltas = pd.DataFrame()
else:
    deltas = pd.DataFrame()

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def diff(a, b):
    if pd.notna(a) and pd.notna(b):
        return b - a
    return np.nan

def pct(a, b):
    if pd.notna(a) and pd.notna(b) and abs(a) > 1e-8:
        return 100.0 * (b - a) / a
    return np.nan

if deltas.empty or "patient_folder" not in deltas.columns:
    rows = []
    nodes_work = nodes.copy()
    for c in ["visit_index", "label_pCR", "ftv_volume", "sphericity", "longest_diameter", "bpe_5slice_mean"]:
        if c in nodes_work.columns:
            nodes_work[c] = pd.to_numeric(nodes_work[c], errors="coerce")

    for patient, sub in nodes_work.groupby("patient_folder"):
        sub = sub.sort_values("visit_index").reset_index(drop=True)
        if len(sub) < 2:
            continue

        first = sub.iloc[0]
        last = sub.iloc[-1]

        vol_first = first.get("ftv_volume", np.nan)
        vol_last = last.get("ftv_volume", np.nan)
        sph_first = first.get("sphericity", np.nan)
        sph_last = last.get("sphericity", np.nan)
        ld_first = first.get("longest_diameter", np.nan)
        ld_last = last.get("longest_diameter", np.nan)
        bpe_first = first.get("bpe_5slice_mean", np.nan)
        bpe_last = last.get("bpe_5slice_mean", np.nan)

        rows.append({
            "patient_folder": patient,
            "label_pCR": int(first["label_pCR"]),
            "first_visit": first["visit_label"],
            "last_visit": last["visit_label"],
            "n_visits": len(sub),
            "ftv_volume_first": vol_first,
            "ftv_volume_last": vol_last,
            "ftv_volume_delta_first_to_last": diff(vol_first, vol_last),
            "ftv_volume_pct_delta_first_to_last": pct(vol_first, vol_last),
            "sphericity_first": sph_first,
            "sphericity_last": sph_last,
            "sphericity_delta_first_to_last": diff(sph_first, sph_last),
            "sphericity_pct_delta_first_to_last": pct(sph_first, sph_last),
            "longest_diameter_first": ld_first,
            "longest_diameter_last": ld_last,
            "longest_diameter_delta_first_to_last": diff(ld_first, ld_last),
            "longest_diameter_pct_delta_first_to_last": pct(ld_first, ld_last),
            "bpe_first": bpe_first,
            "bpe_last": bpe_last,
            "bpe_delta_first_to_last": diff(bpe_first, bpe_last),
            "bpe_pct_delta_first_to_last": pct(bpe_first, bpe_last),
        })

    deltas = pd.DataFrame(rows)
    deltas.to_csv(DELTA_PATH, index=False)

# -------------------------------------------------------
# 2. Build modeling dataset
# -------------------------------------------------------
if "label_pCR" not in deltas.columns:
    raise SystemExit("Temporal delta table has no label_pCR column.")

deltas["label"] = pd.to_numeric(deltas["label_pCR"], errors="coerce")
deltas = deltas.dropna(subset=["label"]).copy()
deltas["label"] = deltas["label"].astype(int)

exclude_words = [
    "patient",
    "label",
    "visit",
    "source",
    "target",
    "node",
    "edge",
]

feature_cols = []
for col in deltas.columns:
    low = col.lower()
    if any(w in low for w in exclude_words):
        continue

    vals = pd.to_numeric(deltas[col], errors="coerce")
    if vals.notna().sum() >= 5 and vals.nunique(dropna=True) > 1:
        feature_cols.append(col)

model_df = deltas[["patient_folder", "label"] + feature_cols].copy()
for col in feature_cols:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

model_df.to_csv(TABLES / "phase9b_temporal_delta_modeling_dataset.csv", index=False)

n_patients = len(model_df)
y = model_df["label"].astype(int).values if n_patients else np.array([])
n_pos = int((y == 1).sum()) if n_patients else 0
n_neg = int((y == 0).sum()) if n_patients else 0
min_class = min(n_pos, n_neg) if n_patients else 0

# -------------------------------------------------------
# 3. Label distribution figure
# -------------------------------------------------------
plt.figure(figsize=(8, 5))
if n_patients:
    pd.Series(y).map({0: "non-pCR", 1: "pCR"}).value_counts().plot(kind="bar")
    plt.title("Phase 9B Temporal-Delta Label Distribution")
    plt.ylabel("Patients")
    plt.xticks(rotation=20)
else:
    plt.text(0.5, 0.5, "No labeled temporal-delta patients", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "290_phase9b_label_distribution.png", dpi=220)
plt.close()

# -------------------------------------------------------
# 4. Run models if possible
# -------------------------------------------------------
ml_possible = n_patients >= 10 and n_pos >= 3 and n_neg >= 3 and len(feature_cols) >= 2

metrics = pd.DataFrame()
predictions = pd.DataFrame()
importance = pd.DataFrame()

if ml_possible:
    X = model_df[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))

    n_splits = min(5, min_class)

    models = {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
        ),
        "SVM_RBF": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=42,
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    metric_rows = []
    pred_rows = []
    probs_by_model = {}

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for model_name, model in models.items():
        probs = np.zeros(n_patients, dtype=float)
        preds = np.zeros(n_patients, dtype=int)

        for train_idx, test_idx in cv.split(X, y):
            model.fit(X.iloc[train_idx], y[train_idx])
            p = model.predict_proba(X.iloc[test_idx])[:, 1]
            probs[test_idx] = p
            preds[test_idx] = (p >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()

        metric_rows.append({
            "model": model_name,
            "n_patients": n_patients,
            "positive_pCR": n_pos,
            "negative_non_pCR": n_neg,
            "n_features": len(feature_cols),
            "n_splits": n_splits,
            "AUROC": float(roc_auc_score(y, probs)),
            "AUPRC": float(average_precision_score(y, probs)),
            "Brier_score": float(brier_score_loss(y, probs)),
            "accuracy": float(accuracy_score(y, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
            "F1": float(f1_score(y, preds, zero_division=0)),
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
            "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
        })

        for pid, true, prob, pred in zip(model_df["patient_folder"], y, probs, preds):
            pred_rows.append({
                "model": model_name,
                "patient_folder": pid,
                "true_label": int(true),
                "predicted_probability": float(prob),
                "predicted_label": int(pred),
            })

        probs_by_model[model_name] = probs

    metrics = pd.DataFrame(metric_rows).sort_values(["AUROC", "AUPRC", "Brier_score"], ascending=[False, False, True])
    predictions = pd.DataFrame(pred_rows)

    metrics.to_csv(TABLES / "phase9b_temporal_delta_model_performance.csv", index=False)
    predictions.to_csv(TABLES / "phase9b_temporal_delta_model_predictions.csv", index=False)

    # Feature importance for best model
    best_model_name = str(metrics.iloc[0]["model"])
    best_model = models[best_model_name]
    best_model.fit(X, y)

    try:
        perm = permutation_importance(
            best_model,
            X,
            y,
            scoring="roc_auc",
            n_repeats=30,
            random_state=42,
        )

        rows = []
        for i, col in enumerate(feature_cols):
            rows.append({
                "feature": col,
                "permutation_importance_mean": float(perm.importances_mean[i]),
                "permutation_importance_std": float(perm.importances_std[i]),
                "mean_pCR": float(X.loc[y == 1, col].mean()),
                "mean_non_pCR": float(X.loc[y == 0, col].mean()),
                "mean_difference_pCR_minus_non_pCR": float(
                    X.loc[y == 1, col].mean() - X.loc[y == 0, col].mean()
                ),
            })

        importance = pd.DataFrame(rows).sort_values("permutation_importance_mean", ascending=False)

    except Exception as e:
        importance = pd.DataFrame([{"importance_error": str(e)[:300]}])

    importance.to_csv(TABLES / "phase9b_temporal_delta_feature_importance.csv", index=False)

    # Figure 291 AUROC
    plt.figure(figsize=(9, 5))
    metrics.sort_values("AUROC").plot(x="model", y="AUROC", kind="barh", legend=False)
    plt.xlabel("AUROC")
    plt.title("Phase 9B Temporal-Delta Model AUROC")
    plt.tight_layout()
    plt.savefig(FIGS / "291_phase9b_model_auroc.png", dpi=220)
    plt.close()

    # Figure 292 AUPRC
    plt.figure(figsize=(9, 5))
    metrics.sort_values("AUPRC").plot(x="model", y="AUPRC", kind="barh", legend=False)
    plt.xlabel("AUPRC")
    plt.title("Phase 9B Temporal-Delta Model AUPRC")
    plt.tight_layout()
    plt.savefig(FIGS / "292_phase9b_model_auprc.png", dpi=220)
    plt.close()

    # Figure 293 importance
    plt.figure(figsize=(10, 7))
    if "permutation_importance_mean" in importance.columns:
        top = importance.head(12).sort_values("permutation_importance_mean")
        plt.barh(top["feature"], top["permutation_importance_mean"])
        plt.title("Phase 9B Temporal-Delta Feature Importance")
        plt.xlabel("Permutation importance")
    else:
        plt.text(0.5, 0.5, "No feature importance available", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "293_phase9b_feature_importance.png", dpi=220)
    plt.close()

else:
    pd.DataFrame([{
        "reason": "ML not run",
        "n_patients": n_patients,
        "positive_pCR": n_pos,
        "negative_non_pCR": n_neg,
        "n_features": len(feature_cols),
        "rule": "Need at least 10 patients, 3 pCR, 3 non-pCR, and 2 numeric features."
    }]).to_csv(TABLES / "phase9b_temporal_delta_model_performance.csv", index=False)

    plt.figure(figsize=(8, 5))
    txt = (
        "Phase 9B ML not run\n\n"
        f"Patients: {n_patients}\n"
        f"pCR: {n_pos}\n"
        f"non-pCR: {n_neg}\n"
        f"features: {len(feature_cols)}"
    )
    plt.text(0.5, 0.5, txt, ha="center", va="center", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "291_phase9b_model_auroc.png", dpi=220)
    plt.close()

# -------------------------------------------------------
# 5. Compare with Phase 8C
# -------------------------------------------------------
if PHASE8C_PERF.exists():
    phase8c = pd.read_csv(PHASE8C_PERF)
else:
    phase8c = pd.DataFrame()

def best_from_perf(perf):
    if perf.empty or "AUROC" not in perf.columns:
        return {}
    d = perf.copy()
    d["AUROC"] = pd.to_numeric(d["AUROC"], errors="coerce")
    d["AUPRC"] = pd.to_numeric(d["AUPRC"], errors="coerce")
    d["Brier_score"] = pd.to_numeric(d["Brier_score"], errors="coerce")
    d = d.sort_values(["AUROC", "AUPRC", "Brier_score"], ascending=[False, False, True])
    return d.iloc[0].to_dict()

best8 = best_from_perf(phase8c)
best9 = best_from_perf(metrics)

comparison_rows = []

if best8:
    comparison_rows.append({
        "analysis": "Phase 8C FTV support-data model",
        "best_model": best8.get("model", ""),
        "n_patients": best8.get("n_patients", ""),
        "AUROC": best8.get("AUROC", ""),
        "AUPRC": best8.get("AUPRC", ""),
        "Brier_score": best8.get("Brier_score", ""),
    })

if best9:
    comparison_rows.append({
        "analysis": "Phase 9B temporal-delta baseline",
        "best_model": best9.get("model", ""),
        "n_patients": best9.get("n_patients", ""),
        "AUROC": best9.get("AUROC", ""),
        "AUPRC": best9.get("AUPRC", ""),
        "Brier_score": best9.get("Brier_score", ""),
    })

comparison = pd.DataFrame(comparison_rows)
comparison.to_csv(TABLES / "phase9b_vs_phase8c_comparison.csv", index=False)

# Figure 294 comparison
plt.figure(figsize=(9, 5))
if len(comparison) >= 2:
    plot = comparison.copy()
    plot["AUROC"] = pd.to_numeric(plot["AUROC"], errors="coerce")
    plot["AUPRC"] = pd.to_numeric(plot["AUPRC"], errors="coerce")
    x = np.arange(len(plot))
    width = 0.35
    plt.bar(x - width/2, plot["AUROC"], width, label="AUROC")
    plt.bar(x + width/2, plot["AUPRC"], width, label="AUPRC")
    plt.xticks(x, plot["analysis"], rotation=20, ha="right")
    plt.ylabel("Score")
    plt.title("Phase 9B vs Phase 8C Comparison")
    plt.legend()
else:
    plt.text(0.5, 0.5, "Comparison not available", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "294_phase9b_vs_phase8c_comparison.png", dpi=220)
plt.close()

# -------------------------------------------------------
# 6. Markdown previews
# -------------------------------------------------------
def make_md(csv_path, md_path, title, max_rows=80, max_cols=12):
    if not Path(csv_path).exists():
        return
    data = pd.read_csv(csv_path, dtype=str).fillna("")
    small = data.head(max_rows).iloc[:, :max_cols]

    def clean(x):
        x = str(x).replace("|", "\\|").replace("\n", " ")
        if len(x) > 90:
            x = x[:87] + "..."
        return x

    lines = []
    lines.append("# " + title)
    lines.append("")
    lines.append("Rows: " + str(len(data)))
    lines.append("Columns: " + str(len(data.columns)))
    lines.append("")
    lines.append("| " + " | ".join(clean(c) for c in small.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(small.columns)) + " |")

    for _, r in small.iterrows():
        lines.append("| " + " | ".join(clean(v) for v in r.tolist()) + " |")

    md_path.write_text("\n".join(lines))

make_md(TABLES / "phase9b_temporal_delta_modeling_dataset.csv", VIEWS / "phase9b_temporal_delta_modeling_dataset.md", "Phase 9B Temporal-Delta Modeling Dataset")
make_md(TABLES / "phase9b_temporal_delta_model_performance.csv", VIEWS / "phase9b_temporal_delta_model_performance.md", "Phase 9B Temporal-Delta Model Performance")
make_md(TABLES / "phase9b_temporal_delta_model_predictions.csv", VIEWS / "phase9b_temporal_delta_model_predictions.md", "Phase 9B Temporal-Delta Model Predictions")
make_md(TABLES / "phase9b_temporal_delta_feature_importance.csv", VIEWS / "phase9b_temporal_delta_feature_importance.md", "Phase 9B Temporal-Delta Feature Importance")
make_md(TABLES / "phase9b_vs_phase8c_comparison.csv", VIEWS / "phase9b_vs_phase8c_comparison.md", "Phase 9B vs Phase 8C Comparison")

# -------------------------------------------------------
# 7. Report
# -------------------------------------------------------
report = []
report.append("# Phase 9B Temporal-Delta Baseline Model Report")
report.append("")
report.append("This phase trains a simple temporal-delta baseline model using longitudinal FTV graph features.")
report.append("")
report.append("## Main outputs")
report.append("")
report.append("- reports/tables/phase9b_temporal_delta_modeling_dataset.csv")
report.append("- reports/tables/phase9b_temporal_delta_model_performance.csv")
report.append("- reports/tables/phase9b_temporal_delta_model_predictions.csv")
report.append("- reports/tables/phase9b_temporal_delta_feature_importance.csv")
report.append("- reports/tables/phase9b_vs_phase8c_comparison.csv")
report.append("- reports/figures/290_phase9b_label_distribution.png")
report.append("- reports/figures/291_phase9b_model_auroc.png")
report.append("- reports/figures/292_phase9b_model_auprc.png")
report.append("- reports/figures/293_phase9b_feature_importance.png")
report.append("- reports/figures/294_phase9b_vs_phase8c_comparison.png")
report.append("")
report.append("## Main counts")
report.append("")
report.append("- Patients: " + str(n_patients))
report.append("- pCR positive: " + str(n_pos))
report.append("- non-pCR negative: " + str(n_neg))
report.append("- Temporal-delta features: " + str(len(feature_cols)))
report.append("")
report.append("## Modeling result")
report.append("")
if best9:
    report.append("ML was run successfully.")
    report.append("- Best model: " + str(best9.get("model", "")))
    report.append("- AUROC: " + f"{float(best9.get('AUROC', np.nan)):.4f}")
    report.append("- AUPRC: " + f"{float(best9.get('AUPRC', np.nan)):.4f}")
    report.append("- Brier score: " + f"{float(best9.get('Brier_score', np.nan)):.4f}")
else:
    report.append("ML was not run because the dataset did not meet minimum sample and class requirements.")
report.append("")
report.append("## Interpretation")
report.append("")
report.append("This is the first temporal baseline for the future graph-learning direction. It tests whether simple first-to-last visit changes carry pCR signal before building a temporal GNN.")
report.append("")
report.append("## Next step")
report.append("")
report.append("Phase 9C should compare Phase 8C and Phase 9B, then decide whether a temporal GNN is justified or whether the classical FTV SVM remains the strongest pilot model.")

(REPORTS / "Phase9B_Temporal_Delta_Baseline_Report.md").write_text("\n".join(report))

# Dashboard update
dash = REPORTS / "Dashboard.md"
old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
dash_add = (
    "\n\n## Phase 9B temporal-delta baseline\n\n"
    "- [Temporal-delta baseline report](Phase9B_Temporal_Delta_Baseline_Report.md)\n"
    "- [Temporal-delta model performance](table_views/phase9b_temporal_delta_model_performance.md)\n"
    "- [Phase 9B vs Phase 8C comparison](table_views/phase9b_vs_phase8c_comparison.md)\n"
)
if "Phase 9B temporal-delta baseline" not in old_dash:
    dash.write_text(old_dash.rstrip() + dash_add)

# README update
readme = REPO / "README.md"
old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
readme_add = (
    "\n\n### Phase 9B: temporal-delta baseline model\n\n"
    "Main outputs:\n\n"
    "- reports/Phase9B_Temporal_Delta_Baseline_Report.md\n"
    "- reports/tables/phase9b_temporal_delta_model_performance.csv\n"
    "- reports/tables/phase9b_vs_phase8c_comparison.csv\n"
)
if "Phase 9B: temporal-delta baseline model" not in old_readme:
    readme.write_text(old_readme.rstrip() + readme_add)

print("Phase 9B complete.")
print("Patients:", n_patients)
print("pCR positive:", n_pos)
print("non-pCR negative:", n_neg)
print("Temporal-delta features:", len(feature_cols))
if best9:
    print("Best temporal model:", best9.get("model", ""))
    print("Best AUROC:", round(float(best9.get("AUROC", np.nan)), 4))
    print("Best AUPRC:", round(float(best9.get("AUPRC", np.nan)), 4))
else:
    print("ML was not run.")
