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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance

REPO = Path.home() / "ISPY2-3DGCNN"
REPORTS = REPO / "reports"
TABLES = REPORTS / "tables"
FIGS = REPORTS / "figures"
VIEWS = REPORTS / "table_views"

for p in [REPORTS, TABLES, FIGS, VIEWS]:
    p.mkdir(parents=True, exist_ok=True)

FEATURE_PATH = TABLES / "phase6a_qc_pass_tumor_vascular_features.csv"
LABEL_PATH = TABLES / "phase4a_required_label_template.csv"
BASELINE_METRICS_PATH = TABLES / "phase5a_model_calibration_metrics.csv"

if not FEATURE_PATH.exists():
    raise SystemExit("Missing Phase 6A feature table. Run Phase 6A first.")

if not LABEL_PATH.exists():
    raise SystemExit("Missing pCR label table. Run Phase 4E first.")

features = pd.read_csv(FEATURE_PATH, dtype=str).fillna("")
labels = pd.read_csv(LABEL_PATH, dtype=str).fillna("")

if "patient_folder" not in features.columns:
    raise SystemExit("Phase 6A feature table has no patient_folder column.")

if "patient_folder" not in labels.columns or "pCR_label" not in labels.columns:
    raise SystemExit("Label table must contain patient_folder and pCR_label.")

labels = labels[["patient_folder", "pCR_label"]].copy()
labels["pCR_label"] = pd.to_numeric(labels["pCR_label"], errors="coerce")

merged = features.merge(labels, on="patient_folder", how="left")
merged = merged.dropna(subset=["pCR_label"]).copy()
merged["label"] = merged["pCR_label"].astype(int)

# Keep only successful feature rows
if "feature_status" in merged.columns:
    merged = merged[merged["feature_status"].eq("success")].copy()

# Numeric vascular/tumor feature selection
exclude_words = [
    "label", "pcr", "patient", "study", "uid", "status", "file",
    "path", "source", "error", "case_no", "frame"
]

feature_cols = []
for col in merged.columns:
    low = col.lower()
    if any(w in low for w in exclude_words):
        continue

    vals = pd.to_numeric(merged[col], errors="coerce")
    min_nonmissing = max(4, int(0.50 * len(merged))) if len(merged) else 4

    if vals.notna().sum() >= min_nonmissing and vals.nunique(dropna=True) > 1:
        feature_cols.append(col)

# Patient-level median aggregation
if len(feature_cols) == 0:
    summary = [
        "# Phase 6B Vascular Feature ML Summary",
        "",
        "No usable numeric vascular features were found.",
        "",
        "Run Phase 6A again or inspect the feature table.",
    ]
    (REPORTS / "Phase6B_Vascular_Feature_ML_Summary.md").write_text("\n".join(summary))
    print("Phase 6B stopped: no usable numeric features.")
    raise SystemExit(0)

model_df = merged[["patient_folder", "label"] + feature_cols].copy()

for col in feature_cols:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

model_df = model_df.groupby("patient_folder", as_index=False).agg(
    {**{"label": "max"}, **{c: "median" for c in feature_cols}}
)

model_df.to_csv(TABLES / "phase6b_vascular_modeling_dataset.csv", index=False)

y = model_df["label"].astype(int).values
X = model_df[feature_cols].copy()
X = X.fillna(X.median(numeric_only=True))

n_patients = len(model_df)
n_pos = int((y == 1).sum())
n_neg = int((y == 0).sum())
min_class = min(n_pos, n_neg)

# Always create label distribution figure
plt.figure(figsize=(8, 5))
pd.Series(y).map({0: "non-pCR", 1: "pCR"}).value_counts().plot(kind="bar")
plt.title("Phase 6B Vascular Cohort Label Distribution")
plt.ylabel("Patients")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(FIGS / "150_phase6b_label_distribution.png", dpi=220)
plt.close()

if n_patients < 6 or min_class < 2:
    summary = [
        "# Phase 6B Vascular Feature ML Summary",
        "",
        "Phase 6B created the vascular modeling dataset, but supervised ML was not run because there are not enough labeled patients in both classes.",
        "",
        "## Main counts",
        "",
        "- Patients in vascular modeling dataset: " + str(n_patients),
        "- pCR positive: " + str(n_pos),
        "- non-pCR negative: " + str(n_neg),
        "- Numeric vascular features: " + str(len(feature_cols)),
        "",
        "## Main output",
        "",
        "- reports/tables/phase6b_vascular_modeling_dataset.csv",
        "",
        "## Next step",
        "",
        "Run Phase 6A on more QC-passed cases, then rerun Phase 6B.",
    ]
    (REPORTS / "Phase6B_Vascular_Feature_ML_Summary.md").write_text("\n".join(summary))
    print("Phase 6B created dataset but did not run ML.")
    print("Patients:", n_patients)
    print("Positive:", n_pos)
    print("Negative:", n_neg)
    raise SystemExit(0)

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
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=min(3, max(1, n_patients // 4))),
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    ),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

xgb_status = "not_available"
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    xgb_status = "available"
except Exception:
    pass

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

metric_rows = []
prediction_rows = []
oof_probs = {}

for name, model in models.items():
    probs = np.zeros(n_patients)
    preds = np.zeros(n_patients, dtype=int)

    for train_idx, test_idx in cv.split(X, y):
        model.fit(X.iloc[train_idx], y[train_idx])
        p = model.predict_proba(X.iloc[test_idx])[:, 1]
        probs[test_idx] = p
        preds[test_idx] = (p >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()

    metric_rows.append({
        "model": name,
        "n_patients": n_patients,
        "positive_labels": n_pos,
        "negative_labels": n_neg,
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
        prediction_rows.append({
            "model": name,
            "patient_folder": pid,
            "true_label": int(true),
            "predicted_probability": float(prob),
            "predicted_label": int(pred),
        })

    oof_probs[name] = probs

metrics = pd.DataFrame(metric_rows).sort_values(["AUROC", "AUPRC", "Brier_score"], ascending=[False, False, True])
predictions = pd.DataFrame(prediction_rows)

metrics.to_csv(TABLES / "phase6b_vascular_model_performance.csv", index=False)
predictions.to_csv(TABLES / "phase6b_vascular_model_predictions.csv", index=False)

best_model_name = str(metrics.iloc[0]["model"])
best_model = models[best_model_name]
best_model.fit(X, y)

# Feature importance
importance_rows = []
try:
    perm = permutation_importance(
        best_model,
        X,
        y,
        scoring="roc_auc",
        n_repeats=30,
        random_state=42,
    )

    for i, col in enumerate(feature_cols):
        pos_vals = X.loc[y == 1, col]
        neg_vals = X.loc[y == 0, col]

        importance_rows.append({
            "feature": col,
            "permutation_importance_mean": float(perm.importances_mean[i]),
            "permutation_importance_std": float(perm.importances_std[i]),
            "mean_pCR_positive": float(pos_vals.mean()),
            "mean_non_pCR_negative": float(neg_vals.mean()),
            "mean_difference_positive_minus_negative": float(pos_vals.mean() - neg_vals.mean()),
        })
except Exception as e:
    for col in feature_cols:
        importance_rows.append({
            "feature": col,
            "permutation_importance_mean": np.nan,
            "permutation_importance_std": np.nan,
            "importance_error": str(e)[:200],
        })

importance = pd.DataFrame(importance_rows).sort_values("permutation_importance_mean", ascending=False)
importance.to_csv(TABLES / "phase6b_vascular_feature_importance.csv", index=False)

# Compare with Phase 5A
baseline_best_model = ""
baseline_auroc = np.nan
baseline_auprc = np.nan
baseline_brier = np.nan

if BASELINE_METRICS_PATH.exists():
    base = pd.read_csv(BASELINE_METRICS_PATH)
    if len(base) > 0:
        base = base.sort_values(["AUROC", "AUPRC", "Brier_score"], ascending=[False, False, True])
        baseline_best_model = str(base.iloc[0]["model"])
        baseline_auroc = float(base.iloc[0]["AUROC"])
        baseline_auprc = float(base.iloc[0]["AUPRC"])
        baseline_brier = float(base.iloc[0]["Brier_score"])

comparison = pd.DataFrame([{
    "baseline_phase5a_best_model": baseline_best_model,
    "baseline_phase5a_AUROC": baseline_auroc,
    "baseline_phase5a_AUPRC": baseline_auprc,
    "baseline_phase5a_Brier": baseline_brier,
    "vascular_phase6b_best_model": best_model_name,
    "vascular_phase6b_AUROC": float(metrics.iloc[0]["AUROC"]),
    "vascular_phase6b_AUPRC": float(metrics.iloc[0]["AUPRC"]),
    "vascular_phase6b_Brier": float(metrics.iloc[0]["Brier_score"]),
    "AUROC_difference_phase6b_minus_phase5a": float(metrics.iloc[0]["AUROC"]) - baseline_auroc if not np.isnan(baseline_auroc) else np.nan,
    "AUPRC_difference_phase6b_minus_phase5a": float(metrics.iloc[0]["AUPRC"]) - baseline_auprc if not np.isnan(baseline_auprc) else np.nan,
    "note": "Comparison is exploratory because Phase 6B uses only the vascular QC subset."
}])

comparison.to_csv(TABLES / "phase6b_vs_phase5a_model_comparison.csv", index=False)

# Figures
plt.figure(figsize=(9, 5))
metrics.sort_values("AUROC").plot(x="model", y="AUROC", kind="barh", legend=False)
plt.xlabel("AUROC")
plt.title("Phase 6B Vascular Feature Model AUROC")
plt.tight_layout()
plt.savefig(FIGS / "151_phase6b_model_auroc.png", dpi=220)
plt.close()

plt.figure(figsize=(9, 5))
metrics.sort_values("AUPRC").plot(x="model", y="AUPRC", kind="barh", legend=False)
plt.xlabel("AUPRC")
plt.title("Phase 6B Vascular Feature Model AUPRC")
plt.tight_layout()
plt.savefig(FIGS / "152_phase6b_model_auprc.png", dpi=220)
plt.close()

plt.figure(figsize=(10, 7))
top_imp = importance.head(15).sort_values("permutation_importance_mean")
if len(top_imp) > 0:
    plt.barh(top_imp["feature"], top_imp["permutation_importance_mean"])
    plt.xlabel("Permutation importance")
    plt.title("Phase 6B Vascular Feature Importance")
else:
    plt.text(0.5, 0.5, "No importance values", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "153_phase6b_vascular_feature_importance.png", dpi=220)
plt.close()

plt.figure(figsize=(8, 5))
if len(importance) > 0:
    top_feature = str(importance.iloc[0]["feature"])
    vals0 = X.loc[y == 0, top_feature]
    vals1 = X.loc[y == 1, top_feature]
    plt.boxplot([vals0, vals1], labels=["non-pCR", "pCR"])
    plt.ylabel(top_feature)
    plt.title("Top Vascular Feature by pCR Label")
else:
    plt.text(0.5, 0.5, "No top feature", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "154_phase6b_top_feature_by_label.png", dpi=220)
plt.close()

# Markdown helper
def make_md(csv_path, md_path, title, max_rows=50, max_cols=10):
    dfm = pd.read_csv(csv_path, dtype=str).fillna("")
    small = dfm.head(max_rows).iloc[:, :max_cols]

    def clean(x):
        x = str(x).replace("|", "\\|").replace("\n", " ")
        if len(x) > 90:
            x = x[:87] + "..."
        return x

    lines = []
    lines.append("# " + title)
    lines.append("")
    lines.append("Rows: " + str(len(dfm)))
    lines.append("Columns: " + str(len(dfm.columns)))
    lines.append("")
    lines.append("| " + " | ".join(clean(c) for c in small.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(small.columns)) + " |")

    for _, row in small.iterrows():
        lines.append("| " + " | ".join(clean(v) for v in row.tolist()) + " |")

    md_path.write_text("\n".join(lines))

make_md(TABLES / "phase6b_vascular_modeling_dataset.csv", VIEWS / "phase6b_vascular_modeling_dataset.md", "Phase 6B Vascular Modeling Dataset")
make_md(TABLES / "phase6b_vascular_model_performance.csv", VIEWS / "phase6b_vascular_model_performance.md", "Phase 6B Vascular Model Performance")
make_md(TABLES / "phase6b_vascular_feature_importance.csv", VIEWS / "phase6b_vascular_feature_importance.md", "Phase 6B Vascular Feature Importance")
make_md(TABLES / "phase6b_vs_phase5a_model_comparison.csv", VIEWS / "phase6b_vs_phase5a_model_comparison.md", "Phase 6B vs Phase 5A Model Comparison")

summary = []
summary.append("# Phase 6B Vascular Feature ML Summary")
summary.append("")
summary.append("This phase merges QC-passed tumor vascular features with pCR labels and reruns pilot ML.")
summary.append("")
summary.append("## Main outputs")
summary.append("")
summary.append("- reports/tables/phase6b_vascular_modeling_dataset.csv")
summary.append("- reports/tables/phase6b_vascular_model_performance.csv")
summary.append("- reports/tables/phase6b_vascular_model_predictions.csv")
summary.append("- reports/tables/phase6b_vascular_feature_importance.csv")
summary.append("- reports/tables/phase6b_vs_phase5a_model_comparison.csv")
summary.append("- reports/figures/150_phase6b_label_distribution.png")
summary.append("- reports/figures/151_phase6b_model_auroc.png")
summary.append("- reports/figures/152_phase6b_model_auprc.png")
summary.append("- reports/figures/153_phase6b_vascular_feature_importance.png")
summary.append("- reports/figures/154_phase6b_top_feature_by_label.png")
summary.append("")
summary.append("## Main counts")
summary.append("")
summary.append("- Labeled vascular patients: " + str(n_patients))
summary.append("- pCR positive: " + str(n_pos))
summary.append("- non-pCR negative: " + str(n_neg))
summary.append("- Numeric vascular features: " + str(len(feature_cols)))
summary.append("- CV folds: " + str(n_splits))
summary.append("- XGBoost available: " + xgb_status)
summary.append("")
summary.append("## Best vascular-feature model")
summary.append("")
summary.append("- Model: " + best_model_name)
summary.append("- AUROC: " + f"{float(metrics.iloc[0]['AUROC']):.4f}")
summary.append("- AUPRC: " + f"{float(metrics.iloc[0]['AUPRC']):.4f}")
summary.append("- Brier score: " + f"{float(metrics.iloc[0]['Brier_score']):.4f}")
summary.append("")
summary.append("## Comparison note")
summary.append("")
summary.append("The Phase 6B comparison with Phase 5A is exploratory because Phase 6B uses the vascular QC subset, not necessarily the same full cohort.")
summary.append("")
summary.append("## Next step")
summary.append("")
summary.append("Phase 6C should expand Phase 6A feature extraction beyond 12 cases, then rerun Phase 6B for a stronger evaluation.")

(REPORTS / "Phase6B_Vascular_Feature_ML_Summary.md").write_text("\n".join(summary))

# Dashboard update
dash = REPORTS / "Dashboard.md"
old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
add = "\n\n## Phase 6B table views\n\n- [Vascular model performance](table_views/phase6b_vascular_model_performance.md)\n- [Vascular feature importance](table_views/phase6b_vascular_feature_importance.md)\n- [Phase 6B vs Phase 5A comparison](table_views/phase6b_vs_phase5a_model_comparison.md)\n"
if "Phase 6B table views" not in old_dash:
    dash.write_text(old_dash.rstrip() + add)

# README update
readme = REPO / "README.md"
old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
add_readme = "\n\n### Phase 6B: vascular feature ML\n\nMain outputs:\n\n- reports/tables/phase6b_vascular_model_performance.csv\n- reports/tables/phase6b_vascular_feature_importance.csv\n- reports/Phase6B_Vascular_Feature_ML_Summary.md\n"
if "Phase 6B: vascular feature ML" not in old_readme:
    readme.write_text(old_readme.rstrip() + add_readme)

print("Phase 6B complete.")
print("Labeled vascular patients:", n_patients)
print("pCR positive:", n_pos)
print("non-pCR negative:", n_neg)
print("Numeric vascular features:", len(feature_cols))
print("Best vascular model:", best_model_name)
print("Best AUROC:", round(float(metrics.iloc[0]["AUROC"]), 4))
print("Best AUPRC:", round(float(metrics.iloc[0]["AUPRC"]), 4))
