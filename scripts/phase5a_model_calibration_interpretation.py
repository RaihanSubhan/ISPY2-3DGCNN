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
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

REPO = Path.home() / "ISPY2-3DGCNN"
REPORTS = REPO / "reports"
TABLES = REPORTS / "tables"
FIGS = REPORTS / "figures"
VIEWS = REPORTS / "table_views"

for p in [TABLES, FIGS, VIEWS]:
    p.mkdir(parents=True, exist_ok=True)

DATA_PATH = TABLES / "phase4b_modeling_dataset.csv"

if not DATA_PATH.exists():
    raise SystemExit("Missing reports/tables/phase4b_modeling_dataset.csv. Run Phase 4B first.")

df = pd.read_csv(DATA_PATH, dtype=str).fillna("")

if "label" not in df.columns:
    raise SystemExit("phase4b_modeling_dataset.csv does not contain a label column.")

y = pd.to_numeric(df["label"], errors="coerce")
valid_y = y.notna()
df = df.loc[valid_y].copy()
y = y.loc[valid_y].astype(int).values

exclude_words = [
    "label",
    "source",
    "patient",
    "folder",
    "uid",
    "status",
    "tier",
    "study",
    "path",
    "file",
]

feature_cols = []
for col in df.columns:
    low = col.lower()
    if any(w in low for w in exclude_words):
        continue
    vals = pd.to_numeric(df[col], errors="coerce")
    if vals.notna().sum() >= 5 and vals.nunique(dropna=True) > 1:
        feature_cols.append(col)

if len(feature_cols) < 2:
    raise SystemExit("Not enough numeric biomarker features for Phase 5A.")

X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median(numeric_only=True))

n_patients = len(y)
n_pos = int((y == 1).sum())
n_neg = int((y == 0).sum())
min_class = min(n_pos, n_neg)

if min_class < 2:
    raise SystemExit("Need at least 2 patients in each class for cross-validation.")

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
        KNeighborsClassifier(n_neighbors=min(5, max(1, n_patients // 6))),
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        max_depth=None,
    ),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

xgb_status = "not_available"
try:
    from xgboost import XGBClassifier

    models["XGBoost"] = XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    xgb_status = "available"
except Exception:
    xgb_status = "not_available"

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

metric_rows = []
prediction_rows = []
oof_prob_by_model = {}

for model_name, model in models.items():
    probs = np.zeros(n_patients, dtype=float)
    preds = np.zeros(n_patients, dtype=int)

    for train_idx, test_idx in cv.split(X, y):
        model.fit(X.iloc[train_idx], y[train_idx])

        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X.iloc[test_idx])[:, 1]
        else:
            p = model.decision_function(X.iloc[test_idx])
            p = (p - p.min()) / (p.max() - p.min() + 1e-9)

        probs[test_idx] = p
        preds[test_idx] = (p >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()

    try:
        prob_true, prob_pred = calibration_curve(y, probs, n_bins=min(8, n_patients // 4), strategy="quantile")
        calibration_mae = float(np.mean(np.abs(prob_true - prob_pred)))
    except Exception:
        calibration_mae = np.nan

    row = {
        "model": model_name,
        "n_patients": n_patients,
        "positive_labels": n_pos,
        "negative_labels": n_neg,
        "n_splits": n_splits,
        "AUROC": float(roc_auc_score(y, probs)),
        "AUPRC": float(average_precision_score(y, probs)),
        "Brier_score": float(brier_score_loss(y, probs)),
        "calibration_MAE": calibration_mae,
        "accuracy": float(accuracy_score(y, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "F1": float(f1_score(y, preds, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }
    metric_rows.append(row)

    for i in range(n_patients):
        prediction_rows.append({
            "model": model_name,
            "patient_folder": df.iloc[i].get("patient_folder", ""),
            "true_label": int(y[i]),
            "predicted_probability": float(probs[i]),
            "predicted_label": int(preds[i]),
        })

    oof_prob_by_model[model_name] = probs

metrics = pd.DataFrame(metric_rows)
metrics = metrics.sort_values(["AUROC", "AUPRC", "Brier_score"], ascending=[False, False, True])
predictions = pd.DataFrame(prediction_rows)

metrics.to_csv(TABLES / "phase5a_model_calibration_metrics.csv", index=False)
predictions.to_csv(TABLES / "phase5a_oof_predictions.csv", index=False)

best_model_name = str(metrics.iloc[0]["model"])
best_probs = oof_prob_by_model[best_model_name]
best_preds = (best_probs >= 0.5).astype(int)

# Fit best model on full data for permutation importance
best_model = models[best_model_name]
best_model.fit(X, y)

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
            "mean_pCR_negative": float(neg_vals.mean()),
            "mean_difference_positive_minus_negative": float(pos_vals.mean() - neg_vals.mean()),
        })

except Exception as e:
    for col in feature_cols:
        pos_vals = X.loc[y == 1, col]
        neg_vals = X.loc[y == 0, col]
        importance_rows.append({
            "feature": col,
            "permutation_importance_mean": np.nan,
            "permutation_importance_std": np.nan,
            "mean_pCR_positive": float(pos_vals.mean()),
            "mean_pCR_negative": float(neg_vals.mean()),
            "mean_difference_positive_minus_negative": float(pos_vals.mean() - neg_vals.mean()),
            "importance_error": str(e)[:200],
        })

importance = pd.DataFrame(importance_rows)
importance = importance.sort_values("permutation_importance_mean", ascending=False)
importance.to_csv(TABLES / "phase5a_feature_importance.csv", index=False)

recommendation = pd.DataFrame([{
    "recommended_model": best_model_name,
    "selection_rule": "highest AUROC, then highest AUPRC, then lowest Brier score",
    "AUROC": metrics.iloc[0]["AUROC"],
    "AUPRC": metrics.iloc[0]["AUPRC"],
    "Brier_score": metrics.iloc[0]["Brier_score"],
    "why": "This is a pilot model. It gives the best cross-validated discrimination among the tested models, but it must be validated on a larger cohort.",
    "xgboost_available": xgb_status,
}])
recommendation.to_csv(TABLES / "phase5a_model_recommendation.csv", index=False)

# Figure 110 ROC curves
plt.figure(figsize=(8, 6))
for model_name in metrics["model"]:
    probs = oof_prob_by_model[model_name]
    fpr, tpr, _ = roc_curve(y, probs)
    auc_val = roc_auc_score(y, probs)
    plt.plot(fpr, tpr, label=f"{model_name} AUROC={auc_val:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Phase 5A ROC Curves")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIGS / "110_phase5a_roc_curves.png", dpi=220)
plt.close()

# Figure 111 PR curves
plt.figure(figsize=(8, 6))
for model_name in metrics["model"]:
    probs = oof_prob_by_model[model_name]
    precision, recall, _ = precision_recall_curve(y, probs)
    ap = average_precision_score(y, probs)
    plt.plot(recall, precision, label=f"{model_name} AUPRC={ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Phase 5A Precision-Recall Curves")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIGS / "111_phase5a_pr_curves.png", dpi=220)
plt.close()

# Figure 112 calibration curve
plt.figure(figsize=(8, 6))
for model_name in metrics["model"].head(4):
    probs = oof_prob_by_model[model_name]
    try:
        prob_true, prob_pred = calibration_curve(y, probs, n_bins=min(8, n_patients // 4), strategy="quantile")
        plt.plot(prob_pred, prob_true, marker="o", label=model_name)
    except Exception:
        pass
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed pCR fraction")
plt.title("Phase 5A Calibration Curve")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIGS / "112_phase5a_calibration_curve.png", dpi=220)
plt.close()

# Figure 113 feature importance
plt.figure(figsize=(10, 7))
top_imp = importance.head(15).sort_values("permutation_importance_mean")
if len(top_imp) > 0:
    plt.barh(top_imp["feature"], top_imp["permutation_importance_mean"])
    plt.xlabel("Permutation importance, AUROC drop")
    plt.title(f"Phase 5A Feature Importance: {best_model_name}")
else:
    plt.text(0.5, 0.5, "No feature importance available", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "113_phase5a_feature_importance.png", dpi=220)
plt.close()

# Figure 114 confusion matrix
cm = confusion_matrix(y, best_preds, labels=[0, 1])
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title(f"Phase 5A Confusion Matrix: {best_model_name}")
plt.xticks([0, 1], ["Pred 0", "Pred 1"])
plt.yticks([0, 1], ["True 0", "True 1"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
plt.colorbar()
plt.tight_layout()
plt.savefig(FIGS / "114_phase5a_confusion_matrix.png", dpi=220)
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

make_md(TABLES / "phase5a_model_calibration_metrics.csv", VIEWS / "phase5a_model_calibration_metrics.md", "Phase 5A Model Calibration Metrics")
make_md(TABLES / "phase5a_feature_importance.csv", VIEWS / "phase5a_feature_importance.md", "Phase 5A Feature Importance")
make_md(TABLES / "phase5a_model_recommendation.csv", VIEWS / "phase5a_model_recommendation.md", "Phase 5A Model Recommendation")

summary = []
summary.append("# Phase 5A Model Calibration and Interpretation Summary")
summary.append("")
summary.append("This phase evaluates baseline pCR prediction models using cross-validated probabilities, calibration metrics, and feature importance.")
summary.append("")
summary.append("## Main outputs")
summary.append("")
summary.append("- reports/tables/phase5a_model_calibration_metrics.csv")
summary.append("- reports/tables/phase5a_oof_predictions.csv")
summary.append("- reports/tables/phase5a_feature_importance.csv")
summary.append("- reports/tables/phase5a_model_recommendation.csv")
summary.append("- reports/figures/110_phase5a_roc_curves.png")
summary.append("- reports/figures/111_phase5a_pr_curves.png")
summary.append("- reports/figures/112_phase5a_calibration_curve.png")
summary.append("- reports/figures/113_phase5a_feature_importance.png")
summary.append("- reports/figures/114_phase5a_confusion_matrix.png")
summary.append("")
summary.append("## Main counts")
summary.append("")
summary.append("- Labeled patients: " + str(n_patients))
summary.append("- Positive pCR labels: " + str(n_pos))
summary.append("- Negative non-pCR labels: " + str(n_neg))
summary.append("- Numeric biomarker features: " + str(len(feature_cols)))
summary.append("- CV folds: " + str(n_splits))
summary.append("- XGBoost available: " + xgb_status)
summary.append("")
summary.append("## Recommended pilot model")
summary.append("")
summary.append("- Model: " + best_model_name)
summary.append("- AUROC: " + f"{float(metrics.iloc[0]['AUROC']):.4f}")
summary.append("- AUPRC: " + f"{float(metrics.iloc[0]['AUPRC']):.4f}")
summary.append("- Brier score: " + f"{float(metrics.iloc[0]['Brier_score']):.4f}")
summary.append("")
summary.append("## Interpretation")
summary.append("")
summary.append("This is still a pilot model because only 33 labeled patients are available. The current result can support feasibility and method development, but it should not be claimed as final clinical performance.")
summary.append("")
summary.append("## Next step")
summary.append("")
summary.append("Phase 5B should create a final manuscript-ready results report with the best model, limitations, feature interpretation, and next validation plan.")

(REPORTS / "Phase5A_Model_Calibration_Interpretation_Summary.md").write_text("\n".join(summary))

# Dashboard update
dash = REPORTS / "Dashboard.md"
old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
dash_add = "\n\n## Phase 5A table views\n\n- [Model calibration metrics](table_views/phase5a_model_calibration_metrics.md)\n- [Feature importance](table_views/phase5a_feature_importance.md)\n- [Model recommendation](table_views/phase5a_model_recommendation.md)\n"
if "Phase 5A table views" not in old_dash:
    dash.write_text(old_dash.rstrip() + dash_add)

# README update
readme = REPO / "README.md"
old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
readme_add = "\n\n### Phase 5A: model calibration and interpretation\n\nMain outputs:\n\n- reports/tables/phase5a_model_calibration_metrics.csv\n- reports/tables/phase5a_feature_importance.csv\n- reports/Phase5A_Model_Calibration_Interpretation_Summary.md\n"
if "Phase 5A: model calibration and interpretation" not in old_readme:
    readme.write_text(old_readme.rstrip() + readme_add)

print("Phase 5A complete.")
print("Labeled patients:", n_patients)
print("Positive labels:", n_pos)
print("Negative labels:", n_neg)
print("Features:", len(feature_cols))
print("Best pilot model:", best_model_name)
print("XGBoost available:", xgb_status)
