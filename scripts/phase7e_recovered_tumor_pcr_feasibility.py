from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, LeaveOneOut
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

REPO = Path.home() / "ISPY2-3DGCNN"
REPORTS = REPO / "reports"
TABLES = REPORTS / "tables"
FIGS = REPORTS / "figures"
VIEWS = REPORTS / "table_views"

for p in [REPORTS, TABLES, FIGS, VIEWS]:
    p.mkdir(parents=True, exist_ok=True)

FEATURE_PATH = TABLES / "phase7d_recovered_tumor_vascular_features.csv"
LABEL_PATH = TABLES / "phase4a_required_label_template.csv"

if not FEATURE_PATH.exists():
    raise SystemExit("Missing reports/tables/phase7d_recovered_tumor_vascular_features.csv. Run Phase 7D first.")

if not LABEL_PATH.exists():
    raise SystemExit("Missing reports/tables/phase4a_required_label_template.csv. Run Phase 4E first.")

features = pd.read_csv(FEATURE_PATH, dtype=str).fillna("")
labels = pd.read_csv(LABEL_PATH, dtype=str).fillna("")

if "patient_folder" not in features.columns:
    raise SystemExit("Feature table has no patient_folder column.")

if "patient_folder" not in labels.columns or "pCR_label" not in labels.columns:
    raise SystemExit("Label table must contain patient_folder and pCR_label.")

labels = labels[["patient_folder", "pCR_label"]].copy()
labels["pCR_label"] = pd.to_numeric(labels["pCR_label"], errors="coerce")

merged = features.merge(labels, on="patient_folder", how="left")
merged["pCR_label_numeric"] = pd.to_numeric(merged["pCR_label"], errors="coerce")

if "feature_status" in merged.columns:
    merged = merged[merged["feature_status"].eq("success")].copy()

merged.to_csv(TABLES / "phase7e_recovered_tumor_label_merge.csv", index=False)

# Select numeric feature columns
exclude_words = [
    "case_no", "patient", "study", "uid", "status", "file", "path",
    "label", "pcr", "source", "error", "frame_index"
]

numeric_features = []
for col in merged.columns:
    low = col.lower()
    if any(w in low for w in exclude_words):
        continue
    vals = pd.to_numeric(merged[col], errors="coerce")
    if vals.notna().sum() >= 2 and vals.nunique(dropna=True) > 1:
        numeric_features.append(col)

# Patient-level aggregation
valid = merged.dropna(subset=["pCR_label_numeric"]).copy()

if len(valid) > 0 and len(numeric_features) > 0:
    for col in numeric_features:
        valid[col] = pd.to_numeric(valid[col], errors="coerce")

    agg_dict = {"pCR_label_numeric": "max"}
    for col in numeric_features:
        agg_dict[col] = "median"

    model_df = valid[["patient_folder", "pCR_label_numeric"] + numeric_features].groupby(
        "patient_folder", as_index=False
    ).agg(agg_dict)

    model_df = model_df.rename(columns={"pCR_label_numeric": "label"})
else:
    model_df = pd.DataFrame(columns=["patient_folder", "label"] + numeric_features)

model_df.to_csv(TABLES / "phase7e_recovered_tumor_modeling_dataset.csv", index=False)

n_rows = len(merged)
n_unique_feature_patients = merged["patient_folder"].nunique() if "patient_folder" in merged.columns else 0
n_labeled_records = len(valid)
n_labeled_patients = len(model_df)

if len(model_df) > 0:
    y = model_df["label"].astype(int).values
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
else:
    y = np.array([])
    n_pos = 0
    n_neg = 0

# Figure 210: label distribution
plt.figure(figsize=(8, 5))
if len(model_df) > 0:
    pd.Series(y).map({0: "non-pCR", 1: "pCR"}).value_counts().plot(kind="bar")
    plt.title("Phase 7E Recovered Tumor Label Distribution")
    plt.ylabel("Patients")
    plt.xticks(rotation=20)
else:
    plt.text(0.5, 0.5, "No labeled recovered tumor patients", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "210_phase7e_label_distribution.png", dpi=220)
plt.close()

# Figure 211: a top simple feature by label
top_feature = ""
if len(model_df) > 0 and len(numeric_features) > 0:
    # rank by absolute mean difference
    diffs = []
    for col in numeric_features:
        vals = pd.to_numeric(model_df[col], errors="coerce")
        if vals.notna().sum() < 2:
            continue
        pos = vals[y == 1]
        neg = vals[y == 0]
        if len(pos) > 0 and len(neg) > 0:
            diffs.append((col, abs(pos.mean() - neg.mean())))

    if diffs:
        top_feature = sorted(diffs, key=lambda x: x[1], reverse=True)[0][0]

plt.figure(figsize=(8, 5))
if top_feature:
    vals0 = pd.to_numeric(model_df.loc[model_df["label"] == 0, top_feature], errors="coerce")
    vals1 = pd.to_numeric(model_df.loc[model_df["label"] == 1, top_feature], errors="coerce")
    plt.boxplot([vals0.dropna(), vals1.dropna()], labels=["non-pCR", "pCR"])
    plt.ylabel(top_feature)
    plt.title("Phase 7E Top Recovered Tumor Feature by Label")
else:
    plt.text(0.5, 0.5, "No feature-label comparison available", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "211_phase7e_top_feature_by_label.png", dpi=220)
plt.close()

# Helper for markdown previews
def make_md(csv_path, md_path, title, max_rows=80, max_cols=12):
    if not csv_path.exists():
        return

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
    TABLES / "phase7e_recovered_tumor_label_merge.csv",
    VIEWS / "phase7e_recovered_tumor_label_merge.md",
    "Phase 7E Recovered Tumor Label Merge"
)

make_md(
    TABLES / "phase7e_recovered_tumor_modeling_dataset.csv",
    VIEWS / "phase7e_recovered_tumor_modeling_dataset.md",
    "Phase 7E Recovered Tumor Modeling Dataset"
)

# Decide whether ML is possible
ml_possible = n_labeled_patients >= 6 and n_pos >= 2 and n_neg >= 2 and len(numeric_features) >= 2

performance = pd.DataFrame()
predictions = pd.DataFrame()
importance = pd.DataFrame()

if ml_possible:
    X = model_df[numeric_features].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    y = model_df["label"].astype(int).values
    min_class = min(n_pos, n_neg)
    n_splits = min(5, min_class)

    models = {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)
        ),
        "SVM_RBF": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42
        )
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    perf_rows = []
    pred_rows = {}

    for model_name, model in models.items():
        probs = np.zeros(len(y), dtype=float)
        preds = np.zeros(len(y), dtype=int)

        for tr, te in cv.split(X, y):
            model.fit(X.iloc[tr], y[tr])
            p = model.predict_proba(X.iloc[te])[:, 1]
            probs[te] = p
            preds[te] = (p >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()

        perf_rows.append({
            "model": model_name,
            "n_patients": len(y),
            "positive_pCR": n_pos,
            "negative_non_pCR": n_neg,
            "n_features": len(numeric_features),
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
            "FN": int(fn)
        })

        for pid, true, prob, pred in zip(model_df["patient_folder"], y, probs, preds):
            pred_rows.setdefault("rows", []).append({
                "model": model_name,
                "patient_folder": pid,
                "true_label": int(true),
                "predicted_probability": float(prob),
                "predicted_label": int(pred)
            })

    performance = pd.DataFrame(perf_rows).sort_values(["AUROC", "AUPRC"], ascending=[False, False])
    predictions = pd.DataFrame(pred_rows.get("rows", []))

    performance.to_csv(TABLES / "phase7e_recovered_tumor_model_performance.csv", index=False)
    predictions.to_csv(TABLES / "phase7e_recovered_tumor_model_predictions.csv", index=False)

    best_name = str(performance.iloc[0]["model"])
    best_model = models[best_name]
    best_model.fit(X, y)

    try:
        perm = permutation_importance(
            best_model, X, y, scoring="roc_auc", n_repeats=30, random_state=42
        )

        rows = []
        for i, col in enumerate(numeric_features):
            rows.append({
                "feature": col,
                "permutation_importance_mean": float(perm.importances_mean[i]),
                "permutation_importance_std": float(perm.importances_std[i]),
                "mean_pCR": float(X.loc[y == 1, col].mean()),
                "mean_non_pCR": float(X.loc[y == 0, col].mean()),
                "mean_difference_pCR_minus_non_pCR": float(X.loc[y == 1, col].mean() - X.loc[y == 0, col].mean())
            })
        importance = pd.DataFrame(rows).sort_values("permutation_importance_mean", ascending=False)

    except Exception as e:
        rows = []
        for col in numeric_features:
            rows.append({
                "feature": col,
                "permutation_importance_mean": np.nan,
                "permutation_importance_std": np.nan,
                "importance_error": str(e)[:200]
            })
        importance = pd.DataFrame(rows)

    importance.to_csv(TABLES / "phase7e_recovered_tumor_feature_importance.csv", index=False)

    make_md(
        TABLES / "phase7e_recovered_tumor_model_performance.csv",
        VIEWS / "phase7e_recovered_tumor_model_performance.md",
        "Phase 7E Recovered Tumor Model Performance"
    )

    make_md(
        TABLES / "phase7e_recovered_tumor_feature_importance.csv",
        VIEWS / "phase7e_recovered_tumor_feature_importance.md",
        "Phase 7E Recovered Tumor Feature Importance"
    )

    # Figure 212 model performance
    plt.figure(figsize=(8, 5))
    performance.sort_values("AUROC").plot(x="model", y="AUROC", kind="barh", legend=False)
    plt.xlabel("AUROC")
    plt.title("Phase 7E Recovered Tumor Model AUROC")
    plt.tight_layout()
    plt.savefig(FIGS / "212_phase7e_model_performance.png", dpi=220)
    plt.close()

else:
    # Still create placeholder files so GitHub output is clean
    pd.DataFrame([{
        "reason": "ML not run",
        "labeled_patients": n_labeled_patients,
        "positive_pCR": n_pos,
        "negative_non_pCR": n_neg,
        "numeric_features": len(numeric_features),
        "rule": "Need at least 6 labeled patients, at least 2 pCR, at least 2 non-pCR, and at least 2 numeric features."
    }]).to_csv(TABLES / "phase7e_recovered_tumor_model_performance.csv", index=False)

    plt.figure(figsize=(8, 5))
    txt = (
        "Recovered tumor ML not run\n\n"
        f"Labeled patients: {n_labeled_patients}\n"
        f"pCR positive: {n_pos}\n"
        f"non-pCR negative: {n_neg}\n"
        f"numeric features: {len(numeric_features)}\n\n"
        "Need both classes and more patients."
    )
    plt.text(0.5, 0.5, txt, ha="center", va="center", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "212_phase7e_model_performance.png", dpi=220)
    plt.close()

    make_md(
        TABLES / "phase7e_recovered_tumor_model_performance.csv",
        VIEWS / "phase7e_recovered_tumor_model_performance.md",
        "Phase 7E Recovered Tumor Model Performance"
    )

# Report
report = []
report.append("# Phase 7E Recovered Tumor pCR Feasibility Report")
report.append("")
report.append("This phase merges recovered Phase 7D tumor/vascular features with pCR labels.")
report.append("")
report.append("## Main outputs")
report.append("")
report.append("- reports/tables/phase7e_recovered_tumor_label_merge.csv")
report.append("- reports/tables/phase7e_recovered_tumor_modeling_dataset.csv")
report.append("- reports/tables/phase7e_recovered_tumor_model_performance.csv")
report.append("- reports/table_views/phase7e_recovered_tumor_label_merge.md")
report.append("- reports/table_views/phase7e_recovered_tumor_modeling_dataset.md")
report.append("- reports/figures/210_phase7e_label_distribution.png")
report.append("- reports/figures/211_phase7e_top_feature_by_label.png")
report.append("- reports/figures/212_phase7e_model_performance.png")
report.append("")
report.append("## Main counts")
report.append("")
report.append("- Recovered feature rows: " + str(n_rows))
report.append("- Unique recovered feature patients: " + str(n_unique_feature_patients))
report.append("- Labeled recovered records: " + str(n_labeled_records))
report.append("- Labeled recovered patients: " + str(n_labeled_patients))
report.append("- pCR positive patients: " + str(n_pos))
report.append("- non-pCR negative patients: " + str(n_neg))
report.append("- Numeric recovered tumor features: " + str(len(numeric_features)))
report.append("")
report.append("## Feasibility result")
report.append("")
if ml_possible:
    report.append("A small feasibility model was run. This is not final clinical evidence, because the recovered tumor cohort is still small.")
    report.append("")
    report.append("- Best model: " + str(performance.iloc[0]["model"]))
    report.append("- AUROC: " + f"{float(performance.iloc[0]['AUROC']):.4f}")
    report.append("- AUPRC: " + f"{float(performance.iloc[0]['AUPRC']):.4f}")
else:
    report.append("Supervised ML was not run because the recovered tumor cohort is still too small or lacks both pCR classes.")
report.append("")
report.append("## Next step")
report.append("")
if ml_possible:
    report.append("Run Phase 7F to expand Phase 7C/7D to more cases and compare the recovered-mask model with the earlier baseline model.")
else:
    report.append("Run Phase 7F to expand fractional threshold recovery beyond the current 8 recovered cases. The goal is to obtain more labeled pCR and non-pCR patients before model training.")

(REPORTS / "Phase7E_Recovered_Tumor_pCR_Feasibility_Report.md").write_text("\n".join(report))

# Dashboard update
dash = REPORTS / "Dashboard.md"
old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
dash_add = "\n\n## Phase 7E recovered tumor pCR feasibility\n\n- [Recovered tumor pCR feasibility report](Phase7E_Recovered_Tumor_pCR_Feasibility_Report.md)\n- [Recovered tumor label merge](table_views/phase7e_recovered_tumor_label_merge.md)\n- [Recovered tumor model performance](table_views/phase7e_recovered_tumor_model_performance.md)\n"
if "Phase 7E recovered tumor pCR feasibility" not in old_dash:
    dash.write_text(old_dash.rstrip() + dash_add)

# README update
readme = REPO / "README.md"
old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
readme_add = "\n\n### Phase 7E: recovered tumor pCR feasibility\n\nMain outputs:\n\n- reports/Phase7E_Recovered_Tumor_pCR_Feasibility_Report.md\n- reports/tables/phase7e_recovered_tumor_label_merge.csv\n- reports/tables/phase7e_recovered_tumor_model_performance.csv\n"
if "Phase 7E: recovered tumor pCR feasibility" not in old_readme:
    readme.write_text(old_readme.rstrip() + readme_add)

print("Phase 7E complete.")
print("Recovered feature rows:", n_rows)
print("Unique recovered feature patients:", n_unique_feature_patients)
print("Labeled recovered patients:", n_labeled_patients)
print("pCR positive patients:", n_pos)
print("non-pCR negative patients:", n_neg)
print("Numeric recovered tumor features:", len(numeric_features))
print("ML possible:", "yes" if ml_possible else "no")
