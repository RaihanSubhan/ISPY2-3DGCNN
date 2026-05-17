from pathlib import Path
import re
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

MANIFEST = TABLES / "phase8b_tumor_source_manifest.csv"
LABELS = TABLES / "phase4a_required_label_template.csv"
BASELINE = TABLES / "phase5a_model_calibration_metrics.csv"

if not MANIFEST.exists():
    raise SystemExit("Missing reports/tables/phase8b_tumor_source_manifest.csv. Run Phase 8B first.")

if not LABELS.exists():
    raise SystemExit("Missing reports/tables/phase4a_required_label_template.csv. Run Phase 4E first.")

manifest = pd.read_csv(MANIFEST, dtype=str).fillna("")
labels = pd.read_csv(LABELS, dtype=str).fillna("")

if "patient_folder" not in labels.columns or "pCR_label" not in labels.columns:
    raise SystemExit("Label table must contain patient_folder and pCR_label.")

label_patients = set(labels["patient_folder"].astype(str))

def norm_patient_id(x):
    s = str(x).strip().upper()
    s = s.replace(".0", "")
    s = s.replace("_", "-")

    if s.startswith("ISPY2-"):
        return s

    if s.startswith("ACRIN-6698-"):
        return s

    m = re.search(r"ISPY2[- ]?(\d+)", s)
    if m:
        return "ISPY2-" + m.group(1)

    m = re.search(r"ACRIN[- ]?6698[- ]?(\d+)", s)
    if m:
        return "ACRIN-6698-" + m.group(1)

    m = re.search(r"(\d{6})", s)
    if m:
        n = m.group(1)
        if "ISPY2-" + n in label_patients:
            return "ISPY2-" + n
        if "ACRIN-6698-" + n in label_patients:
            return "ACRIN-6698-" + n
        return "ISPY2-" + n

    return s

def read_source_table(path):
    path = Path(path)
    if not path.exists():
        return None

    try:
        if path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(path, dtype=str).fillna("")
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, dtype=str).fillna("")
        if path.suffix.lower() == ".tsv":
            return pd.read_csv(path, sep="\t", dtype=str).fillna("")
        if path.suffix.lower() == ".txt":
            return pd.read_csv(path, sep=None, engine="python", dtype=str).fillna("")
    except Exception:
        return None

    return None

def is_ftv_source(row):
    text = " ".join([
        str(row.get("file_path", "")),
        str(row.get("source_class", "")),
        str(row.get("columns_or_keys_preview", "")),
        str(row.get("first_row_or_doc_preview", "")),
    ]).lower()

    return (
        "ftv" in text or
        "volume_tum" in text or
        "functional tumor" in text or
        "ftv_pch" in text or
        "sphericity_t0" in text
    )

def find_patient_column(df):
    preferred = [
        "CLINICAL-TRIAL-SUBJECT-ID",
        "Clinical Trial Subject ID",
        "Patient ID",
        "patient_folder",
        "PatientID",
        "Subject ID",
    ]

    for c in preferred:
        if c in df.columns:
            return c

    best = None
    best_overlap = -1

    for c in df.columns:
        vals = df[c].astype(str).map(norm_patient_id)
        overlap = len(set(vals).intersection(label_patients))
        low = c.lower()
        if "patient" in low or "subject" in low or "id" == low:
            overlap += 2
        if overlap > best_overlap:
            best_overlap = overlap
            best = c

    return best

# -------------------------------------------------------------------
# 1. Select best FTV / multi-feature source
# -------------------------------------------------------------------
ftv_rows = manifest[manifest.apply(is_ftv_source, axis=1)].copy()

source_rows = []
for _, row in ftv_rows.iterrows():
    path = row.get("file_path", "")
    df = read_source_table(path)

    if df is None or df.empty:
        source_rows.append({
            "file_path": path,
            "read_status": "failed_or_empty",
            "rows": 0,
            "columns": 0,
            "patient_column": "",
            "patient_overlap_with_labels": 0,
            "numeric_feature_count": 0,
            "source_score": -1,
        })
        continue

    patient_col = find_patient_column(df)
    if patient_col is None:
        overlap = 0
    else:
        overlap = len(set(df[patient_col].astype(str).map(norm_patient_id)).intersection(label_patients))

    numeric_count = 0
    for col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= 5 and vals.nunique(dropna=True) > 1:
            numeric_count += 1

    score = overlap * 10 + numeric_count + min(len(df), 1000) / 1000.0

    source_rows.append({
        "file_path": path,
        "read_status": "read_ok",
        "rows": len(df),
        "columns": len(df.columns),
        "patient_column": patient_col if patient_col else "",
        "patient_overlap_with_labels": overlap,
        "numeric_feature_count": numeric_count,
        "source_score": score,
    })

source_eval = pd.DataFrame(source_rows).sort_values("source_score", ascending=False)
source_eval.to_csv(TABLES / "phase8c_ftv_source_selection.csv", index=False)

if source_eval.empty or float(source_eval.iloc[0]["source_score"]) < 0:
    raise SystemExit("No usable FTV / multi-feature source was found.")

best_path = source_eval.iloc[0]["file_path"]
best_patient_col = source_eval.iloc[0]["patient_column"]

source_df = read_source_table(best_path)
if source_df is None or source_df.empty:
    raise SystemExit("Selected source could not be read.")

source_df = source_df.copy()
source_df["patient_folder"] = source_df[best_patient_col].astype(str).map(norm_patient_id)

# -------------------------------------------------------------------
# 2. Build feature table
# -------------------------------------------------------------------
exclude_words = [
    "patient", "subject", "id", "label", "source",
    "date", "name", "sex", "birth", "ethnic",
]

feature_cols = []
for col in source_df.columns:
    low = col.lower()

    if col == "patient_folder":
        continue

    if any(w in low for w in exclude_words):
        continue

    vals = pd.to_numeric(source_df[col], errors="coerce")

    if vals.notna().sum() >= 10 and vals.nunique(dropna=True) > 1:
        feature_cols.append(col)

features = source_df[["patient_folder"] + feature_cols].copy()

for col in feature_cols:
    features[col] = pd.to_numeric(features[col], errors="coerce")

# Some support files may have duplicate patient rows. Median aggregation is safe for patient-level features.
patient_features = features.groupby("patient_folder", as_index=False).median(numeric_only=True)

labels_small = labels[["patient_folder", "pCR_label"]].copy()
labels_small["label"] = pd.to_numeric(labels_small["pCR_label"], errors="coerce")

model_df = patient_features.merge(labels_small[["patient_folder", "label"]], on="patient_folder", how="inner")
model_df = model_df.dropna(subset=["label"]).copy()
model_df["label"] = model_df["label"].astype(int)

model_df.to_csv(TABLES / "phase8c_ftv_modeling_dataset.csv", index=False)

# feature completeness
comp = []
for col in feature_cols:
    vals = pd.to_numeric(model_df[col], errors="coerce") if col in model_df.columns else pd.Series([])
    comp.append({
        "feature": col,
        "non_missing": int(vals.notna().sum()),
        "missing": int(vals.isna().sum()),
        "mean": float(vals.mean()) if vals.notna().sum() else np.nan,
        "std": float(vals.std()) if vals.notna().sum() > 1 else np.nan,
        "min": float(vals.min()) if vals.notna().sum() else np.nan,
        "max": float(vals.max()) if vals.notna().sum() else np.nan,
    })

feature_comp = pd.DataFrame(comp).sort_values("non_missing", ascending=False)
feature_comp.to_csv(TABLES / "phase8c_ftv_feature_completeness.csv", index=False)

# -------------------------------------------------------------------
# 3. Run ML if possible
# -------------------------------------------------------------------
n_patients = len(model_df)
if n_patients > 0:
    y = model_df["label"].astype(int).values
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
else:
    y = np.array([])
    n_pos = 0
    n_neg = 0

usable_features = []
for col in feature_cols:
    if col not in model_df.columns:
        continue
    vals = pd.to_numeric(model_df[col], errors="coerce")
    if vals.notna().sum() >= max(6, int(0.5 * max(n_patients, 1))) and vals.nunique(dropna=True) > 1:
        usable_features.append(col)

ml_possible = n_patients >= 10 and n_pos >= 3 and n_neg >= 3 and len(usable_features) >= 2

metrics = pd.DataFrame()
predictions = pd.DataFrame()
importance = pd.DataFrame()

plt.figure(figsize=(8, 5))
if n_patients > 0:
    pd.Series(y).map({0: "non-pCR", 1: "pCR"}).value_counts().plot(kind="bar")
    plt.title("Phase 8C FTV Modeling Label Distribution")
    plt.ylabel("Patients")
    plt.xticks(rotation=20)
else:
    plt.text(0.5, 0.5, "No labeled patients after merge", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "250_phase8c_ftv_label_distribution.png", dpi=220)
plt.close()

if ml_possible:
    X = model_df[usable_features].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    min_class = min(n_pos, n_neg)
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
            random_state=42,
            class_weight="balanced",
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

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
    except Exception:
        pass

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    metric_rows = []
    prediction_rows = []
    prob_store = {}

    for model_name, model in models.items():
        probs = np.zeros(len(y), dtype=float)
        preds = np.zeros(len(y), dtype=int)

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
            "n_features": len(usable_features),
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
                "model": model_name,
                "patient_folder": pid,
                "true_label": int(true),
                "predicted_probability": float(prob),
                "predicted_label": int(pred),
            })

        prob_store[model_name] = probs

    metrics = pd.DataFrame(metric_rows).sort_values(["AUROC", "AUPRC", "Brier_score"], ascending=[False, False, True])
    predictions = pd.DataFrame(prediction_rows)

    metrics.to_csv(TABLES / "phase8c_ftv_model_performance.csv", index=False)
    predictions.to_csv(TABLES / "phase8c_ftv_model_predictions.csv", index=False)

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
        for i, col in enumerate(usable_features):
            rows.append({
                "feature": col,
                "permutation_importance_mean": float(perm.importances_mean[i]),
                "permutation_importance_std": float(perm.importances_std[i]),
                "mean_pCR": float(X.loc[y == 1, col].mean()),
                "mean_non_pCR": float(X.loc[y == 0, col].mean()),
                "mean_difference_pCR_minus_non_pCR": float(X.loc[y == 1, col].mean() - X.loc[y == 0, col].mean()),
            })
        importance = pd.DataFrame(rows).sort_values("permutation_importance_mean", ascending=False)
    except Exception as e:
        importance = pd.DataFrame([{"importance_error": str(e)[:300]}])

    importance.to_csv(TABLES / "phase8c_ftv_feature_importance.csv", index=False)

    # Figure 251 model performance
    plt.figure(figsize=(9, 5))
    metrics.sort_values("AUROC").plot(x="model", y="AUROC", kind="barh", legend=False)
    plt.title("Phase 8C FTV Model AUROC")
    plt.xlabel("AUROC")
    plt.tight_layout()
    plt.savefig(FIGS / "251_phase8c_ftv_model_auroc.png", dpi=220)
    plt.close()

    # Figure 252 AUPRC
    plt.figure(figsize=(9, 5))
    metrics.sort_values("AUPRC").plot(x="model", y="AUPRC", kind="barh", legend=False)
    plt.title("Phase 8C FTV Model AUPRC")
    plt.xlabel("AUPRC")
    plt.tight_layout()
    plt.savefig(FIGS / "252_phase8c_ftv_model_auprc.png", dpi=220)
    plt.close()

    # Figure 253 importance
    plt.figure(figsize=(10, 7))
    if "permutation_importance_mean" in importance.columns:
        top = importance.head(15).sort_values("permutation_importance_mean")
        plt.barh(top["feature"], top["permutation_importance_mean"])
        plt.title("Phase 8C FTV Feature Importance")
        plt.xlabel("Permutation importance")
    else:
        plt.text(0.5, 0.5, "No importance available", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "253_phase8c_ftv_feature_importance.png", dpi=220)
    plt.close()

else:
    pd.DataFrame([{
        "reason": "ML not run",
        "n_patients": n_patients,
        "positive_pCR": n_pos,
        "negative_non_pCR": n_neg,
        "usable_features": len(usable_features),
        "rule": "Need at least 10 labeled patients, 3 pCR, 3 non-pCR, and 2 numeric features."
    }]).to_csv(TABLES / "phase8c_ftv_model_performance.csv", index=False)

    plt.figure(figsize=(8, 5))
    txt = (
        "Phase 8C ML not run\n\n"
        f"Patients: {n_patients}\n"
        f"pCR: {n_pos}\n"
        f"non-pCR: {n_neg}\n"
        f"usable features: {len(usable_features)}"
    )
    plt.text(0.5, 0.5, txt, ha="center", va="center", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "251_phase8c_ftv_model_auroc.png", dpi=220)
    plt.close()

# -------------------------------------------------------------------
# 4. Markdown previews
# -------------------------------------------------------------------
def make_md(csv_path, md_path, title, max_rows=80, max_cols=12):
    if not Path(csv_path).exists():
        return

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

make_md(TABLES / "phase8c_ftv_source_selection.csv", VIEWS / "phase8c_ftv_source_selection.md", "Phase 8C FTV Source Selection")
make_md(TABLES / "phase8c_ftv_modeling_dataset.csv", VIEWS / "phase8c_ftv_modeling_dataset.md", "Phase 8C FTV Modeling Dataset")
make_md(TABLES / "phase8c_ftv_feature_completeness.csv", VIEWS / "phase8c_ftv_feature_completeness.md", "Phase 8C FTV Feature Completeness")
make_md(TABLES / "phase8c_ftv_model_performance.csv", VIEWS / "phase8c_ftv_model_performance.md", "Phase 8C FTV Model Performance")
make_md(TABLES / "phase8c_ftv_feature_importance.csv", VIEWS / "phase8c_ftv_feature_importance.md", "Phase 8C FTV Feature Importance")

# -------------------------------------------------------------------
# 5. Report
# -------------------------------------------------------------------
report = []
report.append("# Phase 8C FTV Feature Modeling Report")
report.append("")
report.append("This phase uses the strongest Phase 8B support-data source, the multi-feature MRI NACT / FTV table, to build a patient-level pCR modeling dataset.")
report.append("")
report.append("## Selected source")
report.append("")
report.append("- File: " + str(best_path))
report.append("- Patient column: " + str(best_patient_col))
report.append("- Rows in source: " + str(len(source_df)))
report.append("- Numeric source features: " + str(len(feature_cols)))
report.append("")
report.append("## Main outputs")
report.append("")
report.append("- reports/tables/phase8c_ftv_source_selection.csv")
report.append("- reports/tables/phase8c_ftv_modeling_dataset.csv")
report.append("- reports/tables/phase8c_ftv_feature_completeness.csv")
report.append("- reports/tables/phase8c_ftv_model_performance.csv")
report.append("- reports/table_views/phase8c_ftv_modeling_dataset.md")
report.append("- reports/figures/250_phase8c_ftv_label_distribution.png")
report.append("- reports/figures/251_phase8c_ftv_model_auroc.png")
report.append("")
report.append("## Main counts")
report.append("")
report.append("- Labeled FTV patients: " + str(n_patients))
report.append("- pCR positive: " + str(n_pos))
report.append("- non-pCR negative: " + str(n_neg))
report.append("- Usable numeric FTV features: " + str(len(usable_features)))
report.append("")
report.append("## Modeling status")
report.append("")
if ml_possible:
    report.append("ML was run successfully.")
    report.append("- Best model: " + str(metrics.iloc[0]["model"]))
    report.append("- AUROC: " + f"{float(metrics.iloc[0]['AUROC']):.4f}")
    report.append("- AUPRC: " + f"{float(metrics.iloc[0]['AUPRC']):.4f}")
    report.append("- Brier score: " + f"{float(metrics.iloc[0]['Brier_score']):.4f}")
else:
    report.append("ML was not run because the FTV-label merge did not meet the minimum sample/class rule.")
report.append("")
report.append("## Interpretation")
report.append("")
report.append("This is now a stronger path than the recovered DICOM SEG path because it uses official multi-feature MRI/FTV support data rather than weak fractional DICOM SEG recovery.")
report.append("")
report.append("## Next step")
report.append("")
if ml_possible:
    report.append("Phase 8D should compare the FTV support-data model against Phase 5A and prepare a clean manuscript-ready result.")
else:
    report.append("If ML was not run, inspect patient ID overlap and the label table, then decide whether to expand the label template.")

(REPORTS / "Phase8C_FTV_Feature_Modeling_Report.md").write_text("\n".join(report))

# Dashboard update
dash = REPORTS / "Dashboard.md"
old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
dash_add = "\n\n## Phase 8C FTV feature modeling\n\n- [FTV feature modeling report](Phase8C_FTV_Feature_Modeling_Report.md)\n- [FTV modeling dataset](table_views/phase8c_ftv_modeling_dataset.md)\n- [FTV model performance](table_views/phase8c_ftv_model_performance.md)\n"
if "Phase 8C FTV feature modeling" not in old_dash:
    dash.write_text(old_dash.rstrip() + dash_add)

# README update
readme = REPO / "README.md"
old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
readme_add = "\n\n### Phase 8C: FTV feature modeling\n\nMain outputs:\n\n- reports/Phase8C_FTV_Feature_Modeling_Report.md\n- reports/tables/phase8c_ftv_modeling_dataset.csv\n- reports/tables/phase8c_ftv_model_performance.csv\n"
if "Phase 8C: FTV feature modeling" not in old_readme:
    readme.write_text(old_readme.rstrip() + readme_add)

print("Phase 8C complete.")
print("Selected source:", best_path)
print("Labeled FTV patients:", n_patients)
print("pCR positive:", n_pos)
print("non-pCR negative:", n_neg)
print("Usable numeric FTV features:", len(usable_features))
print("ML possible:", "yes" if ml_possible else "no")
if ml_possible:
    print("Best model:", metrics.iloc[0]["model"])
    print("Best AUROC:", round(float(metrics.iloc[0]["AUROC"]), 4))
    print("Best AUPRC:", round(float(metrics.iloc[0]["AUPRC"]), 4))
