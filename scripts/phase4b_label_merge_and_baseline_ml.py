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
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

REPO = Path.home() / "ISPY2-3DGCNN"
REPORTS = REPO / "reports"
TABLES = REPORTS / "tables"
FIGS = REPORTS / "figures"
VIEWS = REPORTS / "table_views"

TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
VIEWS.mkdir(parents=True, exist_ok=True)

bio_path = TABLES / "phase3d_verified_biomarker_table.csv"
inv_path = TABLES / "phase4a_label_source_inventory.csv"
template_path = TABLES / "phase4a_required_label_template.csv"

if not bio_path.exists():
    raise SystemExit("Missing phase3d_verified_biomarker_table.csv. Run Phase 3D first.")

bio = pd.read_csv(bio_path, dtype=str).fillna("")
patients = set(bio["patient_folder"].astype(str).unique())

def norm_patient(x):
    s = str(x).upper().strip()
    m = re.search(r"ISPY2[-_ ]?(\d+)", s)
    if m:
        return "ISPY2-" + m.group(1)
    return s

def read_table(path):
    path = Path(path)
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, dtype=str).fillna("")
        if path.suffix.lower() == ".tsv":
            return pd.read_csv(path, sep="\t", dtype=str).fillna("")
        if path.suffix.lower() == ".txt":
            return pd.read_csv(path, sep=None, engine="python", dtype=str).fillna("")
        if path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(path, dtype=str).fillna("")
    except Exception:
        return None
    return None

def find_patient_column(df):
    best_col = None
    best_overlap = 0
    for col in df.columns:
        vals = df[col].astype(str).map(norm_patient)
        overlap = len(set(vals).intersection(patients))
        col_low = col.lower()
        if "patient" in col_low or "subject" in col_low or "case" in col_low or col_low in ["id", "patientid"]:
            overlap += 2
        if overlap > best_overlap:
            best_overlap = overlap
            best_col = col
    if best_overlap <= 0:
        return None, 0
    return best_col, best_overlap

def possible_label_columns(df):
    keys = ["pcr", "pathologic", "complete", "response", "responder", "rcb", "outcome"]
    bad = ["path", "file", "folder", "series", "study", "uid", "description"]
    cols = []
    for col in df.columns:
        low = col.lower()
        if any(k in low for k in keys) and not any(b in low for b in bad):
            cols.append(col)
    return cols

def map_label_value(value, colname):
    s = str(value).strip().lower()
    c = str(colname).lower()

    if s in ["", "nan", "none", "null", "na", "n/a", "unknown"]:
        return np.nan

    neg_words = ["non-pcr", "non pcr", "no pcr", "not pcr", "no", "false", "non responder", "non-responder", "nonresponse", "not complete"]
    pos_words = ["pcr", "yes", "true", "complete response", "pathologic complete", "responder", "response"]

    if "rcb" in c:
        if s in ["0", "0.0", "rcb0", "rcb-0", "rcb 0", "class 0"]:
            return 1
        if s in ["1", "1.0", "2", "2.0", "3", "3.0", "rcb1", "rcb2", "rcb3", "rcb-i", "rcb-ii", "rcb-iii"]:
            return 0
        m = re.search(r"(\d+)", s)
        if m:
            return 1 if int(m.group(1)) == 0 else 0

    for w in neg_words:
        if w in s:
            return 0

    for w in pos_words:
        if w in s:
            return 1

    if s in ["1", "1.0"]:
        return 1
    if s in ["0", "0.0"]:
        return 0

    return np.nan

candidate_paths = []

if inv_path.exists():
    inv = pd.read_csv(inv_path, dtype=str).fillna("")
    if "file_path" in inv.columns:
        for p in inv["file_path"].tolist():
            if p and Path(p).exists():
                candidate_paths.append(p)

if template_path.exists():
    candidate_paths.append(str(template_path))

candidate_paths = list(dict.fromkeys(candidate_paths))

merge_reports = []
label_datasets = []

for p in candidate_paths:
    df = read_table(p)
    if df is None or df.empty:
        continue

    patient_col, overlap = find_patient_column(df)
    label_cols = possible_label_columns(df)

    if patient_col is None or not label_cols:
        merge_reports.append({
            "file_path": p,
            "patient_column": patient_col if patient_col else "",
            "patient_overlap": overlap,
            "label_column": "",
            "usable_labeled_patients": 0,
            "n_positive": 0,
            "n_negative": 0,
            "status": "no_patient_or_label_column"
        })
        continue

    tmp = df.copy()
    tmp["patient_folder"] = tmp[patient_col].map(norm_patient)

    for lab_col in label_cols:
        y = tmp[lab_col].map(lambda v: map_label_value(v, lab_col))
        t = tmp[["patient_folder"]].copy()
        t["label"] = y
        t = t.dropna(subset=["label"])
        t["label"] = t["label"].astype(int)
        t = t[t["patient_folder"].isin(patients)]
        t = t.drop_duplicates("patient_folder")

        n_pos = int((t["label"] == 1).sum())
        n_neg = int((t["label"] == 0).sum())
        n_use = len(t)

        status = "usable" if n_use >= 6 and n_pos >= 2 and n_neg >= 2 else "not_enough_two_class_labels"

        merge_reports.append({
            "file_path": p,
            "patient_column": patient_col,
            "patient_overlap": overlap,
            "label_column": lab_col,
            "usable_labeled_patients": n_use,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "status": status
        })

        if status == "usable":
            t["label_source_file"] = p
            t["label_source_column"] = lab_col
            label_datasets.append(t)

merge_report = pd.DataFrame(merge_reports)
merge_report.to_csv(TABLES / "phase4b_label_merge_candidates.csv", index=False)

if len(label_datasets) == 0:
    summary = [
        "# Phase 4B Label Merge and Baseline ML Summary",
        "",
        "No usable two-class pCR or response label was found automatically.",
        "",
        "## What happened",
        "",
        "- Candidate files were scanned.",
        "- Patient ID overlap and label-like columns were checked.",
        "- No label source had enough matched positive and negative labels for safe supervised ML.",
        "",
        "## Main output",
        "",
        "- reports/tables/phase4b_label_merge_candidates.csv",
        "",
        "## Next action",
        "",
        "Open `reports/tables/phase4b_label_merge_candidates.csv` in GitHub. If no true pCR label is present, fill `reports/tables/phase4a_required_label_template.csv` with pCR labels and rerun Phase 4B.",
    ]
    (REPORTS / "Phase4B_Label_Merge_and_Baseline_ML_Summary.md").write_text("\n".join(summary))

    plt.figure(figsize=(9, 5))
    if not merge_report.empty:
        top = merge_report.sort_values("usable_labeled_patients", ascending=False).head(15)
        plt.barh(top["label_column"].fillna("none").astype(str), top["usable_labeled_patients"].astype(float))
        plt.xlabel("Usable labeled patients")
        plt.title("Phase 4B Label Merge Candidates")
    else:
        plt.text(0.5, 0.5, "No label candidate files found", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "70_phase4b_label_merge_candidates.png", dpi=220)
    plt.close()

    print("Phase 4B complete: no usable supervised label found.")
    raise SystemExit(0)

# choose best label dataset
best = sorted(label_datasets, key=lambda x: len(x), reverse=True)[0]

# patient-level biomarker table
num_cols = []
for col in bio.columns:
    vals = pd.to_numeric(bio[col], errors="coerce")
    if vals.notna().sum() >= 5 and vals.nunique(dropna=True) > 1:
        num_cols.append(col)

patient_features = bio[["patient_folder"] + num_cols].copy()
for col in num_cols:
    patient_features[col] = pd.to_numeric(patient_features[col], errors="coerce")

patient_features = patient_features.groupby("patient_folder", as_index=False).median(numeric_only=True)
model_df = patient_features.merge(best[["patient_folder", "label", "label_source_file", "label_source_column"]], on="patient_folder", how="inner")
model_df.to_csv(TABLES / "phase4b_modeling_dataset.csv", index=False)

X_df = model_df[num_cols].copy()
X_df = X_df.fillna(X_df.median(numeric_only=True))
y = model_df["label"].astype(int).values

models = {
    "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "SVM_RBF": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, class_weight="balanced")),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=min(5, max(1, len(model_df)//4)))),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

min_class = int(pd.Series(y).value_counts().min())
n_splits = min(5, min_class)

performance_rows = []
pred_rows = []

if n_splits >= 2:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for name, model in models.items():
        probs = np.zeros(len(y))
        preds = np.zeros(len(y), dtype=int)

        for train_idx, test_idx in cv.split(X_df, y):
            model.fit(X_df.iloc[train_idx], y[train_idx])
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X_df.iloc[test_idx])[:, 1]
            else:
                p = model.decision_function(X_df.iloc[test_idx])
                p = (p - p.min()) / (p.max() - p.min() + 1e-9)
            probs[test_idx] = p
            preds[test_idx] = (p >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()

        performance_rows.append({
            "model": name,
            "n_patients": len(y),
            "n_splits": n_splits,
            "AUROC": roc_auc_score(y, probs),
            "AUPRC": average_precision_score(y, probs),
            "accuracy": accuracy_score(y, preds),
            "balanced_accuracy": balanced_accuracy_score(y, preds),
            "F1": f1_score(y, preds, zero_division=0),
            "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
            "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        })

        for pid, true, prob, pred in zip(model_df["patient_folder"], y, probs, preds):
            pred_rows.append({
                "model": name,
                "patient_folder": pid,
                "true_label": int(true),
                "predicted_probability": float(prob),
                "predicted_label": int(pred),
            })

perf = pd.DataFrame(performance_rows)
preds = pd.DataFrame(pred_rows)

perf.to_csv(TABLES / "phase4b_model_performance.csv", index=False)
preds.to_csv(TABLES / "phase4b_model_predictions.csv", index=False)

# Figures
plt.figure(figsize=(8, 5))
pd.Series(y).map({0: "non-pCR/negative", 1: "pCR/positive"}).value_counts().plot(kind="bar")
plt.title("Phase 4B Label Distribution")
plt.ylabel("Patients")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(FIGS / "70_phase4b_label_distribution.png", dpi=220)
plt.close()

plt.figure(figsize=(9, 5))
if not perf.empty:
    perf.sort_values("AUROC").plot(x="model", y="AUROC", kind="barh", legend=False)
    plt.xlabel("Cross-validated AUROC")
    plt.title("Phase 4B Baseline Model AUROC")
else:
    plt.text(0.5, 0.5, "Not enough labels for CV", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "71_phase4b_model_auroc.png", dpi=220)
plt.close()

plt.figure(figsize=(9, 5))
if not perf.empty:
    perf.sort_values("AUPRC").plot(x="model", y="AUPRC", kind="barh", legend=False)
    plt.xlabel("Cross-validated AUPRC")
    plt.title("Phase 4B Baseline Model AUPRC")
else:
    plt.text(0.5, 0.5, "Not enough labels for CV", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "72_phase4b_model_auprc.png", dpi=220)
plt.close()

# simple Markdown table views
def make_md(csv_path, md_path, title, max_rows=40, max_cols=10):
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    small = df.head(max_rows).iloc[:, :max_cols]

    def clean(x):
        x = str(x).replace("|", "\\|").replace("\n", " ")
        if len(x) > 80:
            x = x[:77] + "..."
        return x

    lines = ["# " + title, "", "Rows: " + str(len(df)), "Columns: " + str(len(df.columns)), ""]
    lines.append("| " + " | ".join(clean(c) for c in small.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(small.columns)) + " |")
    for _, row in small.iterrows():
        lines.append("| " + " | ".join(clean(v) for v in row.tolist()) + " |")
    md_path.write_text("\n".join(lines))

make_md(TABLES / "phase4b_label_merge_candidates.csv", VIEWS / "phase4b_label_merge_candidates.md", "Phase 4B Label Merge Candidates")
make_md(TABLES / "phase4b_modeling_dataset.csv", VIEWS / "phase4b_modeling_dataset.md", "Phase 4B Modeling Dataset")
if not perf.empty:
    make_md(TABLES / "phase4b_model_performance.csv", VIEWS / "phase4b_model_performance.md", "Phase 4B Model Performance")

summary = [
    "# Phase 4B Label Merge and Baseline ML Summary",
    "",
    "This phase merges the best available pCR/response label with the biomarker table and trains baseline ML models.",
    "",
    "## Main outputs",
    "",
    "- reports/tables/phase4b_label_merge_candidates.csv",
    "- reports/tables/phase4b_modeling_dataset.csv",
    "- reports/tables/phase4b_model_performance.csv",
    "- reports/tables/phase4b_model_predictions.csv",
    "- reports/figures/70_phase4b_label_distribution.png",
    "- reports/figures/71_phase4b_model_auroc.png",
    "- reports/figures/72_phase4b_model_auprc.png",
    "",
    "## Label source",
    "",
    "- File: " + str(best["label_source_file"].iloc[0]),
    "- Column: " + str(best["label_source_column"].iloc[0]),
    "",
    "## Main counts",
    "",
    "- Labeled patients used: " + str(len(y)),
    "- Positive labels: " + str(int((y == 1).sum())),
    "- Negative labels: " + str(int((y == 0).sum())),
    "- Numeric biomarker features: " + str(len(num_cols)),
    "- CV folds: " + str(n_splits),
    "",
    "## Interpretation",
    "",
    "These are pilot baseline models. Because the cohort is small, results should be treated as feasibility results, not final clinical performance.",
    "",
    "## Next step",
    "",
    "Phase 4C should add model calibration, SHAP-style interpretation, and a cleaner train/test design after labels are verified.",
]

(REPORTS / "Phase4B_Label_Merge_and_Baseline_ML_Summary.md").write_text("\n".join(summary))

dash_path = REPORTS / "Dashboard.md"
old = dash_path.read_text() if dash_path.exists() else "# ISPY2 4D Atlas Dashboard\n"
add = "\n\n## Phase 4B table views\n\n- [Model performance](table_views/phase4b_model_performance.md)\n- [Modeling dataset](table_views/phase4b_modeling_dataset.md)\n- [Label merge candidates](table_views/phase4b_label_merge_candidates.md)\n"
if "Phase 4B table views" not in old:
    dash_path.write_text(old.rstrip() + add)

print("Phase 4B complete.")
print("Labeled patients used:", len(y))
print("Positive labels:", int((y == 1).sum()))
print("Negative labels:", int((y == 0).sum()))
print("Numeric biomarker features:", len(num_cols))
print("CV folds:", n_splits)
