from pathlib import Path
import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path.home() / "ISPY2-3DGCNN"
DATA = Path.home() / "ispy2"
REPORTS = REPO / "reports"
TABLES = REPORTS / "tables"
FIGS = REPORTS / "figures"
VIEWS = REPORTS / "table_views"

TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
VIEWS.mkdir(parents=True, exist_ok=True)

bio_path = TABLES / "phase3d_verified_biomarker_table.csv"

if not bio_path.exists():
    raise SystemExit("Missing phase3d_verified_biomarker_table.csv. Run Phase 3D first.")

bio = pd.read_csv(bio_path, dtype=str).fillna("")

# -------------------------
# 1. Find possible label files
# -------------------------
label_keywords = [
    "pcr", "pathologic", "complete", "response", "outcome", "rcb",
    "residual", "burden", "treatment", "arm", "subtype", "her2",
    "er", "pr", "hr", "tnbc", "clinical"
]

def find_small_table_files(root, max_depth=5):
    rows = []
    root = Path(root)
    if not root.exists():
        return rows

    root_parts = len(root.parts)

    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        depth = len(p.parts) - root_parts
        if depth > max_depth:
            dirnames[:] = []
            continue

        for name in filenames:
            low = name.lower()
            if low.endswith((".csv", ".tsv", ".txt", ".xlsx")):
                fp = p / name
                try:
                    size_mb = fp.stat().st_size / (1024 * 1024)
                except Exception:
                    size_mb = np.nan

                rows.append({
                    "file_path": str(fp),
                    "file_name": name,
                    "size_mb": round(size_mb, 4) if pd.notna(size_mb) else "",
                    "source_root": str(root),
                })

    return rows

candidate_files = []
candidate_files.extend(find_small_table_files(REPO, max_depth=6))
candidate_files.extend(find_small_table_files(DATA, max_depth=4))

scan_rows = []

for item in candidate_files:
    path = Path(item["file_path"])
    columns = []
    status = "not_read"
    matched_cols = []

    try:
        if path.suffix.lower() == ".csv":
            df0 = pd.read_csv(path, nrows=5)
            columns = list(df0.columns)
            status = "read_csv"
        elif path.suffix.lower() == ".tsv":
            df0 = pd.read_csv(path, sep="\t", nrows=5)
            columns = list(df0.columns)
            status = "read_tsv"
        elif path.suffix.lower() == ".txt":
            df0 = pd.read_csv(path, sep=None, engine="python", nrows=5)
            columns = list(df0.columns)
            status = "read_txt"
        else:
            columns = []
            status = "xlsx_listed_not_read"
    except Exception as e:
        status = "read_failed: " + str(e)[:80]

    for c in columns:
        c_low = str(c).lower()
        if any(k in c_low for k in label_keywords):
            matched_cols.append(str(c))

    scan_rows.append({
        **item,
        "read_status": status,
        "n_columns_read": len(columns),
        "matched_label_like_columns": "; ".join(matched_cols),
        "all_columns_preview": "; ".join([str(c) for c in columns[:30]]),
        "label_signal_score": len(matched_cols),
    })

label_inventory = pd.DataFrame(scan_rows)
label_inventory = label_inventory.sort_values(
    ["label_signal_score", "size_mb"],
    ascending=[False, True]
) if not label_inventory.empty else pd.DataFrame()

label_inventory.to_csv(TABLES / "phase4a_label_source_inventory.csv", index=False)

# -------------------------
# 2. Create label template
# -------------------------
patients = sorted(bio["patient_folder"].dropna().astype(str).unique()) if "patient_folder" in bio.columns else []

template = pd.DataFrame({
    "patient_folder": patients,
    "pCR_label": "",
    "response_label": "",
    "tumor_subtype": "",
    "treatment_arm": "",
    "notes": ""
})

template.to_csv(TABLES / "phase4a_required_label_template.csv", index=False)

# -------------------------
# 3. Feature completeness table
# -------------------------
feature_df = bio.copy()

for col in feature_df.columns:
    try:
        feature_df[col + "_numtest"] = pd.to_numeric(feature_df[col], errors="coerce")
    except Exception:
        pass

numeric_candidates = []
for col in bio.columns:
    vals = pd.to_numeric(bio[col], errors="coerce")
    if vals.notna().sum() > 0:
        numeric_candidates.append(col)

comp_rows = []
for col in numeric_candidates:
    vals = pd.to_numeric(bio[col], errors="coerce")
    comp_rows.append({
        "feature": col,
        "non_missing": int(vals.notna().sum()),
        "missing": int(vals.isna().sum()),
        "missing_pct": round(float(vals.isna().mean()), 4),
        "mean": round(float(vals.mean()), 5) if vals.notna().sum() else "",
        "std": round(float(vals.std()), 5) if vals.notna().sum() > 1 else "",
        "min": round(float(vals.min()), 5) if vals.notna().sum() else "",
        "max": round(float(vals.max()), 5) if vals.notna().sum() else "",
    })

feature_completeness = pd.DataFrame(comp_rows)
feature_completeness.to_csv(TABLES / "phase4a_feature_completeness.csv", index=False)

# -------------------------
# 4. Create simple unsupervised PCA map if possible
# -------------------------
pca_status = "not_run"
pca_df = pd.DataFrame()

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    numeric = bio[numeric_candidates].apply(pd.to_numeric, errors="coerce")

    good_cols = []
    for col in numeric.columns:
        if numeric[col].notna().sum() >= 5 and numeric[col].nunique(dropna=True) > 1:
            good_cols.append(col)

    numeric = numeric[good_cols]

    if len(numeric) >= 5 and len(good_cols) >= 2:
        numeric = numeric.fillna(numeric.median(numeric_only=True))
        X = StandardScaler().fit_transform(numeric)

        pca = PCA(n_components=2)
        pc = pca.fit_transform(X)

        k = min(3, len(numeric))
        clusters = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)

        pca_df = pd.DataFrame({
            "patient_folder": bio["patient_folder"].values if "patient_folder" in bio.columns else "",
            "PC1": pc[:, 0],
            "PC2": pc[:, 1],
            "cluster": clusters,
        })

        pca_df.to_csv(TABLES / "phase4a_unsupervised_pca_clusters.csv", index=False)
        pca_status = "success"
    else:
        pca_status = "not_enough_numeric_features"

except Exception as e:
    pca_status = "failed: " + str(e)[:120]

# -------------------------
# 5. Figures
# -------------------------
plt.figure(figsize=(9, 5))
if not label_inventory.empty:
    top = label_inventory.head(15).copy()
    top["plot_label"] = top["file_name"].astype(str).str[:35]
    plt.barh(top["plot_label"], top["label_signal_score"].astype(float))
    plt.xlabel("Label-like column score")
    plt.title("Phase 4A Candidate Label Files")
else:
    plt.text(0.5, 0.5, "No candidate label files found", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "60_phase4a_label_candidates.png", dpi=220)
plt.close()

plt.figure(figsize=(9, 6))
if not feature_completeness.empty:
    fc = feature_completeness.sort_values("non_missing", ascending=False).head(20)
    plt.barh(fc["feature"], fc["non_missing"])
    plt.xlabel("Non-missing values")
    plt.title("Top Available Biomarker Features")
else:
    plt.text(0.5, 0.5, "No numeric features found", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "61_phase4a_feature_completeness.png", dpi=220)
plt.close()

plt.figure(figsize=(8, 5))
if "biomarker_tier" in bio.columns:
    bio["biomarker_tier"].fillna("unknown").value_counts().sort_values().plot(kind="barh")
    plt.xlabel("Number of records")
    plt.title("Biomarker Tier Counts")
else:
    plt.text(0.5, 0.5, "No biomarker_tier column found", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "62_phase4a_biomarker_tier_counts.png", dpi=220)
plt.close()

plt.figure(figsize=(8, 6))
if pca_status == "success":
    plt.scatter(pca_df["PC1"], pca_df["PC2"], c=pca_df["cluster"])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Unsupervised Biomarker PCA Map")
else:
    plt.text(0.5, 0.5, "PCA not created: " + pca_status, ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "63_phase4a_unsupervised_pca.png", dpi=220)
plt.close()

# -------------------------
# 6. Markdown table view
# -------------------------
def make_md(csv_path, md_path, title, max_rows=40, max_cols=10):
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    small = df.head(max_rows).iloc[:, :max_cols]

    def clean(x):
        x = str(x).replace("|", "\\|").replace("\n", " ")
        if len(x) > 80:
            x = x[:77] + "..."
        return x

    lines = []
    lines.append("# " + title)
    lines.append("")
    lines.append("Rows: " + str(len(df)))
    lines.append("Columns: " + str(len(df.columns)))
    lines.append("")
    lines.append("| " + " | ".join([clean(c) for c in small.columns]) + " |")
    lines.append("| " + " | ".join(["---"] * len(small.columns)) + " |")

    for _, row in small.iterrows():
        lines.append("| " + " | ".join([clean(v) for v in row.tolist()]) + " |")

    md_path.write_text("\n".join(lines))

make_md(
    TABLES / "phase4a_label_source_inventory.csv",
    VIEWS / "phase4a_label_source_inventory.md",
    "Phase 4A Label Source Inventory"
)

make_md(
    TABLES / "phase4a_feature_completeness.csv",
    VIEWS / "phase4a_feature_completeness.md",
    "Phase 4A Feature Completeness"
)

# -------------------------
# 7. Summary
# -------------------------
n_label_files = len(label_inventory)
n_high_signal = int((label_inventory["label_signal_score"] > 0).sum()) if not label_inventory.empty else 0
n_features = len(numeric_candidates)
n_patients = len(patients)

summary = [
    "# Phase 4A Label and Modeling Readiness Summary",
    "",
    "This phase checks whether response labels are available and prepares the biomarker table for modeling.",
    "",
    "## Main outputs",
    "",
    "- reports/tables/phase4a_label_source_inventory.csv",
    "- reports/tables/phase4a_required_label_template.csv",
    "- reports/tables/phase4a_feature_completeness.csv",
    "- reports/table_views/phase4a_label_source_inventory.md",
    "- reports/table_views/phase4a_feature_completeness.md",
    "- reports/figures/60_phase4a_label_candidates.png",
    "- reports/figures/61_phase4a_feature_completeness.png",
    "- reports/figures/62_phase4a_biomarker_tier_counts.png",
    "- reports/figures/63_phase4a_unsupervised_pca.png",
    "",
    "## Main counts",
    "",
    "- Patients in biomarker table: " + str(n_patients),
    "- Candidate table files scanned: " + str(n_label_files),
    "- Files with label-like columns: " + str(n_high_signal),
    "- Numeric biomarker features detected: " + str(n_features),
    "- PCA status: " + pca_status,
    "",
    "## Interpretation",
    "",
    "If a pCR or response label file is found, Phase 4B can train supervised models. If no label file is found, the label template must be filled or downloaded from the official clinical support data.",
    "",
    "## Next step",
    "",
    "Phase 4B should merge true labels with the biomarker table and train baseline ML models: logistic regression, SVM, random forest, and XGBoost.",
]

(REPORTS / "Phase4A_Label_and_Modeling_Readiness_Summary.md").write_text("\n".join(summary))

# Dashboard update
dash_path = REPORTS / "Dashboard.md"
old = dash_path.read_text() if dash_path.exists() else "# ISPY2 4D Atlas Dashboard\n"
add = "\n\n## Phase 4A table views\n\n- [Label source inventory](table_views/phase4a_label_source_inventory.md)\n- [Feature completeness](table_views/phase4a_feature_completeness.md)\n"
if "Label source inventory" not in old:
    dash_path.write_text(old.rstrip() + add)

# README update
readme_path = REPO / "README.md"
old_readme = readme_path.read_text() if readme_path.exists() else "# ISPY2-3DGCNN\n"
readme_add = "\n\n### Phase 4A: label and modeling readiness\n\nMain outputs:\n\n- reports/tables/phase4a_label_source_inventory.csv\n- reports/tables/phase4a_required_label_template.csv\n- reports/Phase4A_Label_and_Modeling_Readiness_Summary.md\n"
if "Phase 4A: label and modeling readiness" not in old_readme:
    readme_path.write_text(old_readme.rstrip() + readme_add)

print("Phase 4A complete.")
print("Patients in biomarker table:", n_patients)
print("Candidate table files scanned:", n_label_files)
print("Files with label-like columns:", n_high_signal)
print("Numeric biomarker features:", n_features)
print("PCA status:", pca_status)
