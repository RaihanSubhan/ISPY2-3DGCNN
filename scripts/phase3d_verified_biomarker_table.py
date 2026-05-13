from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path.home() / "ISPY2-3DGCNN"
REPORTS = REPO / "reports"
TABLES = REPORTS / "tables"
FIGS = REPORTS / "figures"
VIEWS = REPORTS / "table_views"

TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
VIEWS.mkdir(parents=True, exist_ok=True)

src = TABLES / "phase3c_geometry_matching_qc.csv"

if not src.exists():
    raise SystemExit("Missing Phase 3C table. Run Phase 3C first.")

df = pd.read_csv(src, dtype=str).fillna("")

# Convert useful numeric columns
numeric_cols = [
    "seg_frames",
    "seg_rows",
    "seg_cols",
    "nonzero_seg_frames",
    "total_mask_pixels",
    "row_spacing_mm",
    "col_spacing_mm",
    "slice_thickness_mm",
    "volume_proxy_mm3",
    "mr_slices_with_z",
    "nonzero_frames_with_seg_z",
    "matched_nonzero_frames",
    "shape_matched_frames",
    "mean_abs_z_error_mm",
    "median_abs_z_error_mm",
    "max_abs_z_error_mm",
    "pct_frames_within_1mm",
    "pct_frames_within_2mm",
    "pct_frames_within_5mm",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Strict verified set
verified = df[df["status"].eq("geometry_ready")].copy()

# If no strict cases, make a warning-level table too
usable = df[df["status"].isin(["geometry_ready", "z_ready_shape_warning", "geometry_warning"])].copy()

# Biomarker table
bio = usable.copy()

bio["biomarker_tier"] = bio["status"].map({
    "geometry_ready": "strict_verified",
    "z_ready_shape_warning": "z_verified_shape_warning",
    "geometry_warning": "geometry_warning"
}).fillna("not_ready")

if "volume_proxy_mm3" in bio.columns:
    bio["tumor_volume_proxy_cm3"] = bio["volume_proxy_mm3"] / 1000.0
else:
    bio["tumor_volume_proxy_cm3"] = np.nan

if "total_mask_pixels" in bio.columns and "nonzero_seg_frames" in bio.columns:
    bio["mean_mask_pixels_per_nonzero_frame"] = bio["total_mask_pixels"] / bio["nonzero_seg_frames"].replace(0, np.nan)
else:
    bio["mean_mask_pixels_per_nonzero_frame"] = np.nan

if "shape_matched_frames" in bio.columns and "matched_nonzero_frames" in bio.columns:
    bio["shape_match_fraction"] = bio["shape_matched_frames"] / bio["matched_nonzero_frames"].replace(0, np.nan)
else:
    bio["shape_match_fraction"] = np.nan

# Keep compact columns for modeling
keep = [
    "patient_folder",
    "study_folder",
    "seg_series_uid",
    "matched_mr_series_uid",
    "link_status",
    "link_confidence",
    "status",
    "biomarker_tier",
    "nonzero_seg_frames",
    "total_mask_pixels",
    "volume_proxy_mm3",
    "tumor_volume_proxy_cm3",
    "mean_mask_pixels_per_nonzero_frame",
    "row_spacing_mm",
    "col_spacing_mm",
    "slice_thickness_mm",
    "matched_nonzero_frames",
    "shape_matched_frames",
    "shape_match_fraction",
    "mean_abs_z_error_mm",
    "median_abs_z_error_mm",
    "max_abs_z_error_mm",
    "pct_frames_within_1mm",
    "pct_frames_within_2mm",
    "pct_frames_within_5mm",
]

keep = [c for c in keep if c in bio.columns]
bio_out = bio[keep].copy()

out_csv = TABLES / "phase3d_verified_biomarker_table.csv"
bio_out.to_csv(out_csv, index=False)

# Figure 50: biomarker tier
plt.figure(figsize=(8, 5))
bio_out["biomarker_tier"].fillna("unknown").value_counts().sort_values().plot(kind="barh")
plt.title("Phase 3D Biomarker Tier")
plt.xlabel("Number of cases")
plt.tight_layout()
plt.savefig(FIGS / "50_phase3d_biomarker_tier.png", dpi=220)
plt.close()

# Figure 51: tumor volume proxy
plt.figure(figsize=(8, 5))
vals = pd.to_numeric(bio_out.get("tumor_volume_proxy_cm3", pd.Series([])), errors="coerce").dropna()
if len(vals) > 0:
    vals.plot(kind="hist", bins=30)
    plt.title("Tumor Volume Proxy Distribution")
    plt.xlabel("Tumor volume proxy, cm3")
    plt.ylabel("Number of cases")
else:
    plt.text(0.5, 0.5, "No tumor volume values available", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "51_phase3d_volume_proxy_cm3.png", dpi=220)
plt.close()

# Figure 52: z error vs volume
plt.figure(figsize=(8, 5))
x = pd.to_numeric(bio_out.get("mean_abs_z_error_mm", pd.Series([])), errors="coerce")
y = pd.to_numeric(bio_out.get("tumor_volume_proxy_cm3", pd.Series([])), errors="coerce")
valid = x.notna() & y.notna()
if valid.sum() > 0:
    plt.scatter(x[valid], y[valid])
    plt.xlabel("Mean absolute z error, mm")
    plt.ylabel("Tumor volume proxy, cm3")
    plt.title("Geometry Error vs Tumor Volume Proxy")
else:
    plt.text(0.5, 0.5, "No paired z-error and volume data available", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "52_phase3d_z_error_vs_volume.png", dpi=220)
plt.close()

# Figure 53: per-patient count
plt.figure(figsize=(9, 5))
if "patient_folder" in bio_out.columns and len(bio_out) > 0:
    bio_out.groupby("patient_folder").size().value_counts().sort_index().plot(kind="bar")
    plt.title("Number of Biomarker-Ready Records per Patient")
    plt.xlabel("Records per patient")
    plt.ylabel("Number of patients")
else:
    plt.text(0.5, 0.5, "No patient data available", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "53_phase3d_records_per_patient.png", dpi=220)
plt.close()

# Markdown table view
preview = bio_out.head(50).iloc[:, :12].fillna("")

def clean(x):
    x = str(x).replace("|", "\\|").replace("\n", " ")
    if len(x) > 80:
        x = x[:77] + "..."
    return x

lines = []
lines.append("# Phase 3D Verified Biomarker Table")
lines.append("")
lines.append("Rows: " + str(len(bio_out)))
lines.append("Columns: " + str(len(bio_out.columns)))
lines.append("")
lines.append("This preview shows the first 50 rows and first 12 columns.")
lines.append("")
lines.append("| " + " | ".join([clean(c) for c in preview.columns]) + " |")
lines.append("| " + " | ".join(["---"] * len(preview.columns)) + " |")

for _, row in preview.iterrows():
    lines.append("| " + " | ".join([clean(v) for v in row.tolist()]) + " |")

(VIEWS / "phase3d_verified_biomarker_table.md").write_text("\n".join(lines))

# Summary
n_total = len(df)
n_strict = len(verified)
n_usable = len(bio_out)
n_patients = bio_out["patient_folder"].nunique() if "patient_folder" in bio_out.columns else 0

summary = [
    "# Phase 3D Verified Biomarker Table Summary",
    "",
    "This phase creates a compact modeling-ready biomarker table from the Phase 3C geometry matching QC output.",
    "",
    "## Main outputs",
    "",
    "- reports/tables/phase3d_verified_biomarker_table.csv",
    "- reports/table_views/phase3d_verified_biomarker_table.md",
    "- reports/figures/50_phase3d_biomarker_tier.png",
    "- reports/figures/51_phase3d_volume_proxy_cm3.png",
    "- reports/figures/52_phase3d_z_error_vs_volume.png",
    "- reports/figures/53_phase3d_records_per_patient.png",
    "",
    "## Main counts",
    "",
    "- Phase 3C cases reviewed: " + str(n_total),
    "- Strict geometry-ready cases: " + str(n_strict),
    "- Biomarker-table cases included: " + str(n_usable),
    "- Patients represented: " + str(n_patients),
    "",
    "## Interpretation",
    "",
    "The strict tier should be used for primary analysis. Warning-tier records can be used for sensitivity analysis only.",
    "",
    "## Important limitation",
    "",
    "This table is still based on geometry QC and volume-proxy features. It is not yet the final vascular-perfusion feature table.",
    "",
    "## Next step",
    "",
    "Phase 4 should add modeling labels and train baseline prediction models such as logistic regression, SVM, random forest, and XGBoost.",
]

(REPORTS / "Phase3D_Verified_Biomarker_Table_Summary.md").write_text("\n".join(summary))

# Dashboard update
dash_path = REPORTS / "Dashboard.md"
old = dash_path.read_text() if dash_path.exists() else "# ISPY2 4D Atlas Dashboard\n"
add = "\n\n## Phase 3D table view\n\n- [Phase 3D verified biomarker table](table_views/phase3d_verified_biomarker_table.md)\n"
if "Phase 3D verified biomarker table" not in old:
    dash_path.write_text(old.rstrip() + add)

# README update
readme_path = REPO / "README.md"
old_readme = readme_path.read_text() if readme_path.exists() else "# ISPY2-3DGCNN\n"
readme_add = "\n\n### Phase 3D: verified biomarker table\n\nMain outputs:\n\n- reports/tables/phase3d_verified_biomarker_table.csv\n- reports/table_views/phase3d_verified_biomarker_table.md\n- reports/Phase3D_Verified_Biomarker_Table_Summary.md\n"
if "Phase 3D: verified biomarker table" not in old_readme:
    readme_path.write_text(old_readme.rstrip() + readme_add)

print("Phase 3D complete.")
print("Phase 3C cases reviewed:", n_total)
print("Strict geometry-ready cases:", n_strict)
print("Biomarker-table cases included:", n_usable)
print("Patients represented:", n_patients)
