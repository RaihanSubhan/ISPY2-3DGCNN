from pathlib import Path
import pandas as pd

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

link_path = TABLES / "seg_mr_link_table.csv"
qc_path = TABLES / "seg_mask_qc_features.csv"

if not link_path.exists():
    raise SystemExit("Missing reports/tables/seg_mr_link_table.csv")

links = pd.read_csv(link_path, dtype=str).fillna("")

if qc_path.exists():
    qc = pd.read_csv(qc_path, dtype=str).fillna("")
else:
    qc = pd.DataFrame()

keep_cols = [
    "patient_folder",
    "study_folder",
    "seg_series_uid",
    "seg_first_dcm",
    "seg_series_path",
    "link_status",
    "link_confidence",
    "matched_mr_series_uid",
    "matched_mr_series_path",
    "matched_mr_first_dcm",
    "matched_mr_series_description",
    "matched_mr_protocol_name",
    "matched_mr_dcm_count",
    "segment_labels",
]

for col in keep_cols:
    if col not in links.columns:
        links[col] = ""

features = links[keep_cols].copy()

features["has_seg_file"] = features["seg_first_dcm"].astype(str).map(lambda x: "yes" if x.strip() else "no")
features["has_mr_file"] = features["matched_mr_first_dcm"].astype(str).map(lambda x: "yes" if x.strip() else "no")

features["phase3_ready_basic"] = (
    (features["has_seg_file"] == "yes") &
    (features["has_mr_file"] == "yes") &
    (features["link_confidence"].astype(str) != "")
).map(lambda x: "yes" if x else "no")

if not qc.empty and "seg_series_uid" in qc.columns:
    qc_cols = [
        "seg_series_uid",
        "decode_status",
        "n_frames",
        "nonzero_frames",
        "total_mask_pixels",
        "volume_proxy_mm3",
        "segment_pixel_counts",
    ]
    qc_cols = [c for c in qc_cols if c in qc.columns]
    features = features.merge(qc[qc_cols], on="seg_series_uid", how="left")
else:
    features["decode_status"] = "not_available"

features["decode_status"] = features.get("decode_status", "not_available")
features["decode_status"] = features["decode_status"].fillna("not_available")

out_csv = TABLES / "phase3b_tumor_peritumor_features_pilot.csv"
features.to_csv(out_csv, index=False)

plt.figure(figsize=(8, 5))
features["link_confidence"].fillna("unknown").value_counts().sort_values().plot(kind="barh")
plt.title("Phase 3B Link Confidence")
plt.xlabel("Number of SEG-MR pairs")
plt.tight_layout()
plt.savefig(FIGS / "30_phase3b_link_confidence.png", dpi=220)
plt.close()

plt.figure(figsize=(8, 5))
features["phase3_ready_basic"].value_counts().sort_values().plot(kind="barh")
plt.title("Phase 3B Basic Feature Readiness")
plt.xlabel("Number of SEG-MR pairs")
plt.tight_layout()
plt.savefig(FIGS / "31_phase3b_feature_readiness.png", dpi=220)
plt.close()

plt.figure(figsize=(10, 7))
features["matched_mr_series_description"].fillna("unknown").replace("", "unknown").value_counts().head(20).sort_values().plot(kind="barh")
plt.title("Top Matched MR Series Descriptions")
plt.xlabel("Count")
plt.tight_layout()
plt.savefig(FIGS / "32_phase3b_top_mr_descriptions.png", dpi=220)
plt.close()

plt.figure(figsize=(12, 6))
roadmap = [
    "Phase 3B: Tumor and Peritumor Feature Pilot",
    "",
    "SEG-MR linked pairs",
    "-> Basic readiness check",
    "-> SEG decode QC when available",
    "-> Next: geometry-aligned tumor features",
    "-> Final: volume, shape, vesselness, radial signal, and longitudinal deltas",
]
plt.text(0.5, 0.5, "\n".join(roadmap), ha="center", va="center", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "33_phase3b_feature_roadmap.png", dpi=220)
plt.close()

preview = features.head(40).iloc[:, :10].fillna("")

def clean(x):
    x = str(x).replace("|", "\\|").replace("\n", " ")
    if len(x) > 80:
        x = x[:77] + "..."
    return x

lines = []
lines.append("# Phase 3B Tumor and Peritumor Feature Pilot")
lines.append("")
lines.append("Rows: " + str(len(features)))
lines.append("Columns: " + str(len(features.columns)))
lines.append("")
lines.append("This preview shows the first 40 rows and first 10 columns.")
lines.append("")
lines.append("| " + " | ".join([clean(c) for c in preview.columns]) + " |")
lines.append("| " + " | ".join(["---"] * len(preview.columns)) + " |")

for _, row in preview.iterrows():
    lines.append("| " + " | ".join([clean(v) for v in row.tolist()]) + " |")

(VIEWS / "phase3b_tumor_peritumor_features_pilot.md").write_text("\n".join(lines))

n_pairs = len(features)
n_patients = features["patient_folder"].nunique()
n_ready = int((features["phase3_ready_basic"] == "yes").sum())
n_high = int((features["link_confidence"] == "high").sum())
n_medium = int((features["link_confidence"] == "medium").sum())
n_weak = int((features["link_confidence"] == "weak").sum())
n_decoded = int((features["decode_status"] == "success").sum())

summary_lines = [
    "# Phase 3B Tumor and Peritumor Feature Pilot Summary",
    "",
    "This phase prepares the linked SEG-MR pairs for verified tumor and peritumor feature extraction.",
    "",
    "## Main outputs",
    "",
    "- reports/tables/phase3b_tumor_peritumor_features_pilot.csv",
    "- reports/table_views/phase3b_tumor_peritumor_features_pilot.md",
    "- reports/figures/30_phase3b_link_confidence.png",
    "- reports/figures/31_phase3b_feature_readiness.png",
    "- reports/figures/32_phase3b_top_mr_descriptions.png",
    "- reports/figures/33_phase3b_feature_roadmap.png",
    "",
    "## Main counts",
    "",
    "- SEG-MR pairs checked: " + str(n_pairs),
    "- Unique patients represented: " + str(n_patients),
    "- Basic Phase 3 ready pairs: " + str(n_ready),
    "- High-confidence links: " + str(n_high),
    "- Medium-confidence links: " + str(n_medium),
    "- Weak links: " + str(n_weak),
    "- SEG decode QC success cases available: " + str(n_decoded),
    "",
    "## Interpretation",
    "",
    "This step confirms which linked SEG-MR pairs are ready for tumor and peritumor feature extraction. It does not yet claim final clinical tumor volume.",
    "",
    "## Next step",
    "",
    "Phase 3C should perform geometry-aware matching between SEG frames and MR slices, then compute verified tumor volume, shape, radial signal, Frangi vesselness, peritumor vessel density, and longitudinal change features.",
]

(REPORTS / "Phase3B_Tumor_Peritumor_Features_Summary.md").write_text("\n".join(summary_lines))

dash_path = REPORTS / "Dashboard.md"
old_dash = dash_path.read_text() if dash_path.exists() else "# ISPY2 4D Atlas Dashboard\n"
dash_add = "\n\n## Phase 3B table view\n\n- [Phase 3B tumor and peritumor feature pilot](table_views/phase3b_tumor_peritumor_features_pilot.md)\n"
if "Phase 3B tumor and peritumor feature pilot" not in old_dash:
    dash_path.write_text(old_dash.rstrip() + dash_add)

readme_path = REPO / "README.md"
old_readme = readme_path.read_text() if readme_path.exists() else "# ISPY2-3DGCNN\n"
readme_add = "\n\n### Phase 3B: tumor and peritumor feature pilot\n\nMain outputs:\n\n- reports/tables/phase3b_tumor_peritumor_features_pilot.csv\n- reports/table_views/phase3b_tumor_peritumor_features_pilot.md\n- reports/Phase3B_Tumor_Peritumor_Features_Summary.md\n"
if "Phase 3B: tumor and peritumor feature pilot" not in old_readme:
    readme_path.write_text(old_readme.rstrip() + readme_add)

print("Phase 3B complete.")
print("SEG-MR pairs checked:", n_pairs)
print("Unique patients represented:", n_patients)
print("Basic ready pairs:", n_ready)
print("SEG decode QC success cases:", n_decoded)
