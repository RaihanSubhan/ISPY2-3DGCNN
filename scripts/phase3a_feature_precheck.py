from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path.home() / "ISPY2-3DGCNN"
TABLES = REPO / "reports" / "tables"
FIGS = REPO / "reports" / "figures"
REPORTS = REPO / "reports"

FIGS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

link_path = TABLES / "seg_mr_link_table.csv"
qc_path = TABLES / "seg_mask_qc_features.csv"

if not link_path.exists():
    raise SystemExit("Missing reports/tables/seg_mr_link_table.csv")

links = pd.read_csv(link_path, dtype=str)

if qc_path.exists():
    qc = pd.read_csv(qc_path, dtype=str)
else:
    qc = pd.DataFrame()

# Build a compact Phase 3A manifest
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

available_cols = [c for c in keep_cols if c in links.columns]
manifest = links[available_cols].copy()

# Add readiness flags
manifest["has_seg_file"] = manifest.get("seg_first_dcm", "").fillna("").astype(str).map(lambda x: "yes" if x else "no")
manifest["has_mr_file"] = manifest.get("matched_mr_first_dcm", "").fillna("").astype(str).map(lambda x: "yes" if x else "no")

if "matched_mr_dcm_count" in manifest.columns:
    manifest["matched_mr_dcm_count_num"] = pd.to_numeric(manifest["matched_mr_dcm_count"], errors="coerce")
else:
    manifest["matched_mr_dcm_count_num"] = np.nan

# Merge QC if available
if not qc.empty and "seg_series_uid" in qc.columns:
    qc_small_cols = [
        c for c in [
            "seg_series_uid",
            "decode_status",
            "n_frames",
            "nonzero_frames",
            "total_mask_pixels",
            "volume_proxy_mm3",
            "segment_pixel_counts",
        ] if c in qc.columns
    ]
    manifest = manifest.merge(qc[qc_small_cols], on="seg_series_uid", how="left")
else:
    manifest["decode_status"] = "not_available"

# Ready for Phase 3 if matched MR exists and SEG metadata exists
manifest["phase3_ready_basic"] = (
    (manifest["has_seg_file"] == "yes") &
    (manifest["has_mr_file"] == "yes") &
    (manifest["link_confidence"].fillna("") != "")
).map(lambda x: "yes" if x else "no")

manifest.to_csv(TABLES / "phase3a_feature_precheck.csv", index=False)

# Figure 20: link confidence
plt.figure(figsize=(8, 5))
manifest["link_confidence"].fillna("unknown").value_counts().sort_values().plot(kind="barh")
plt.title("Phase 3A Link Confidence")
plt.xlabel("Number of SEG-MR pairs")
plt.tight_layout()
plt.savefig(FIGS / "20_phase3a_link_confidence.png", dpi=220)
plt.close()

# Figure 21: mask volume proxy
plt.figure(figsize=(8, 5))
if "volume_proxy_mm3" in manifest.columns:
    vals = pd.to_numeric(manifest["volume_proxy_mm3"], errors="coerce").dropna()
    if len(vals) > 0:
        vals.plot(kind="hist", bins=30)
        plt.title("SEG Mask Volume Proxy Distribution")
        plt.xlabel("Volume proxy mm3")
        plt.ylabel("Count")
    else:
        plt.text(0.5, 0.5, "No decoded volume proxy values found", ha="center", va="center")
        plt.axis("off")
else:
    plt.text(0.5, 0.5, "No volume_proxy_mm3 column found", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "21_phase3a_mask_volume_proxy.png", dpi=220)
plt.close()

# Figure 22: MR descriptions
plt.figure(figsize=(10, 7))
if "matched_mr_series_description" in manifest.columns:
    desc = manifest["matched_mr_series_description"].fillna("unknown").replace("", "unknown")
    desc.value_counts().head(20).sort_values().plot(kind="barh")
    plt.title("Top Matched MR Series Descriptions")
    plt.xlabel("Count")
else:
    plt.text(0.5, 0.5, "No MR description column found", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "22_phase3a_top_mr_descriptions.png", dpi=220)
plt.close()

# Summary
n_pairs = len(manifest)
n_ready = int((manifest["phase3_ready_basic"] == "yes").sum())
n_patients = manifest["patient_folder"].nunique() if "patient_folder" in manifest.columns else 0
n_qc_success = int((manifest.get("decode_status", pd.Series([])) == "success").sum()) if "decode_status" in manifest.columns else 0

summary = f"""# Phase 3A Feature Precheck Summary

This step creates a safe precheck table for verified tumor and peritumor feature extraction.

## Main outputs

- `reports/tables/phase3a_feature_precheck.csv`
- `reports/figures/20_phase3a_link_confidence.png`
- `reports/figures/21_phase3a_mask_volume_proxy.png`
- `reports/figures/22_phase3a_top_mr_descriptions.png`

## Main counts

- SEG-MR linked pairs checked: {n_pairs}
- Unique patients represented: {n_patients}
- Basic Phase 3 ready pairs: {n_ready}
- SEG decode QC success cases already available: {n_qc_success}

## Interpretation

Phase 3A does not compute final clinical biomarkers yet. It checks whether each linked SEG-MR pair has enough metadata and file paths for verified feature extraction. This is needed before measuring tumor volume, shape, vesselness, and longitudinal change.

## Next step

Phase 3B should decode SEG masks and align them with matched MR image geometry for a small test group first. After that, the full cohort can be processed.
"""

(REPORTS / "Phase3A_Feature_Precheck_Summary.md").write_text(summary)

print("Phase 3A feature precheck complete.")
print(f"Linked pairs: {n_pairs}")
print(f"Unique patients: {n_patients}")
print(f"Basic ready pairs: {n_ready}")
print(f"QC success cases: {n_qc_success}")
