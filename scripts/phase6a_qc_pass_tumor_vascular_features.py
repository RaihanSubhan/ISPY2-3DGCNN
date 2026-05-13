from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pydicom
from skimage import filters, morphology, measure
from scipy import ndimage as ndi

REPO = Path.home() / "ISPY2-3DGCNN"
REPORTS = REPO / "reports"
TABLES = REPORTS / "tables"
FIGS = REPORTS / "figures"
VIEWS = REPORTS / "table_views"

for p in [REPORTS, TABLES, FIGS, VIEWS]:
    p.mkdir(parents=True, exist_ok=True)

AUDIT_PATH = TABLES / "segmentation_qc_case_audit.csv"

if not AUDIT_PATH.exists():
    raise SystemExit("Missing reports/tables/segmentation_qc_case_audit.csv. Run segmentation QC first.")

def to_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

def normalize01(img):
    img = img.astype(float)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((img - lo) / (hi - lo), 0, 1)

def read_seg(seg_file):
    ds = pydicom.dcmread(seg_file, force=True)
    arr = np.squeeze(np.asarray(ds.pixel_array))

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 4:
        arr = np.squeeze(arr)
        if arr.ndim == 4:
            arr = arr[..., 0]

    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected SEG shape: {arr.shape}")

    mask3d = arr > 0

    row_spacing = 1.0
    col_spacing = 1.0
    slice_thickness = 1.0

    try:
        pm = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
        ps = pm.PixelSpacing
        row_spacing = float(ps[0])
        col_spacing = float(ps[1])
        slice_thickness = float(getattr(pm, "SliceThickness", 1.0) or 1.0)
    except Exception:
        pass

    return ds, mask3d, row_spacing, col_spacing, slice_thickness

def read_mr(mr_file):
    ds = pydicom.dcmread(mr_file, force=True)
    img = ds.pixel_array.astype(float)

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    img = img * slope + intercept

    return ds, normalize01(img)

def peritumor_ring(mask2d, radius=15):
    dilated = morphology.binary_dilation(mask2d, morphology.disk(radius))
    ring = dilated & (~mask2d)
    return ring

def safe_mean(x):
    x = np.asarray(x)
    if x.size == 0:
        return np.nan
    return float(np.mean(x))

def safe_std(x):
    x = np.asarray(x)
    if x.size == 0:
        return np.nan
    return float(np.std(x))

def shape_features(mask2d, row_spacing, col_spacing):
    mask2d = mask2d.astype(bool)
    area_px = int(mask2d.sum())
    area_mm2 = float(area_px * row_spacing * col_spacing)

    if area_px == 0:
        return {
            "tumor_area_px_2d": 0,
            "tumor_area_mm2_2d": 0,
            "perimeter_px_2d": np.nan,
            "circularity_2d": np.nan,
            "solidity_2d": np.nan,
            "eccentricity_2d": np.nan,
            "bbox_height_px": np.nan,
            "bbox_width_px": np.nan,
        }

    perimeter_px = float(measure.perimeter(mask2d))
    circularity = float(4 * np.pi * area_px / (perimeter_px ** 2 + 1e-8))

    labels = measure.label(mask2d)
    props = measure.regionprops(labels)

    if not props:
        return {
            "tumor_area_px_2d": area_px,
            "tumor_area_mm2_2d": area_mm2,
            "perimeter_px_2d": perimeter_px,
            "circularity_2d": circularity,
            "solidity_2d": np.nan,
            "eccentricity_2d": np.nan,
            "bbox_height_px": np.nan,
            "bbox_width_px": np.nan,
        }

    prop = max(props, key=lambda p: p.area)
    minr, minc, maxr, maxc = prop.bbox

    return {
        "tumor_area_px_2d": area_px,
        "tumor_area_mm2_2d": area_mm2,
        "perimeter_px_2d": perimeter_px,
        "circularity_2d": circularity,
        "solidity_2d": float(prop.solidity),
        "eccentricity_2d": float(prop.eccentricity),
        "bbox_height_px": int(maxr - minr),
        "bbox_width_px": int(maxc - minc),
    }

def radial_features(img, mask2d, n_bins=5):
    out = {}
    mask2d = mask2d.astype(bool)

    if mask2d.sum() == 0:
        for i in range(n_bins):
            out[f"radial_signal_bin_{i+1}"] = np.nan
        return out

    cy, cx = ndi.center_of_mass(mask2d.astype(float))
    yy, xx = np.indices(mask2d.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    max_r = np.percentile(rr[mask2d], 95)
    bins = np.linspace(0, max_r, n_bins + 1)

    for i in range(n_bins):
        sel = mask2d & (rr >= bins[i]) & (rr < bins[i + 1])
        out[f"radial_signal_bin_{i+1}"] = safe_mean(img[sel])

    return out

def vessel_features(img, mask2d, ring):
    out = {}

    try:
        vesselness = filters.frangi(img, sigmas=[1, 2, 3, 4], black_ridges=False)
    except Exception:
        vesselness = np.zeros_like(img)

    positive = vesselness[vesselness > 0]
    if positive.size > 0:
        threshold = np.percentile(positive, 90)
    else:
        threshold = 1.0

    vessels = vesselness > threshold
    vessels = morphology.remove_small_objects(vessels.astype(bool), min_size=8)
    skeleton = morphology.skeletonize(vessels)

    tumor_area = max(int(mask2d.sum()), 1)
    ring_area = max(int(ring.sum()), 1)

    out["vessel_density_tumor_2d"] = float(skeleton[mask2d].sum() / tumor_area)
    out["vessel_density_peritumor_2d"] = float(skeleton[ring].sum() / ring_area)
    out["vesselness_mean_tumor_2d"] = safe_mean(vesselness[mask2d])
    out["vesselness_mean_peritumor_2d"] = safe_mean(vesselness[ring])
    out["vessel_density_ratio_tumor_to_peritumor"] = float(
        out["vessel_density_tumor_2d"] / (out["vessel_density_peritumor_2d"] + 1e-8)
    )

    return out, vesselness, skeleton

def habitat_features(img, mask2d):
    out = {}
    vals = img[mask2d]

    if vals.size == 0:
        out["low_enhancement_fraction"] = np.nan
        out["mid_enhancement_fraction"] = np.nan
        out["high_enhancement_fraction"] = np.nan
        return out

    q25 = np.percentile(vals, 25)
    q75 = np.percentile(vals, 75)

    out["low_enhancement_fraction"] = float(np.mean(vals <= q25))
    out["mid_enhancement_fraction"] = float(np.mean((vals > q25) & (vals < q75)))
    out["high_enhancement_fraction"] = float(np.mean(vals >= q75))

    return out

def make_overlay(img, mask2d, ring, skeleton, out_path, title):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("MR")
    axes[0].axis("off")

    axes[1].imshow(mask2d, cmap="gray")
    axes[1].set_title("Tumor SEG")
    axes[1].axis("off")

    axes[2].imshow(img, cmap="gray")
    axes[2].contour(mask2d, linewidths=1)
    axes[2].contour(ring, linewidths=0.7)
    axes[2].set_title("Tumor + peritumor")
    axes[2].axis("off")

    axes[3].imshow(img, cmap="gray")
    axes[3].contour(mask2d, linewidths=1)
    axes[3].imshow(np.ma.masked_where(~skeleton, skeleton), alpha=0.8)
    axes[3].set_title("Vessel skeleton")
    axes[3].axis("off")

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def make_md(csv_path, md_path, title, max_rows=60, max_cols=12):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=12)
    args = parser.parse_args()

    audit = pd.read_csv(AUDIT_PATH, dtype=str).fillna("")

    if "qc_status" in audit.columns:
        audit = audit[audit["qc_status"].eq("visual_qc_pass_candidate")].copy()

    audit = audit.head(args.max_cases)

    rows = []
    overlay_files = []

    for _, r in audit.iterrows():
        case_no = len(rows) + 1

        row = {
            "case_no": case_no,
            "patient_folder": r.get("patient_folder", ""),
            "study_folder": r.get("study_folder", ""),
            "seg_series_uid": r.get("seg_series_uid", ""),
            "qc_status": r.get("qc_status", ""),
            "feature_status": "not_started",
        }

        try:
            seg_file = r.get("seg_file", "")
            mr_file = r.get("matched_mr_file", "")

            if not seg_file or not Path(seg_file).exists():
                row["feature_status"] = "missing_seg_file"
                rows.append(row)
                continue

            if not mr_file or not Path(mr_file).exists():
                row["feature_status"] = "missing_mr_file"
                rows.append(row)
                continue

            _, mask3d, row_spacing, col_spacing, slice_thickness = read_seg(seg_file)
            _, img = read_mr(mr_file)

            frame_idx = to_int(r.get("selected_seg_frame", 0), 0)
            if frame_idx < 0 or frame_idx >= mask3d.shape[0]:
                frame_pixels = mask3d.reshape(mask3d.shape[0], -1).sum(axis=1)
                frame_idx = int(np.argmax(frame_pixels))

            mask2d = mask3d[frame_idx].astype(bool)

            if img.shape != mask2d.shape:
                row["feature_status"] = "shape_mismatch"
                row["mr_shape"] = str(img.shape)
                row["mask_shape"] = str(mask2d.shape)
                rows.append(row)
                continue

            ring = peritumor_ring(mask2d, radius=15)

            row["feature_status"] = "success"
            row["selected_seg_frame"] = frame_idx
            row["row_spacing_mm"] = row_spacing
            row["col_spacing_mm"] = col_spacing
            row["slice_thickness_mm"] = slice_thickness

            # 3D proxy values
            voxel_mm3 = row_spacing * col_spacing * slice_thickness
            row["tumor_volume_proxy_mm3_3d"] = float(mask3d.sum() * voxel_mm3)
            row["tumor_volume_proxy_cm3_3d"] = float(mask3d.sum() * voxel_mm3 / 1000.0)
            row["nonzero_seg_frames_3d"] = int((mask3d.reshape(mask3d.shape[0], -1).sum(axis=1) > 0).sum())

            # 2D shape
            row.update(shape_features(mask2d, row_spacing, col_spacing))

            # signal features
            tumor_vals = img[mask2d]
            ring_vals = img[ring]

            row["mr_signal_mean_tumor"] = safe_mean(tumor_vals)
            row["mr_signal_std_tumor"] = safe_std(tumor_vals)
            row["mr_signal_mean_peritumor"] = safe_mean(ring_vals)
            row["mr_signal_std_peritumor"] = safe_std(ring_vals)
            row["tumor_to_peritumor_signal_ratio"] = float(
                row["mr_signal_mean_tumor"] / (row["mr_signal_mean_peritumor"] + 1e-8)
            )

            row.update(radial_features(img, mask2d, n_bins=5))
            row.update(habitat_features(img, mask2d))

            vessel_out, vesselness, skeleton = vessel_features(img, mask2d, ring)
            row.update(vessel_out)

            out_case = FIGS / f"phase6a_vascular_overlay_case_{case_no:03d}.png"
            make_overlay(
                img,
                mask2d,
                ring,
                skeleton,
                out_case,
                f"Phase 6A vascular features | case {case_no} | {row['patient_folder']}",
            )
            overlay_files.append(out_case)

        except Exception as e:
            row["feature_status"] = "failed_exception"
            row["feature_error"] = str(e)[:300]

        rows.append(row)

    feat = pd.DataFrame(rows)
    out_csv = TABLES / "phase6a_qc_pass_tumor_vascular_features.csv"
    feat.to_csv(out_csv, index=False)

    make_md(
        out_csv,
        VIEWS / "phase6a_qc_pass_tumor_vascular_features.md",
        "Phase 6A QC-Pass Tumor Vascular Features"
    )

    # Figure 140
    plt.figure(figsize=(8, 5))
    vals = pd.to_numeric(feat.get("tumor_area_mm2_2d", pd.Series([])), errors="coerce").dropna()
    if len(vals) > 0:
        vals.plot(kind="hist", bins=15)
        plt.title("Phase 6A Tumor Area Distribution")
        plt.xlabel("Tumor area, mm2")
        plt.ylabel("Cases")
    else:
        plt.text(0.5, 0.5, "No tumor area values", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "140_phase6a_tumor_area_mm2.png", dpi=220)
    plt.close()

    # Figure 141
    plt.figure(figsize=(8, 5))
    x = pd.to_numeric(feat.get("vessel_density_tumor_2d", pd.Series([])), errors="coerce")
    y = pd.to_numeric(feat.get("vessel_density_peritumor_2d", pd.Series([])), errors="coerce")
    valid = x.notna() & y.notna()
    if valid.sum() > 0:
        plt.scatter(x[valid], y[valid])
        plt.xlabel("Tumor vessel density")
        plt.ylabel("Peritumor vessel density")
        plt.title("Phase 6A Tumor vs Peritumor Vessel Density")
    else:
        plt.text(0.5, 0.5, "No vessel density values", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "141_phase6a_vessel_density_scatter.png", dpi=220)
    plt.close()

    # Figure 142
    plt.figure(figsize=(8, 5))
    vals = pd.to_numeric(feat.get("tumor_to_peritumor_signal_ratio", pd.Series([])), errors="coerce").dropna()
    if len(vals) > 0:
        vals.plot(kind="hist", bins=15)
        plt.title("Phase 6A Tumor-to-Peritumor Signal Ratio")
        plt.xlabel("Signal ratio")
        plt.ylabel("Cases")
    else:
        plt.text(0.5, 0.5, "No signal ratio values", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "142_phase6a_signal_ratio.png", dpi=220)
    plt.close()

    # Figure 143
    plt.figure(figsize=(8, 5))
    habitat_cols = ["low_enhancement_fraction", "mid_enhancement_fraction", "high_enhancement_fraction"]
    if all(c in feat.columns for c in habitat_cols):
        means = feat[habitat_cols].apply(pd.to_numeric, errors="coerce").mean()
        means.plot(kind="bar")
        plt.title("Phase 6A Mean Tumor Habitat Fractions")
        plt.ylabel("Mean fraction")
        plt.xticks(rotation=25, ha="right")
    else:
        plt.text(0.5, 0.5, "Habitat fractions not available", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "143_phase6a_habitat_fractions.png", dpi=220)
    plt.close()

    # Figure 144 correlation
    for col in feat.columns:
        feat[col + "_numtmp"] = pd.to_numeric(feat[col], errors="coerce")

    num_cols = [c for c in feat.columns if c.endswith("_numtmp")]
    num = feat[num_cols].dropna(axis=1, how="all")
    num = num.loc[:, num.nunique(dropna=True) > 1]

    plt.figure(figsize=(9, 8))
    if num.shape[1] >= 2:
        corr = num.corr()
        labels = [c.replace("_numtmp", "")[:18] for c in corr.columns]
        plt.imshow(corr, vmin=-1, vmax=1)
        plt.colorbar(label="correlation")
        plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
        plt.yticks(range(len(labels)), labels, fontsize=6)
        plt.title("Phase 6A Feature Correlation")
    else:
        plt.text(0.5, 0.5, "Not enough numeric features for correlation", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "144_phase6a_feature_correlation_heatmap.png", dpi=220)
    plt.close()

    # Figure 145 overlay grid
    files = [p for p in overlay_files if p.exists()]
    if files:
        cols = 2
        rows_n = int(np.ceil(len(files) / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(14, rows_n * 5))
        axes = np.array(axes).reshape(-1)

        for ax, f in zip(axes, files):
            img = plt.imread(f)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f.name, fontsize=8)

        for ax in axes[len(files):]:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(FIGS / "145_phase6a_vascular_feature_overlay_grid.png", dpi=180)
        plt.close()
    else:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "No overlay files created", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(FIGS / "145_phase6a_vascular_feature_overlay_grid.png", dpi=220)
        plt.close()

    n_total = len(feat)
    n_success = int((feat["feature_status"] == "success").sum()) if "feature_status" in feat.columns else 0
    n_failed = n_total - n_success

    summary = []
    summary.append("# Phase 6A QC-Pass Tumor and Vascular Feature Summary")
    summary.append("")
    summary.append("This phase extracts tumor, peritumor, radial signal, habitat, and vesselness features from visual-QC-passed SEG-MR cases.")
    summary.append("")
    summary.append("## Main outputs")
    summary.append("")
    summary.append("- reports/tables/phase6a_qc_pass_tumor_vascular_features.csv")
    summary.append("- reports/table_views/phase6a_qc_pass_tumor_vascular_features.md")
    summary.append("- reports/figures/140_phase6a_tumor_area_mm2.png")
    summary.append("- reports/figures/141_phase6a_vessel_density_scatter.png")
    summary.append("- reports/figures/142_phase6a_signal_ratio.png")
    summary.append("- reports/figures/143_phase6a_habitat_fractions.png")
    summary.append("- reports/figures/144_phase6a_feature_correlation_heatmap.png")
    summary.append("- reports/figures/145_phase6a_vascular_feature_overlay_grid.png")
    summary.append("")
    summary.append("## Main counts")
    summary.append("")
    summary.append("- QC-passed cases attempted: " + str(n_total))
    summary.append("- Feature extraction success cases: " + str(n_success))
    summary.append("- Failed cases: " + str(n_failed))
    summary.append("")
    summary.append("## Features created")
    summary.append("")
    summary.append("- tumor area and volume proxy")
    summary.append("- circularity, solidity, eccentricity, bounding box")
    summary.append("- tumor and peritumor signal")
    summary.append("- tumor-to-peritumor signal ratio")
    summary.append("- radial signal profile")
    summary.append("- low, middle, and high enhancement habitat fractions")
    summary.append("- Frangi vesselness and vessel skeleton density")
    summary.append("")
    summary.append("## Interpretation")
    summary.append("")
    summary.append("These features are stronger than the earlier proxy-only table because they are extracted from visually QC-passed segmentation overlays.")
    summary.append("")
    summary.append("## Limitation")
    summary.append("")
    summary.append("This phase still uses representative 2D tumor slices plus 3D volume proxy. A future phase should extend the same feature extraction over full 3D matched tumor volumes and across longitudinal visits.")
    summary.append("")
    summary.append("## Next step")
    summary.append("")
    summary.append("Phase 6B should merge these vascular features with pCR labels and rerun pilot ML to test whether vascular features improve AUROC and AUPRC.")

    (REPORTS / "Phase6A_QC_Pass_Tumor_Vascular_Feature_Summary.md").write_text("\n".join(summary))

    dash = REPORTS / "Dashboard.md"
    old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
    add = "\n\n## Phase 6A table view\n\n- [QC-pass tumor vascular features](table_views/phase6a_qc_pass_tumor_vascular_features.md)\n"
    if "Phase 6A table view" not in old_dash:
        dash.write_text(old_dash.rstrip() + add)

    readme = REPO / "README.md"
    old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
    add_readme = "\n\n### Phase 6A: QC-pass tumor and vascular features\n\nMain outputs:\n\n- reports/tables/phase6a_qc_pass_tumor_vascular_features.csv\n- reports/Phase6A_QC_Pass_Tumor_Vascular_Feature_Summary.md\n"
    if "Phase 6A: QC-pass tumor and vascular features" not in old_readme:
        readme.write_text(old_readme.rstrip() + add_readme)

    print("Phase 6A complete.")
    print("QC-passed cases attempted:", n_total)
    print("Feature extraction success cases:", n_success)
    print("Failed cases:", n_failed)

if __name__ == "__main__":
    main()
