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

SRC = TABLES / "phase7c_fractional_tumor_mask_candidate_table.csv"
LABELS = TABLES / "phase4a_required_label_template.csv"

if not SRC.exists():
    raise SystemExit("Missing phase7c_fractional_tumor_mask_candidate_table.csv. Run Phase 7C first.")

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def safe_int(x, default=0):
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

def read_seg_array(seg_file):
    ds = pydicom.dcmread(seg_file, force=True)
    arr = np.asarray(ds.pixel_array)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 4:
        arr = np.squeeze(arr)
        if arr.ndim == 4:
            arr = arr[..., 0]

    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected SEG shape: {arr.shape}")

    arr = arr.astype(float)

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

    return ds, arr, row_spacing, col_spacing, slice_thickness

def read_mr_image(mr_file):
    ds = pydicom.dcmread(mr_file, force=True)
    img = ds.pixel_array.astype(float)

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    img = img * slope + intercept

    return normalize01(img)

def peritumor_ring(mask2d, radius=15):
    dilated = morphology.binary_dilation(mask2d.astype(bool), morphology.disk(radius))
    return dilated & (~mask2d.astype(bool))

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
    out = {}

    mask2d = mask2d.astype(bool)
    area_px = int(mask2d.sum())
    area_mm2 = float(area_px * row_spacing * col_spacing)

    out["tumor_area_px_2d"] = area_px
    out["tumor_area_mm2_2d"] = area_mm2
    out["mask_fraction_2d"] = float(area_px / max(mask2d.size, 1))

    if area_px == 0:
        out.update({
            "perimeter_px_2d": np.nan,
            "circularity_2d": np.nan,
            "solidity_2d": np.nan,
            "eccentricity_2d": np.nan,
            "bbox_height_px": np.nan,
            "bbox_width_px": np.nan,
        })
        return out

    perimeter = float(measure.perimeter(mask2d))
    out["perimeter_px_2d"] = perimeter
    out["circularity_2d"] = float(4 * np.pi * area_px / (perimeter ** 2 + 1e-8))

    lab = measure.label(mask2d)
    props = measure.regionprops(lab)

    if props:
        prop = max(props, key=lambda p: p.area)
        minr, minc, maxr, maxc = prop.bbox

        out["solidity_2d"] = float(prop.solidity)
        out["eccentricity_2d"] = float(prop.eccentricity)
        out["bbox_height_px"] = int(maxr - minr)
        out["bbox_width_px"] = int(maxc - minc)
    else:
        out["solidity_2d"] = np.nan
        out["eccentricity_2d"] = np.nan
        out["bbox_height_px"] = np.nan
        out["bbox_width_px"] = np.nan

    return out

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

def habitat_features(img, mask2d):
    out = {}
    vals = img[mask2d.astype(bool)]

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

def vessel_features(img, mask2d, ring):
    out = {}

    try:
        vesselness = filters.frangi(img, sigmas=[1, 2, 3, 4], black_ridges=False)
    except Exception:
        vesselness = np.zeros_like(img)

    positive = vesselness[vesselness > 0]
    threshold = np.percentile(positive, 90) if positive.size else 1.0

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

def make_overlay(img, mask, ring, skeleton, out_path, title):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("MR")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Recovered tumor mask")
    axes[1].axis("off")

    axes[2].imshow(img, cmap="gray")
    axes[2].contour(mask, linewidths=1.2)
    axes[2].contour(ring, linewidths=0.7)
    axes[2].set_title("Tumor + peritumor")
    axes[2].axis("off")

    axes[3].imshow(img, cmap="gray")
    axes[3].contour(mask, linewidths=1.2)
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
    parser.add_argument("--max-cases", type=int, default=100)
    args = parser.parse_args()

    src = pd.read_csv(SRC, dtype=str).fillna("")
    work = src[src["recovery_status"].eq("fractional_tumor_candidate_pass")].copy()

    if args.max_cases:
        work = work.head(args.max_cases)

    rows = []
    overlays = []

    for _, r in work.iterrows():
        case_no = len(rows) + 1

        row = {
            "case_no": case_no,
            "patient_folder": r.get("patient_folder", ""),
            "study_folder": r.get("study_folder", ""),
            "seg_series_uid": r.get("seg_series_uid", ""),
            "recovery_status": r.get("recovery_status", ""),
            "feature_status": "not_started",
        }

        try:
            seg_file = r.get("seg_file", "")
            mr_file = r.get("matched_mr_file", "")

            selected_threshold = safe_float(r.get("selected_threshold", ""))
            selected_frame_index = safe_int(r.get("selected_frame_index", 0), 0)

            if not Path(seg_file).exists():
                row["feature_status"] = "missing_seg_file"
                rows.append(row)
                continue

            if not Path(mr_file).exists():
                row["feature_status"] = "missing_mr_file"
                rows.append(row)
                continue

            _, arr, row_spacing, col_spacing, slice_thickness = read_seg_array(seg_file)
            img = read_mr_image(mr_file)

            mask3d = arr >= selected_threshold

            if selected_frame_index < 0 or selected_frame_index >= mask3d.shape[0]:
                frame_pixels = mask3d.reshape(mask3d.shape[0], -1).sum(axis=1)
                selected_frame_index = int(np.argmax(frame_pixels))

            mask2d = mask3d[selected_frame_index].astype(bool)

            if img.shape != mask2d.shape:
                row["feature_status"] = "shape_mismatch"
                row["mr_shape"] = str(img.shape)
                row["mask_shape"] = str(mask2d.shape)
                rows.append(row)
                continue

            if mask2d.sum() < 20:
                row["feature_status"] = "too_small"
                rows.append(row)
                continue

            if (mask2d.sum() / mask2d.size) >= 0.50:
                row["feature_status"] = "still_too_large"
                rows.append(row)
                continue

            ring = peritumor_ring(mask2d, radius=15)

            row["feature_status"] = "success"
            row["selected_threshold"] = selected_threshold
            row["selected_frame_index"] = selected_frame_index
            row["row_spacing_mm"] = row_spacing
            row["col_spacing_mm"] = col_spacing
            row["slice_thickness_mm"] = slice_thickness

            voxel_mm3 = row_spacing * col_spacing * slice_thickness
            row["tumor_volume_proxy_mm3_3d"] = float(mask3d.sum() * voxel_mm3)
            row["tumor_volume_proxy_cm3_3d"] = float(mask3d.sum() * voxel_mm3 / 1000.0)
            row["nonzero_frames_3d"] = int((mask3d.reshape(mask3d.shape[0], -1).sum(axis=1) > 0).sum())

            row.update(shape_features(mask2d, row_spacing, col_spacing))

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

            overlay_path = FIGS / f"phase7d_recovered_overlay_case_{case_no:03d}.png"
            make_overlay(
                img,
                mask2d,
                ring,
                skeleton,
                overlay_path,
                f"Phase 7D recovered tumor features | case {case_no} | {row['patient_folder']}",
            )
            overlays.append(overlay_path)

        except Exception as e:
            row["feature_status"] = "failed_exception"
            row["feature_error"] = str(e)[:300]

        rows.append(row)

    out = pd.DataFrame(rows)
    out_csv = TABLES / "phase7d_recovered_tumor_vascular_features.csv"
    out.to_csv(out_csv, index=False)

    make_md(
        out_csv,
        VIEWS / "phase7d_recovered_tumor_vascular_features.md",
        "Phase 7D Recovered Tumor Vascular Features"
    )

    # Figure 200 overlay grid
    if overlays:
        cols = 2
        nrows = int(np.ceil(len(overlays) / cols))
        fig, axes = plt.subplots(nrows, cols, figsize=(14, nrows * 5))
        axes = np.array(axes).reshape(-1)

        for ax, f in zip(axes, overlays):
            img = plt.imread(f)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f.name, fontsize=8)

        for ax in axes[len(overlays):]:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(FIGS / "200_phase7d_recovered_overlay_grid.png", dpi=180)
        plt.close()
    else:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "No overlays created", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(FIGS / "200_phase7d_recovered_overlay_grid.png", dpi=220)
        plt.close()

    # Figure 201 status
    plt.figure(figsize=(8, 5))
    if len(out):
        out["feature_status"].value_counts().sort_values().plot(kind="barh")
        plt.title("Phase 7D Feature Status")
        plt.xlabel("Cases")
    else:
        plt.text(0.5, 0.5, "No recovered pass cases", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "201_phase7d_feature_status.png", dpi=220)
    plt.close()

    # Figure 202 area
    plt.figure(figsize=(8, 5))
    vals = pd.to_numeric(out.get("tumor_area_mm2_2d", pd.Series([])), errors="coerce").dropna()
    if len(vals):
        vals.plot(kind="hist", bins=10)
        plt.title("Phase 7D Tumor Area")
        plt.xlabel("Tumor area mm2")
        plt.ylabel("Cases")
    else:
        plt.text(0.5, 0.5, "No tumor area values", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "202_phase7d_tumor_area.png", dpi=220)
    plt.close()

    # Figure 203 vessel density
    plt.figure(figsize=(8, 5))
    x = pd.to_numeric(out.get("vessel_density_tumor_2d", pd.Series([])), errors="coerce")
    y = pd.to_numeric(out.get("vessel_density_peritumor_2d", pd.Series([])), errors="coerce")
    valid = x.notna() & y.notna()
    if valid.sum():
        plt.scatter(x[valid], y[valid])
        plt.xlabel("Tumor vessel density")
        plt.ylabel("Peritumor vessel density")
        plt.title("Phase 7D Vessel Density")
    else:
        plt.text(0.5, 0.5, "No vessel density values", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "203_phase7d_vessel_density.png", dpi=220)
    plt.close()

    n_total = len(out)
    n_success = int((out["feature_status"] == "success").sum()) if len(out) else 0
    n_failed = n_total - n_success

    report = []
    report.append("# Phase 7D Recovered Tumor Feature Rebuild Report")
    report.append("")
    report.append("This phase rebuilds tumor and vascular features using only Phase 7C fractional tumor candidate pass cases.")
    report.append("")
    report.append("## Main outputs")
    report.append("")
    report.append("- reports/tables/phase7d_recovered_tumor_vascular_features.csv")
    report.append("- reports/table_views/phase7d_recovered_tumor_vascular_features.md")
    report.append("- reports/figures/200_phase7d_recovered_overlay_grid.png")
    report.append("- reports/figures/201_phase7d_feature_status.png")
    report.append("- reports/figures/202_phase7d_tumor_area.png")
    report.append("- reports/figures/203_phase7d_vessel_density.png")
    report.append("")
    report.append("## Main counts")
    report.append("")
    report.append("- Phase 7C pass cases attempted: " + str(n_total))
    report.append("- Feature extraction success cases: " + str(n_success))
    report.append("- Failed or rejected cases: " + str(n_failed))
    report.append("")
    report.append("## Interpretation")
    report.append("")
    if n_success > 0:
        report.append("Recovered fractional tumor masks produced usable tumor and vascular features. These are still a small pilot set and should be visually reviewed before modeling.")
    else:
        report.append("No recovered masks produced usable features. The project should switch to support-data ROI, FTV, PE, or SER maps.")
    report.append("")
    report.append("## Next step")
    report.append("")
    if n_success >= 6:
        report.append("Run Phase 7E to compare recovered tumor features with pCR labels. If both pCR classes are present, run a small feasibility model.")
    else:
        report.append("Increase Phase 7C sample size or search support-data ROI/FTV files before running ML.")

    (REPORTS / "Phase7D_Recovered_Tumor_Feature_Rebuild_Report.md").write_text("\n".join(report))

    dash = REPORTS / "Dashboard.md"
    old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
    dash_add = "\n\n## Phase 7D recovered tumor features\n\n- [Recovered tumor feature report](Phase7D_Recovered_Tumor_Feature_Rebuild_Report.md)\n- [Recovered tumor vascular features](table_views/phase7d_recovered_tumor_vascular_features.md)\n"
    if "Phase 7D recovered tumor features" not in old_dash:
        dash.write_text(old_dash.rstrip() + dash_add)

    readme = REPO / "README.md"
    old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
    readme_add = "\n\n### Phase 7D: recovered tumor feature rebuild\n\nMain outputs:\n\n- reports/Phase7D_Recovered_Tumor_Feature_Rebuild_Report.md\n- reports/tables/phase7d_recovered_tumor_vascular_features.csv\n- reports/figures/200_phase7d_recovered_overlay_grid.png\n"
    if "Phase 7D: recovered tumor feature rebuild" not in old_readme:
        readme.write_text(old_readme.rstrip() + readme_add)

    print("Phase 7D complete.")
    print("Phase 7C pass cases attempted:", n_total)
    print("Feature extraction success cases:", n_success)
    print("Failed or rejected cases:", n_failed)

if __name__ == "__main__":
    main()
