from pathlib import Path
import os
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pydicom

REPO = Path.home() / "ISPY2-3DGCNN"
REPORTS = REPO / "reports"
TABLES = REPORTS / "tables"
FIGS = REPORTS / "figures"
VIEWS = REPORTS / "table_views"

for p in [REPORTS, TABLES, FIGS, VIEWS]:
    p.mkdir(parents=True, exist_ok=True)

LINK_PATH = TABLES / "seg_mr_link_table.csv"
QC_PATH = TABLES / "phase3c_geometry_matching_qc.csv"

if not LINK_PATH.exists():
    raise SystemExit("Missing reports/tables/seg_mr_link_table.csv")

if not QC_PATH.exists():
    raise SystemExit("Missing reports/tables/phase3c_geometry_matching_qc.csv")

links = pd.read_csv(LINK_PATH, dtype=str).fillna("")
qc = pd.read_csv(QC_PATH, dtype=str).fillna("")

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def list_dcm(folder):
    files = []
    folder = Path(str(folder))
    if not folder.exists():
        return files
    for root, _, names in os.walk(folder):
        for n in names:
            if n.lower().endswith(".dcm"):
                files.append(str(Path(root) / n))
    return sorted(files)

def normalize01(img):
    img = img.astype(float)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        hi = lo + 1
    return np.clip((img - lo) / (hi - lo), 0, 1)

def read_seg(seg_file):
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

    mask = arr > 0
    return ds, mask

def seg_frame_z(ds, n_frames):
    zs = []
    per = getattr(ds, "PerFrameFunctionalGroupsSequence", None)

    for i in range(n_frames):
        z = np.nan
        try:
            fg = per[i]
            pos = getattr(fg, "PlanePositionSequence", None)
            if pos:
                ipp = getattr(pos[0], "ImagePositionPatient", None)
                if ipp is not None and len(ipp) >= 3:
                    z = safe_float(ipp[2])
        except Exception:
            pass
        zs.append(z)

    return np.array(zs, dtype=float)

def read_mr_table(series_path, max_files=1800):
    files = list_dcm(series_path)
    if len(files) > max_files:
        idx = np.linspace(0, len(files)-1, max_files).astype(int)
        files = [files[i] for i in idx]

    rows = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            ipp = getattr(ds, "ImagePositionPatient", None)
            inst = getattr(ds, "InstanceNumber", "")
            if ipp is not None and len(ipp) >= 3:
                z = safe_float(ipp[2])
            else:
                z = safe_float(inst)

            rows.append({
                "mr_file": f,
                "z": z,
                "rows": int(getattr(ds, "Rows", 0) or 0),
                "cols": int(getattr(ds, "Columns", 0) or 0),
            })
        except Exception:
            pass

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["z"]).sort_values("z")
    return df

def read_mr_image(mr_file):
    ds = pydicom.dcmread(mr_file, force=True)
    img = ds.pixel_array.astype(float)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    img = img * slope + intercept
    return normalize01(img)

def pick_mr_slice(mask2d, seg_z_value, mr_table):
    if mr_table.empty:
        return "", np.nan, "no_mr_geometry"

    rows, cols = mask2d.shape

    same_shape = mr_table[(mr_table["rows"] == rows) & (mr_table["cols"] == cols)].copy()
    search = same_shape if len(same_shape) > 0 else mr_table.copy()

    if np.isfinite(seg_z_value):
        dz = np.abs(search["z"].astype(float).values - seg_z_value)
        j = int(np.argmin(dz))
        return search.iloc[j]["mr_file"], float(dz[j]), "z_match_same_shape" if len(same_shape) > 0 else "z_match_shape_warning"

    if len(same_shape) > 0:
        return same_shape.iloc[len(same_shape)//2]["mr_file"], np.nan, "shape_match_no_seg_z"

    return mr_table.iloc[len(mr_table)//2]["mr_file"], np.nan, "fallback_no_shape_no_seg_z"

def overlay_figure(img, mask, title, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("MR slice")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("SEG mask")
    axes[1].axis("off")

    axes[2].imshow(img, cmap="gray")
    if mask.shape == img.shape and mask.sum() > 0:
        axes[2].contour(mask, linewidths=1.2)
    axes[2].set_title("MR + SEG contour")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def make_md(csv_path, md_path, title, max_rows=50, max_cols=12):
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

    merged = qc.merge(
        links,
        on=["patient_folder", "study_folder", "seg_series_uid"],
        how="left",
        suffixes=("_qc", "_link")
    )

    # Prefer geometry-ready cases, then warning cases.
    priority = {
        "geometry_ready": 0,
        "z_ready_shape_warning": 1,
        "geometry_warning": 2,
        "no_seg_z_geometry": 3,
        "no_mr_z_geometry": 4,
        "failed": 5,
    }
    merged["priority"] = merged["status"].map(priority).fillna(9)

    # Need actual file paths.
    merged = merged[
        (merged["seg_first_dcm"].astype(str) != "") &
        (merged["matched_mr_series_path"].astype(str) != "")
    ].copy()

    merged = merged.sort_values(["priority", "patient_folder"]).head(args.max_cases)

    audit_rows = []
    overlay_paths = []

    for i, row in merged.iterrows():
        case_no = len(audit_rows) + 1
        patient = row.get("patient_folder", "")
        status = row.get("status", "")
        seg_file = row.get("seg_first_dcm", "")
        mr_series = row.get("matched_mr_series_path", "")

        audit = {
            "case_no": case_no,
            "patient_folder": patient,
            "study_folder": row.get("study_folder", ""),
            "seg_series_uid": row.get("seg_series_uid", ""),
            "phase3c_status": status,
            "seg_file": seg_file,
            "matched_mr_series_path": mr_series,
            "qc_status": "not_started",
        }

        try:
            ds, mask3d = read_seg(seg_file)
            frame_pixels = mask3d.reshape(mask3d.shape[0], -1).sum(axis=1)
            nz = np.where(frame_pixels > 0)[0]

            audit["seg_shape"] = str(mask3d.shape)
            audit["seg_total_pixels"] = int(mask3d.sum())
            audit["seg_nonzero_frames"] = int(len(nz))

            if len(nz) == 0:
                audit["qc_status"] = "fail_empty_seg"
                audit_rows.append(audit)
                continue

            frame_idx = int(nz[np.argmax(frame_pixels[nz])])
            mask2d = mask3d[frame_idx]

            zs = seg_frame_z(ds, mask3d.shape[0])
            seg_z_value = zs[frame_idx] if frame_idx < len(zs) else np.nan

            mr_table = read_mr_table(mr_series)
            mr_file, z_error, match_method = pick_mr_slice(mask2d, seg_z_value, mr_table)

            audit["selected_seg_frame"] = frame_idx
            audit["selected_seg_frame_pixels"] = int(mask2d.sum())
            audit["seg_frame_z"] = seg_z_value
            audit["matched_mr_file"] = mr_file
            audit["z_error_mm"] = z_error
            audit["match_method"] = match_method
            audit["mr_slices_checked"] = int(len(mr_table))

            if not mr_file:
                audit["qc_status"] = "fail_no_mr_match"
                audit_rows.append(audit)
                continue

            img = read_mr_image(mr_file)

            audit["mr_shape"] = str(img.shape)
            audit["mask_shape"] = str(mask2d.shape)
            audit["shape_match"] = "yes" if img.shape == mask2d.shape else "no"

            mask_fraction = float(mask2d.sum() / max(mask2d.size, 1))
            audit["mask_fraction_2d"] = mask_fraction

            if img.shape != mask2d.shape:
                audit["qc_status"] = "review_shape_mismatch"
            elif mask2d.sum() < 20:
                audit["qc_status"] = "review_tiny_mask"
            elif np.isfinite(z_error) and z_error > 5:
                audit["qc_status"] = "review_large_z_error"
            else:
                audit["qc_status"] = "visual_qc_pass_candidate"

            out_case = FIGS / f"segmentation_qc_case_{case_no:03d}.png"
            title = f"Case {case_no}: {patient} | {audit['qc_status']} | {match_method}"
            overlay_figure(img, mask2d, title, out_case)
            overlay_paths.append(str(out_case))

        except Exception as e:
            audit["qc_status"] = "failed_exception"
            audit["error"] = str(e)[:300]

        audit_rows.append(audit)

    audit_df = pd.DataFrame(audit_rows)
    audit_csv = TABLES / "segmentation_qc_case_audit.csv"
    audit_df.to_csv(audit_csv, index=False)

    make_md(
        audit_csv,
        VIEWS / "segmentation_qc_case_audit.md",
        "Segmentation QC Case Audit"
    )

    # Overlay grid
    image_files = [Path(p) for p in overlay_paths if Path(p).exists()]
    n = len(image_files)

    if n > 0:
        cols = 2
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 5))
        axes = np.array(axes).reshape(-1)

        for ax, img_path in zip(axes, image_files):
            img = plt.imread(img_path)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(img_path.name, fontsize=8)

        for ax in axes[len(image_files):]:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(FIGS / "130_segmentation_qc_overlay_grid.png", dpi=180)
        plt.close()
    else:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "No overlay images were created.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(FIGS / "130_segmentation_qc_overlay_grid.png", dpi=220)
        plt.close()

    # Status counts
    plt.figure(figsize=(9, 5))
    if len(audit_df) > 0:
        audit_df["qc_status"].value_counts().sort_values().plot(kind="barh")
        plt.title("Segmentation QC Status Counts")
        plt.xlabel("Number of cases")
    else:
        plt.text(0.5, 0.5, "No audit rows", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "131_segmentation_qc_status_counts.png", dpi=220)
    plt.close()

    # Volume proxy / pixel count
    plt.figure(figsize=(8, 5))
    vals = pd.to_numeric(audit_df.get("seg_total_pixels", pd.Series([])), errors="coerce").dropna()
    if len(vals) > 0:
        vals.plot(kind="hist", bins=20)
        plt.title("SEG Total Pixel Count Distribution")
        plt.xlabel("Total SEG pixels")
        plt.ylabel("Number of cases")
    else:
        plt.text(0.5, 0.5, "No SEG pixel values", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "132_segmentation_qc_volume_proxy.png", dpi=220)
    plt.close()

    n_cases = len(audit_df)
    n_pass = int((audit_df["qc_status"] == "visual_qc_pass_candidate").sum()) if n_cases else 0
    n_review = int(audit_df["qc_status"].astype(str).str.contains("review").sum()) if n_cases else 0
    n_fail = int(audit_df["qc_status"].astype(str).str.contains("fail|failed").sum()) if n_cases else 0

    report = []
    report.append("# Segmentation QC Report")
    report.append("")
    report.append("This report audits whether the DICOM SEG masks are visually and geometrically usable for tumor biomarker extraction.")
    report.append("")
    report.append("## Important conclusion")
    report.append("")
    report.append("This step verifies existing DICOM SEG masks. It does not train a new segmentation model.")
    report.append("")
    report.append("## Main outputs")
    report.append("")
    report.append("- reports/tables/segmentation_qc_case_audit.csv")
    report.append("- reports/table_views/segmentation_qc_case_audit.md")
    report.append("- reports/figures/130_segmentation_qc_overlay_grid.png")
    report.append("- reports/figures/131_segmentation_qc_status_counts.png")
    report.append("- reports/figures/132_segmentation_qc_volume_proxy.png")
    report.append("- reports/figures/segmentation_qc_case_001.png and related case images")
    report.append("")
    report.append("## Main counts")
    report.append("")
    report.append("- Cases audited: " + str(n_cases))
    report.append("- Visual QC pass candidates: " + str(n_pass))
    report.append("- Review-needed cases: " + str(n_review))
    report.append("- Failed cases: " + str(n_fail))
    report.append("")
    report.append("## Interpretation")
    report.append("")
    if n_pass >= max(3, n_cases // 2):
        report.append("The segmentation workflow appears usable for pilot analysis, but full manual review is still needed before final claims.")
    else:
        report.append("The segmentation workflow needs review before final tumor biomarker claims. Use the overlay figures to identify shape mismatch, empty masks, or z-position mismatch.")
    report.append("")
    report.append("## Next action")
    report.append("")
    report.append("Open the overlay grid and each case image in GitHub. Check whether the contour falls inside the breast tumor region and not outside the image or on the wrong anatomy.")
    report.append("")
    report.append("If many overlays are unclear, improve MR series selection or SEG-to-MR matching before extracting final biomarkers.")

    (REPORTS / "Segmentation_QC_Report.md").write_text("\n".join(report))

    # Dashboard update
    dash = REPORTS / "Dashboard.md"
    old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
    add = "\n\n## Segmentation QC\n\n- [Segmentation QC report](Segmentation_QC_Report.md)\n- [Segmentation QC audit table](table_views/segmentation_qc_case_audit.md)\n"
    if "Segmentation QC report" not in old_dash:
        dash.write_text(old_dash.rstrip() + add)

    # README update
    readme = REPO / "README.md"
    old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
    add_readme = "\n\n### Segmentation QC fix\n\nMain outputs:\n\n- reports/Segmentation_QC_Report.md\n- reports/tables/segmentation_qc_case_audit.csv\n- reports/figures/130_segmentation_qc_overlay_grid.png\n"
    if "Segmentation QC fix" not in old_readme:
        readme.write_text(old_readme.rstrip() + add_readme)

    print("Segmentation QC fix complete.")
    print("Cases audited:", n_cases)
    print("Visual QC pass candidates:", n_pass)
    print("Review-needed cases:", n_review)
    print("Failed cases:", n_fail)

if __name__ == "__main__":
    main()
