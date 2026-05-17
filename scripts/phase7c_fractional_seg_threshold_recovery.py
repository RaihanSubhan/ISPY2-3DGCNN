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

if not LINK_PATH.exists():
    raise SystemExit("Missing reports/tables/seg_mr_link_table.csv")

links = pd.read_csv(LINK_PATH, dtype=str).fillna("")

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def clean_text(x):
    return "" if x is None else str(x).strip().replace("\n", " ").replace("\r", " ")

def list_dcm(folder):
    out = []
    folder = Path(str(folder))
    if not folder.exists():
        return out

    for root, _, names in os.walk(folder):
        for n in names:
            if n.lower().endswith(".dcm"):
                out.append(str(Path(root) / n))

    return sorted(out)

def normalize01(img):
    img = img.astype(float)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
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
        raise RuntimeError(f"Unexpected SEG pixel array shape: {arr.shape}")

    arr = arr.astype(float)

    return ds, arr

def get_frame_z_positions(ds, n_frames):
    out = []
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
        out.append(z)

    return np.array(out, dtype=float)

def get_segment_text(ds):
    parts = []
    parts.append(clean_text(getattr(ds, "SeriesDescription", "")))
    parts.append(clean_text(getattr(ds, "ContentLabel", "")))
    parts.append(clean_text(getattr(ds, "ContentDescription", "")))
    parts.append(clean_text(getattr(ds, "SegmentationType", "")))

    seq = getattr(ds, "SegmentSequence", None)
    if seq:
        for item in seq:
            parts.append(clean_text(getattr(item, "SegmentLabel", "")))
            parts.append(clean_text(getattr(item, "SegmentDescription", "")))
            try:
                cat = item.SegmentedPropertyCategoryCodeSequence[0].CodeMeaning
                parts.append(clean_text(cat))
            except Exception:
                pass
            try:
                typ = item.SegmentedPropertyTypeCodeSequence[0].CodeMeaning
                parts.append(clean_text(typ))
            except Exception:
                pass

    return " | ".join([p for p in parts if p])

def read_mr_table(series_path, max_files=1800):
    files = list_dcm(series_path)

    if len(files) > max_files:
        idx = np.linspace(0, len(files) - 1, max_files).astype(int)
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
                "instance": str(inst),
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

def candidate_thresholds(arr):
    nonzero = arr[arr > 0]

    if nonzero.size == 0:
        return []

    maxv = float(np.max(nonzero))
    minv = float(np.min(nonzero))

    thresholds = set()

    # Absolute low thresholds
    thresholds.add(minv)
    thresholds.add(1.0)

    # Percent of max
    for frac in [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90]:
        thresholds.add(maxv * frac)

    # Percentile thresholds
    for q in [50, 60, 70, 75, 80, 85, 90, 92, 95, 97, 98, 99, 99.5]:
        thresholds.add(float(np.percentile(nonzero, q)))

    thresholds = sorted([t for t in thresholds if np.isfinite(t) and t > 0])

    # Avoid duplicates that are almost the same
    clean = []
    for t in thresholds:
        if not clean or abs(t - clean[-1]) > 1e-8:
            clean.append(t)

    return clean

def score_threshold(mask3d, threshold, frame_area):
    frame_pixels = mask3d.reshape(mask3d.shape[0], -1).sum(axis=1)
    nonzero = frame_pixels[frame_pixels > 0]

    nonzero_frames = int((frame_pixels > 0).sum())
    max_pixels = int(frame_pixels.max()) if len(frame_pixels) else 0
    median_pixels = float(np.median(nonzero)) if len(nonzero) else 0.0

    max_fraction = float(max_pixels / max(frame_area, 1))
    median_fraction = float(median_pixels / max(frame_area, 1))
    total_fraction = float(mask3d.sum() / max(mask3d.size, 1))

    status = "reject"

    if max_pixels < 20:
        status = "reject_too_small"
    elif max_fraction >= 0.50:
        status = "reject_full_or_large"
    elif total_fraction >= 0.35:
        status = "review_large_total"
    elif nonzero_frames < 1:
        status = "reject_empty"
    else:
        status = "candidate"

    # prefer reasonable tumor-like size, not too tiny, not too large
    score = 0.0

    if status == "candidate":
        score += 100
        score += min(nonzero_frames, 50) * 0.2
        score -= max_fraction * 30
        score -= total_fraction * 10
    elif status == "review_large_total":
        score += 30
        score -= max_fraction * 30
    else:
        score -= 100

    return {
        "threshold": float(threshold),
        "threshold_status": status,
        "score": float(score),
        "nonzero_frames": nonzero_frames,
        "max_frame_pixels": max_pixels,
        "median_nonzero_frame_pixels": median_pixels,
        "max_frame_fraction": max_fraction,
        "median_nonzero_frame_fraction": median_fraction,
        "total_mask_fraction_3d": total_fraction,
        "total_mask_pixels": int(mask3d.sum()),
    }

def match_mr_slice(mask2d, seg_z, mr_table):
    if mr_table.empty:
        return "", np.nan, "no_mr_geometry"

    rows, cols = mask2d.shape

    same_shape = mr_table[(mr_table["rows"] == rows) & (mr_table["cols"] == cols)].copy()
    search = same_shape if len(same_shape) else mr_table.copy()

    if np.isfinite(seg_z):
        zvals = search["z"].astype(float).values
        dz = np.abs(zvals - seg_z)
        j = int(np.argmin(dz))
        method = "z_match_same_shape" if len(same_shape) else "z_match_shape_warning"
        return search.iloc[j]["mr_file"], float(dz[j]), method

    if len(same_shape):
        j = len(same_shape) // 2
        return same_shape.iloc[j]["mr_file"], np.nan, "shape_match_no_seg_z"

    j = len(search) // 2
    return search.iloc[j]["mr_file"], np.nan, "fallback_no_shape_no_seg_z"

def make_overlay(img, mask, title, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("MR")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Recovered tumor candidate")
    axes[1].axis("off")

    axes[2].imshow(img, cmap="gray")
    if img.shape == mask.shape and mask.sum() > 0:
        axes[2].contour(mask, linewidths=1.2)
    axes[2].set_title("MR + recovered contour")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def make_md(csv_path, md_path, title, max_rows=80, max_cols=12):
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
    parser.add_argument("--max-cases", type=int, default=300)
    parser.add_argument("--max-overlays", type=int, default=20)
    args = parser.parse_args()

    work = links.copy()

    for col in ["seg_first_dcm", "matched_mr_series_path", "patient_folder", "study_folder", "seg_series_uid"]:
        if col not in work.columns:
            raise SystemExit(f"Missing column in seg_mr_link_table.csv: {col}")

    work = work[
        (work["seg_first_dcm"].astype(str) != "") &
        (work["matched_mr_series_path"].astype(str) != "")
    ].copy()

    if "link_confidence" in work.columns:
        priority = {"high": 0, "medium": 1, "weak": 2}
        work["link_priority"] = work["link_confidence"].map(priority).fillna(9)
        work = work.sort_values(["link_priority", "patient_folder"])
    else:
        work = work.sort_values("patient_folder")

    work = work.head(args.max_cases)

    case_rows = []
    threshold_rows = []
    overlay_files = []

    for _, row in work.iterrows():
        case_no = len(case_rows) + 1

        patient = row.get("patient_folder", "")
        study = row.get("study_folder", "")
        seg_uid = row.get("seg_series_uid", "")
        seg_file = row.get("seg_first_dcm", "")
        mr_series = row.get("matched_mr_series_path", "")

        case = {
            "case_no": case_no,
            "patient_folder": patient,
            "study_folder": study,
            "seg_series_uid": seg_uid,
            "seg_file": seg_file,
            "matched_mr_series_path": mr_series,
            "recovery_status": "not_started",
        }

        try:
            ds, arr = read_seg(seg_file)

            n_frames, rows, cols = arr.shape
            frame_area = rows * cols

            seg_text = get_segment_text(ds)
            nonzero = arr[arr > 0]

            case["segmentation_type"] = clean_text(getattr(ds, "SegmentationType", ""))
            case["segment_text"] = seg_text
            case["n_frames"] = int(n_frames)
            case["rows"] = int(rows)
            case["cols"] = int(cols)
            case["frame_area"] = int(frame_area)
            case["nonzero_pixel_count"] = int(nonzero.size)
            case["pixel_min_nonzero"] = float(np.min(nonzero)) if nonzero.size else np.nan
            case["pixel_max"] = float(np.max(arr)) if arr.size else np.nan
            case["pixel_mean_nonzero"] = float(np.mean(nonzero)) if nonzero.size else np.nan

            thresholds = candidate_thresholds(arr)

            best = None

            for thr in thresholds:
                mask3d = arr >= thr
                info = score_threshold(mask3d, thr, frame_area)

                row_out = dict(info)
                row_out.update({
                    "case_no": case_no,
                    "patient_folder": patient,
                    "study_folder": study,
                    "seg_series_uid": seg_uid,
                    "seg_file": seg_file,
                })
                threshold_rows.append(row_out)

                if info["threshold_status"] in ["candidate", "review_large_total"]:
                    if best is None or info["score"] > best["score"]:
                        best = dict(info)
                        best["mask3d"] = mask3d

            if best is None:
                case["recovery_status"] = "no_threshold_candidate"
                case_rows.append(case)
                continue

            mask3d = best["mask3d"]
            frame_pixels = mask3d.reshape(mask3d.shape[0], -1).sum(axis=1)
            nz = np.where(frame_pixels > 0)[0]

            if len(nz) == 0:
                case["recovery_status"] = "empty_best_threshold"
                case_rows.append(case)
                continue

            frame_idx = int(nz[np.argmax(frame_pixels[nz])])
            mask2d = mask3d[frame_idx]

            frame_z = get_frame_z_positions(ds, n_frames)
            seg_z = frame_z[frame_idx] if frame_idx < len(frame_z) else np.nan

            mr_table = read_mr_table(mr_series)
            mr_file, z_error, method = match_mr_slice(mask2d, seg_z, mr_table)

            case["selected_threshold"] = best["threshold"]
            case["selected_threshold_status"] = best["threshold_status"]
            case["selected_threshold_score"] = best["score"]
            case["selected_frame_index"] = frame_idx
            case["selected_frame_pixels"] = int(mask2d.sum())
            case["selected_frame_fraction"] = float(mask2d.sum() / max(mask2d.size, 1))
            case["selected_total_mask_pixels"] = best["total_mask_pixels"]
            case["selected_total_mask_fraction_3d"] = best["total_mask_fraction_3d"]
            case["selected_nonzero_frames"] = best["nonzero_frames"]
            case["matched_mr_file"] = mr_file
            case["z_error_mm"] = z_error
            case["mr_match_method"] = method
            case["mr_slices_checked"] = int(len(mr_table))

            if not mr_file:
                case["recovery_status"] = "candidate_no_mr_match"
                case_rows.append(case)
                continue

            img = read_mr_image(mr_file)

            case["mr_shape"] = str(img.shape)
            case["mask_shape"] = str(mask2d.shape)
            case["shape_match"] = "yes" if img.shape == mask2d.shape else "no"

            if img.shape != mask2d.shape:
                case["recovery_status"] = "candidate_shape_mismatch"
            elif mask2d.sum() < 20:
                case["recovery_status"] = "candidate_too_small_after_overlay"
            elif np.isfinite(z_error) and z_error > 5:
                case["recovery_status"] = "candidate_large_z_error"
            else:
                case["recovery_status"] = "fractional_tumor_candidate_pass"

            if len(overlay_files) < args.max_overlays:
                out_img = FIGS / f"phase7c_fractional_overlay_case_{case_no:03d}.png"
                title = f"Phase 7C case {case_no}: {patient} | thr={best['threshold']:.3f} | {case['recovery_status']}"
                make_overlay(img, mask2d, title, out_img)
                overlay_files.append(out_img)

        except Exception as e:
            case["recovery_status"] = "failed_exception"
            case["error"] = str(e)[:300]

        case_rows.append(case)

    case_df = pd.DataFrame(case_rows)
    thr_df = pd.DataFrame(threshold_rows)

    case_csv = TABLES / "phase7c_fractional_tumor_mask_candidate_table.csv"
    thr_csv = TABLES / "phase7c_fractional_threshold_audit.csv"

    case_df.to_csv(case_csv, index=False)
    thr_df.to_csv(thr_csv, index=False)

    make_md(case_csv, VIEWS / "phase7c_fractional_tumor_mask_candidate_table.md", "Phase 7C Fractional Tumor Mask Candidate Table")
    make_md(thr_csv, VIEWS / "phase7c_fractional_threshold_audit.md", "Phase 7C Fractional Threshold Audit")

    # Overlay grid
    if overlay_files:
        cols = 2
        rows_n = int(np.ceil(len(overlay_files) / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(14, rows_n * 5))
        axes = np.array(axes).reshape(-1)

        for ax, f in zip(axes, overlay_files):
            img = plt.imread(f)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f.name, fontsize=8)

        for ax in axes[len(overlay_files):]:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(FIGS / "192_phase7c_fractional_overlay_grid.png", dpi=180)
        plt.close()
    else:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "No overlay files created", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(FIGS / "192_phase7c_fractional_overlay_grid.png", dpi=220)
        plt.close()

    # Status plot
    plt.figure(figsize=(9, 5))
    if len(case_df):
        case_df["recovery_status"].value_counts().sort_values().plot(kind="barh")
        plt.title("Phase 7C Fractional SEG Recovery Status")
        plt.xlabel("Number of cases")
    else:
        plt.text(0.5, 0.5, "No cases", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "190_phase7c_recovery_status.png", dpi=220)
    plt.close()

    # Mask fraction plot
    plt.figure(figsize=(8, 5))
    vals = pd.to_numeric(case_df.get("selected_frame_fraction", pd.Series([])), errors="coerce").dropna()
    if len(vals):
        vals.plot(kind="hist", bins=30)
        plt.axvline(0.50, linestyle="--")
        plt.title("Phase 7C Selected Mask Fraction")
        plt.xlabel("Mask pixels / image pixels")
        plt.ylabel("Cases")
    else:
        plt.text(0.5, 0.5, "No selected mask fractions", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "191_phase7c_selected_mask_fraction.png", dpi=220)
    plt.close()

    n_cases = len(case_df)
    n_pass = int((case_df["recovery_status"] == "fractional_tumor_candidate_pass").sum()) if n_cases else 0
    n_no_candidate = int((case_df["recovery_status"] == "no_threshold_candidate").sum()) if n_cases else 0
    n_review = int(case_df["recovery_status"].astype(str).str.contains("candidate_large|candidate_shape|candidate_no").sum()) if n_cases else 0
    n_failed = int(case_df["recovery_status"].astype(str).str.contains("failed").sum()) if n_cases else 0

    report = []
    report.append("# Phase 7C Fractional SEG Threshold Recovery Report")
    report.append("")
    report.append("This phase tests whether the full-slice DICOM SEG problem can be fixed by thresholding fractional SEG values instead of using all nonzero pixels.")
    report.append("")
    report.append("## Why this was needed")
    report.append("")
    report.append("Phase 7B showed that `arr > 0` selected full-slice analysis masks. Since these SEG files are marked FRACTIONAL, the tumor-like region may require a higher threshold.")
    report.append("")
    report.append("## Main outputs")
    report.append("")
    report.append("- reports/tables/phase7c_fractional_threshold_audit.csv")
    report.append("- reports/tables/phase7c_fractional_tumor_mask_candidate_table.csv")
    report.append("- reports/table_views/phase7c_fractional_threshold_audit.md")
    report.append("- reports/table_views/phase7c_fractional_tumor_mask_candidate_table.md")
    report.append("- reports/figures/190_phase7c_recovery_status.png")
    report.append("- reports/figures/191_phase7c_selected_mask_fraction.png")
    report.append("- reports/figures/192_phase7c_fractional_overlay_grid.png")
    report.append("")
    report.append("## Main counts")
    report.append("")
    report.append("- Cases tested: " + str(n_cases))
    report.append("- Fractional tumor candidate pass cases: " + str(n_pass))
    report.append("- No-threshold-candidate cases: " + str(n_no_candidate))
    report.append("- Review-needed cases: " + str(n_review))
    report.append("- Failed cases: " + str(n_failed))
    report.append("")
    report.append("## Interpretation")
    report.append("")

    if n_pass > 0:
        report.append("Fractional thresholding recovered tumor-like mask candidates. Open the overlay grid and visually inspect whether contours sit on tumor regions.")
    else:
        report.append("Fractional thresholding did not recover tumor-like masks. The current DICOM SEG files should not be used as tumor masks. The next step should search support data for true tumor ROI, FTV, PE, SER, or derived maps.")

    report.append("")
    report.append("## Next step")
    report.append("")
    if n_pass > 0:
        report.append("Run Phase 7D to rebuild tumor and vascular features using only Phase 7C pass cases.")
    else:
        report.append("Run Phase 8A to search local ISPY2 clinical/support folders and IDC/TCIA support downloads for ROI/FTV/PE/SER tumor mask files.")

    (REPORTS / "Phase7C_Fractional_SEG_Threshold_Recovery_Report.md").write_text("\n".join(report))

    # Dashboard update
    dash = REPORTS / "Dashboard.md"
    old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
    dash_add = "\n\n## Phase 7C fractional SEG threshold recovery\n\n- [Fractional SEG recovery report](Phase7C_Fractional_SEG_Threshold_Recovery_Report.md)\n- [Fractional tumor mask candidates](table_views/phase7c_fractional_tumor_mask_candidate_table.md)\n- [Threshold audit](table_views/phase7c_fractional_threshold_audit.md)\n"
    if "Phase 7C fractional SEG threshold recovery" not in old_dash:
        dash.write_text(old_dash.rstrip() + dash_add)

    # README update
    readme = REPO / "README.md"
    old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
    readme_add = "\n\n### Phase 7C: fractional SEG threshold recovery\n\nMain outputs:\n\n- reports/Phase7C_Fractional_SEG_Threshold_Recovery_Report.md\n- reports/tables/phase7c_fractional_tumor_mask_candidate_table.csv\n- reports/figures/192_phase7c_fractional_overlay_grid.png\n"
    if "Phase 7C: fractional SEG threshold recovery" not in old_readme:
        readme.write_text(old_readme.rstrip() + readme_add)

    print("Phase 7C complete.")
    print("Cases tested:", n_cases)
    print("Fractional tumor candidate pass cases:", n_pass)
    print("No-threshold-candidate cases:", n_no_candidate)
    print("Review-needed cases:", n_review)
    print("Failed cases:", n_failed)

if __name__ == "__main__":
    main()
