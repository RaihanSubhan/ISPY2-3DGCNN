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

TUMOR_WORDS = [
    "tumor", "tumour", "lesion", "mass", "target", "cancer",
    "carcinoma", "neoplasm", "enhancing", "ftv", "roi"
]

EXCLUDE_WORDS = [
    "whole", "breast", "background", "body", "skin", "air",
    "normal", "tissue", "fibroglandular", "fgt", "non-tumor",
    "nontumor", "organ", "chest", "thorax"
]

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def clean_text(x):
    return str(x).strip().replace("\n", " ").replace("\r", " ")

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

def get_segment_metadata(ds):
    rows = []

    seq = getattr(ds, "SegmentSequence", None)

    if not seq:
        rows.append({
            "segment_number": "unknown",
            "segment_label": "unknown",
            "segment_description": "",
            "algorithm_type": "",
        })
        return rows

    for item in seq:
        num = clean_text(getattr(item, "SegmentNumber", ""))
        label = clean_text(getattr(item, "SegmentLabel", ""))
        desc = clean_text(getattr(item, "SegmentDescription", ""))
        alg = clean_text(getattr(item, "SegmentAlgorithmType", ""))

        rows.append({
            "segment_number": num,
            "segment_label": label,
            "segment_description": desc,
            "algorithm_type": alg,
        })

    return rows

def get_frame_segment_numbers(ds, n_frames):
    out = []

    per = getattr(ds, "PerFrameFunctionalGroupsSequence", None)

    for i in range(n_frames):
        seg_num = "unknown"

        try:
            fg = per[i]
            seq = getattr(fg, "SegmentIdentificationSequence", None)

            if seq:
                seg_num = clean_text(getattr(seq[0], "ReferencedSegmentNumber", "unknown"))
        except Exception:
            pass

        out.append(seg_num)

    return out

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
        raise RuntimeError(f"Unexpected SEG pixel array shape: {arr.shape}")

    mask_frames = arr > 0

    return ds, mask_frames

def score_segment(label, desc, max_frac, total_frac, nonzero_frames, max_pixels):
    text = (str(label) + " " + str(desc)).lower()

    has_tumor_word = any(w in text for w in TUMOR_WORDS)
    has_exclude_word = any(w in text for w in EXCLUDE_WORDS)

    score = 0.0
    reason = []

    if has_tumor_word:
        score += 100
        reason.append("tumor_label_word")

    if has_exclude_word and not has_tumor_word:
        score -= 100
        reason.append("exclude_label_word")

    if max_pixels < 20:
        score -= 80
        reason.append("too_small")

    if max_frac >= 0.50:
        score -= 120
        reason.append("full_slice_or_too_large")

    if total_frac >= 0.35:
        score -= 40
        reason.append("large_total_fraction")

    if 0.00005 <= max_frac < 0.50:
        score += 25
        reason.append("plausible_mask_fraction")

    if 1 <= nonzero_frames <= 200:
        score += 10
        reason.append("plausible_frame_count")

    return score, ";".join(reason)

def choose_tumor_segment(ds, mask_frames):
    n_frames, rows, cols = mask_frames.shape
    frame_area = rows * cols

    seg_meta = get_segment_metadata(ds)
    frame_seg_nums = get_frame_segment_numbers(ds, n_frames)

    meta_by_num = {str(r["segment_number"]): r for r in seg_meta}

    if all(s == "unknown" for s in frame_seg_nums):
        unique_segments = ["unknown"]
    else:
        unique_segments = sorted(set(frame_seg_nums))

    audit_rows = []
    best = None

    for seg_num in unique_segments:
        frame_idx = [i for i, s in enumerate(frame_seg_nums) if str(s) == str(seg_num)]

        if seg_num == "unknown" and len(frame_idx) == 0:
            frame_idx = list(range(n_frames))

        if len(frame_idx) == 0:
            continue

        seg_mask = np.zeros_like(mask_frames, dtype=bool)
        seg_mask[frame_idx] = mask_frames[frame_idx]

        per_frame_pixels = seg_mask.reshape(n_frames, -1).sum(axis=1)
        nonzero = np.where(per_frame_pixels > 0)[0]

        max_pixels = int(per_frame_pixels.max()) if len(per_frame_pixels) else 0
        max_frac = float(max_pixels / max(frame_area, 1))
        total_frac = float(seg_mask.sum() / max(seg_mask.size, 1))
        nonzero_frames = int(len(nonzero))

        meta = meta_by_num.get(str(seg_num), {})
        label = meta.get("segment_label", "")
        desc = meta.get("segment_description", "")

        score, reason = score_segment(label, desc, max_frac, total_frac, nonzero_frames, max_pixels)

        row = {
            "segment_number": seg_num,
            "segment_label": label,
            "segment_description": desc,
            "algorithm_type": meta.get("algorithm_type", ""),
            "nonzero_frames": nonzero_frames,
            "max_frame_pixels": max_pixels,
            "max_frame_fraction": max_frac,
            "total_mask_pixels": int(seg_mask.sum()),
            "total_mask_fraction_3d": total_frac,
            "score": score,
            "score_reason": reason,
        }

        audit_rows.append(row)

        if best is None or score > best["score"]:
            best = dict(row)
            best["mask3d"] = seg_mask
            best["nonzero_frame_indices"] = nonzero

    return best, pd.DataFrame(audit_rows)

def match_mr_slice(mask2d, seg_z, mr_table):
    if mr_table.empty:
        return "", np.nan, "no_mr_geometry"

    rows, cols = mask2d.shape

    same_shape = mr_table[(mr_table["rows"] == rows) & (mr_table["cols"] == cols)].copy()
    search = same_shape if len(same_shape) else mr_table.copy()

    if np.isfinite(seg_z):
        z_vals = search["z"].astype(float).values
        dz = np.abs(z_vals - seg_z)
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
    axes[0].set_title("MR slice")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Tumor SEG candidate")
    axes[1].axis("off")

    axes[2].imshow(img, cmap="gray")
    if img.shape == mask.shape and mask.sum() > 0:
        axes[2].contour(mask, linewidths=1.2)
    axes[2].set_title("MR + tumor contour")
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

    for _, r in small.iterrows():
        lines.append("| " + " | ".join(clean(v) for v in r.tolist()) + " |")

    md_path.write_text("\n".join(lines))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=80)
    parser.add_argument("--max-overlays", type=int, default=20)
    args = parser.parse_args()

    work = links.copy()

    required = ["seg_first_dcm", "matched_mr_series_path", "patient_folder", "study_folder", "seg_series_uid"]
    for c in required:
        if c not in work.columns:
            raise SystemExit(f"Missing column in seg_mr_link_table.csv: {c}")

    work = work[
        (work["seg_first_dcm"].astype(str) != "") &
        (work["matched_mr_series_path"].astype(str) != "")
    ].copy()

    if "link_confidence" in work.columns:
        order = {"high": 0, "medium": 1, "weak": 2}
        work["link_priority"] = work["link_confidence"].map(order).fillna(9)
        work = work.sort_values(["link_priority", "patient_folder"])
    else:
        work = work.sort_values(["patient_folder"])

    work = work.head(args.max_cases)

    case_rows = []
    segment_rows = []
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
            "link_status": row.get("link_status", ""),
            "link_confidence": row.get("link_confidence", ""),
            "fix_status": "not_started",
        }

        try:
            ds, mask_frames = read_seg(seg_file)
            n_frames, rows, cols = mask_frames.shape

            case["seg_shape"] = str(mask_frames.shape)
            case["seg_frame_area"] = int(rows * cols)

            best, seg_audit = choose_tumor_segment(ds, mask_frames)

            for _, sr in seg_audit.iterrows():
                d = sr.to_dict()
                d.update({
                    "case_no": case_no,
                    "patient_folder": patient,
                    "study_folder": study,
                    "seg_series_uid": seg_uid,
                })
                segment_rows.append(d)

            if best is None:
                case["fix_status"] = "reject_no_segment_candidate"
                case_rows.append(case)
                continue

            case["chosen_segment_number"] = best["segment_number"]
            case["chosen_segment_label"] = best["segment_label"]
            case["chosen_segment_description"] = best["segment_description"]
            case["chosen_segment_score"] = best["score"]
            case["chosen_segment_reason"] = best["score_reason"]
            case["chosen_segment_max_frame_fraction"] = best["max_frame_fraction"]
            case["chosen_segment_total_mask_pixels"] = best["total_mask_pixels"]
            case["chosen_segment_nonzero_frames"] = best["nonzero_frames"]

            if best["max_frame_pixels"] < 20:
                case["fix_status"] = "reject_too_small"
                case_rows.append(case)
                continue

            if best["max_frame_fraction"] >= 0.50:
                case["fix_status"] = "reject_full_slice_mask"
                case_rows.append(case)
                continue

            if best["score"] < 0:
                case["fix_status"] = "review_low_segment_score"
                case_rows.append(case)
                continue

            tumor_mask3d = best["mask3d"]
            frame_pixels = tumor_mask3d.reshape(tumor_mask3d.shape[0], -1).sum(axis=1)
            nz = np.where(frame_pixels > 0)[0]

            if len(nz) == 0:
                case["fix_status"] = "reject_empty_chosen_segment"
                case_rows.append(case)
                continue

            frame_idx = int(nz[np.argmax(frame_pixels[nz])])
            mask2d = tumor_mask3d[frame_idx]

            frame_z = get_frame_z_positions(ds, tumor_mask3d.shape[0])
            seg_z = frame_z[frame_idx] if frame_idx < len(frame_z) else np.nan

            mr_table = read_mr_table(mr_series)
            mr_file, z_error, method = match_mr_slice(mask2d, seg_z, mr_table)

            case["selected_frame_index"] = frame_idx
            case["selected_frame_pixels"] = int(mask2d.sum())
            case["selected_frame_fraction"] = float(mask2d.sum() / max(mask2d.size, 1))
            case["selected_frame_z"] = seg_z
            case["matched_mr_file"] = mr_file
            case["mr_match_method"] = method
            case["z_error_mm"] = z_error
            case["mr_slices_checked"] = int(len(mr_table))

            if not mr_file:
                case["fix_status"] = "review_no_mr_slice_match"
                case_rows.append(case)
                continue

            img = read_mr_image(mr_file)

            case["mr_shape"] = str(img.shape)
            case["mask_shape"] = str(mask2d.shape)
            case["shape_match"] = "yes" if img.shape == mask2d.shape else "no"

            if img.shape != mask2d.shape:
                case["fix_status"] = "review_shape_mismatch"
                case_rows.append(case)
                continue

            if np.isfinite(z_error) and z_error > 5:
                case["fix_status"] = "review_large_z_error"
            else:
                case["fix_status"] = "tumor_mask_candidate_pass"

            if len(overlay_files) < args.max_overlays:
                out_img = FIGS / f"phase7a_tumor_mask_overlay_case_{case_no:03d}.png"
                title = f"Phase 7A case {case_no}: {patient} | {case['fix_status']} | seg={best['segment_label']}"
                make_overlay(img, mask2d, title, out_img)
                overlay_files.append(out_img)

        except Exception as e:
            case["fix_status"] = "failed_exception"
            case["error"] = str(e)[:300]

        case_rows.append(case)

    case_df = pd.DataFrame(case_rows)
    seg_df = pd.DataFrame(segment_rows)

    case_csv = TABLES / "phase7a_tumor_mask_candidate_table.csv"
    seg_csv = TABLES / "phase7a_seg_segment_label_audit.csv"

    case_df.to_csv(case_csv, index=False)
    seg_df.to_csv(seg_csv, index=False)

    make_md(case_csv, VIEWS / "phase7a_tumor_mask_candidate_table.md", "Phase 7A Tumor Mask Candidate Table")
    make_md(seg_csv, VIEWS / "phase7a_seg_segment_label_audit.md", "Phase 7A SEG Segment Label Audit")

    # Overlay grid
    if overlay_files:
        cols = 2
        rows_n = int(np.ceil(len(overlay_files) / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(14, rows_n * 5))
        axes = np.array(axes).reshape(-1)

        for ax, f in zip(axes, overlay_files):
            img = plt.imread(f)
            ax.imshow(img)
            ax.set_title(f.name, fontsize=8)
            ax.axis("off")

        for ax in axes[len(overlay_files):]:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(FIGS / "170_phase7a_tumor_mask_overlay_grid.png", dpi=180)
        plt.close()
    else:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "No overlay images created.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(FIGS / "170_phase7a_tumor_mask_overlay_grid.png", dpi=220)
        plt.close()

    # Status counts
    plt.figure(figsize=(9, 5))
    if len(case_df):
        case_df["fix_status"].value_counts().sort_values().plot(kind="barh")
        plt.title("Phase 7A Tumor Mask Fix Status")
        plt.xlabel("Number of cases")
    else:
        plt.text(0.5, 0.5, "No cases audited", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "171_phase7a_fix_status_counts.png", dpi=220)
    plt.close()

    # Mask fraction
    plt.figure(figsize=(8, 5))
    vals = pd.to_numeric(case_df.get("selected_frame_fraction", pd.Series([])), errors="coerce").dropna()
    if len(vals):
        vals.plot(kind="hist", bins=30)
        plt.axvline(0.50, linestyle="--")
        plt.title("Phase 7A Selected Tumor Mask Fraction")
        plt.xlabel("Mask pixels / image pixels")
        plt.ylabel("Cases")
    else:
        plt.text(0.5, 0.5, "No selected mask fractions", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "172_phase7a_mask_fraction_distribution.png", dpi=220)
    plt.close()

    n_cases = len(case_df)
    n_pass = int((case_df["fix_status"] == "tumor_mask_candidate_pass").sum()) if n_cases else 0
    n_full_slice_reject = int((case_df["fix_status"] == "reject_full_slice_mask").sum()) if n_cases else 0
    n_review = int(case_df["fix_status"].astype(str).str.contains("review").sum()) if n_cases else 0
    n_failed = int(case_df["fix_status"].astype(str).str.contains("failed").sum()) if n_cases else 0

    report = []
    report.append("# Phase 7A Tumor SEG Decoding Fix Report")
    report.append("")
    report.append("This phase fixes the earlier segmentation QC problem by selecting tumor-like DICOM SEG segments instead of blindly using all nonzero SEG frames.")
    report.append("")
    report.append("## Why this fix was needed")
    report.append("")
    report.append("Earlier overlays could look acceptable even when the selected mask covered most or all of the image slice. This phase rejects full-slice masks and records which SEG segment was selected.")
    report.append("")
    report.append("## Main outputs")
    report.append("")
    report.append("- reports/tables/phase7a_seg_segment_label_audit.csv")
    report.append("- reports/tables/phase7a_tumor_mask_candidate_table.csv")
    report.append("- reports/table_views/phase7a_seg_segment_label_audit.md")
    report.append("- reports/table_views/phase7a_tumor_mask_candidate_table.md")
    report.append("- reports/figures/170_phase7a_tumor_mask_overlay_grid.png")
    report.append("- reports/figures/171_phase7a_fix_status_counts.png")
    report.append("- reports/figures/172_phase7a_mask_fraction_distribution.png")
    report.append("")
    report.append("## Main counts")
    report.append("")
    report.append("- Cases audited: " + str(n_cases))
    report.append("- Tumor mask candidate pass cases: " + str(n_pass))
    report.append("- Full-slice masks rejected: " + str(n_full_slice_reject))
    report.append("- Review-needed cases: " + str(n_review))
    report.append("- Failed cases: " + str(n_failed))
    report.append("")
    report.append("## Rule used")
    report.append("")
    report.append("A chosen tumor candidate is rejected if the selected frame mask fraction is >= 0.50, because a tumor mask should not cover the whole image slice.")
    report.append("")
    report.append("## Interpretation")
    report.append("")
    if n_pass > 0:
        report.append("The pipeline found tumor-like mask candidates. These should now be reviewed visually before rebuilding tumor and vascular features.")
    else:
        report.append("No reliable tumor-like mask candidates were found. The SEG label interpretation or MR matching must be reviewed before feature extraction.")
    report.append("")
    report.append("## Next step")
    report.append("")
    report.append("Open the Phase 7A overlay grid in GitHub. If the contours now sit on tumor-like regions, rebuild Phase 6A using the Phase 7A tumor mask table instead of the older segmentation QC table.")

    (REPORTS / "Phase7A_Tumor_SEG_Decoding_Fix_Report.md").write_text("\n".join(report))

    # Overwrite old segmentation QC report with warning so old result is not misleading.
    old_report = []
    old_report.append("# Segmentation QC Report")
    old_report.append("")
    old_report.append("This report has been superseded by Phase 7A.")
    old_report.append("")
    old_report.append("The earlier QC only showed that SEG overlays could be created. Phase 7A adds tumor-segment selection and rejects full-slice masks.")
    old_report.append("")
    old_report.append("Use this file instead:")
    old_report.append("")
    old_report.append("- reports/Phase7A_Tumor_SEG_Decoding_Fix_Report.md")
    old_report.append("- reports/figures/170_phase7a_tumor_mask_overlay_grid.png")
    (REPORTS / "Segmentation_QC_Report.md").write_text("\n".join(old_report))

    dash = REPORTS / "Dashboard.md"
    old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
    dash_add = "\n\n## Phase 7A tumor SEG fix\n\n- [Tumor SEG decoding fix report](Phase7A_Tumor_SEG_Decoding_Fix_Report.md)\n- [Tumor mask candidate table](table_views/phase7a_tumor_mask_candidate_table.md)\n- [SEG segment label audit](table_views/phase7a_seg_segment_label_audit.md)\n"
    if "Phase 7A tumor SEG fix" not in old_dash:
        dash.write_text(old_dash.rstrip() + dash_add)

    readme = REPO / "README.md"
    old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
    readme_add = "\n\n### Phase 7A: tumor SEG decoding fix\n\nMain outputs:\n\n- reports/Phase7A_Tumor_SEG_Decoding_Fix_Report.md\n- reports/tables/phase7a_tumor_mask_candidate_table.csv\n- reports/figures/170_phase7a_tumor_mask_overlay_grid.png\n"
    if "Phase 7A: tumor SEG decoding fix" not in old_readme:
        readme.write_text(old_readme.rstrip() + readme_add)

    print("Phase 7A complete.")
    print("Cases audited:", n_cases)
    print("Tumor mask candidate pass cases:", n_pass)
    print("Full-slice masks rejected:", n_full_slice_reject)
    print("Review-needed cases:", n_review)
    print("Failed cases:", n_failed)

if __name__ == "__main__":
    main()
