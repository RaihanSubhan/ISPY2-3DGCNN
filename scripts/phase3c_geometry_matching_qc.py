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

TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
VIEWS.mkdir(parents=True, exist_ok=True)


def list_dcm(folder):
    out = []
    folder = Path(folder)
    if not folder.exists():
        return out
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".dcm"):
                out.append(str(Path(root) / f))
    return sorted(out)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def read_mr_slice_table(series_path, max_slices=1500):
    files = list_dcm(series_path)
    if len(files) > max_slices:
        idx = np.linspace(0, len(files) - 1, max_slices).astype(int)
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
                "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
            })
        except Exception:
            pass

    tab = pd.DataFrame(rows)
    if not tab.empty:
        tab = tab.dropna(subset=["z"]).sort_values("z")
    return tab


def get_seg_frame_info(ds, n_frames):
    zs = []
    segment_nums = []

    per = getattr(ds, "PerFrameFunctionalGroupsSequence", None)

    for i in range(n_frames):
        z = np.nan
        seg_num = ""

        try:
            fg = per[i]

            pos = getattr(fg, "PlanePositionSequence", None)
            if pos:
                ipp = getattr(pos[0], "ImagePositionPatient", None)
                if ipp is not None and len(ipp) >= 3:
                    z = safe_float(ipp[2])

            segid = getattr(fg, "SegmentIdentificationSequence", None)
            if segid:
                seg_num = str(getattr(segid[0], "ReferencedSegmentNumber", ""))
        except Exception:
            pass

        zs.append(z)
        segment_nums.append(seg_num)

    return np.array(zs, dtype=float), segment_nums


def get_seg_spacing(ds):
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

    return row_spacing, col_spacing, slice_thickness


def normalize_img(x):
    x = x.astype(float)
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((x - lo) / (hi - lo), 0, 1)


def make_overlay(mr_file, mask2d, out_path, title):
    try:
        ds = pydicom.dcmread(mr_file, force=True)
        img = ds.pixel_array.astype(float)
        img = normalize_img(img)

        plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap="gray")
        if mask2d.shape == img.shape and mask2d.sum() > 0:
            plt.contour(mask2d, linewidths=1)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=220)
        plt.close()
        return True
    except Exception:
        return False


def process_one(row, make_example=False):
    out = {
        "patient_folder": row.get("patient_folder", ""),
        "study_folder": row.get("study_folder", ""),
        "seg_series_uid": row.get("seg_series_uid", ""),
        "matched_mr_series_uid": row.get("matched_mr_series_uid", ""),
        "link_status": row.get("link_status", ""),
        "link_confidence": row.get("link_confidence", ""),
        "status": "not_started",
    }

    seg_file = str(row.get("seg_first_dcm", ""))
    mr_series = str(row.get("matched_mr_series_path", ""))

    if not seg_file or not Path(seg_file).exists():
        out["status"] = "missing_seg_file"
        return out

    if not mr_series or not Path(mr_series).exists():
        out["status"] = "missing_mr_series"
        return out

    try:
        ds = pydicom.dcmread(seg_file, force=True)
        arr = np.squeeze(np.asarray(ds.pixel_array))

        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        elif arr.ndim == 4:
            arr = np.squeeze(arr)
            if arr.ndim == 4:
                arr = arr[..., 0]

        if arr.ndim != 3:
            out["status"] = "bad_seg_shape"
            out["seg_shape"] = str(arr.shape)
            return out

        mask = arr > 0
        n_frames, seg_rows, seg_cols = mask.shape
        frame_pixels = mask.reshape(n_frames, -1).sum(axis=1)
        nonzero_idx = np.where(frame_pixels > 0)[0]

        out["seg_frames"] = int(n_frames)
        out["seg_rows"] = int(seg_rows)
        out["seg_cols"] = int(seg_cols)
        out["nonzero_seg_frames"] = int(len(nonzero_idx))
        out["total_mask_pixels"] = int(mask.sum())

        rs, cs, th = get_seg_spacing(ds)
        out["row_spacing_mm"] = rs
        out["col_spacing_mm"] = cs
        out["slice_thickness_mm"] = th
        out["volume_proxy_mm3"] = float(mask.sum() * rs * cs * th)

        seg_z, seg_nums = get_seg_frame_info(ds, n_frames)
        mr_tab = read_mr_slice_table(mr_series)

        out["mr_slices_with_z"] = int(len(mr_tab))

        if len(nonzero_idx) == 0:
            out["status"] = "seg_empty"
            return out

        if mr_tab.empty:
            out["status"] = "no_mr_z_geometry"
            return out

        usable = []
        for i in nonzero_idx:
            if np.isfinite(seg_z[i]):
                usable.append(i)

        out["nonzero_frames_with_seg_z"] = int(len(usable))

        if len(usable) == 0:
            out["status"] = "no_seg_z_geometry"
            return out

        mr_z = mr_tab["z"].values.astype(float)

        abs_errors = []
        shape_matches = 0
        matched_files = []
        matched_frames = []

        for i in usable:
            dz = np.abs(mr_z - seg_z[i])
            j = int(np.argmin(dz))
            err = float(dz[j])
            abs_errors.append(err)
            matched_files.append(mr_tab.iloc[j]["mr_file"])
            matched_frames.append(i)

            if int(mr_tab.iloc[j]["rows"]) == seg_rows and int(mr_tab.iloc[j]["cols"]) == seg_cols:
                shape_matches += 1

        abs_errors = np.array(abs_errors, dtype=float)

        out["matched_nonzero_frames"] = int(len(abs_errors))
        out["shape_matched_frames"] = int(shape_matches)
        out["mean_abs_z_error_mm"] = float(np.mean(abs_errors))
        out["median_abs_z_error_mm"] = float(np.median(abs_errors))
        out["max_abs_z_error_mm"] = float(np.max(abs_errors))
        out["pct_frames_within_1mm"] = float(np.mean(abs_errors <= 1.0))
        out["pct_frames_within_2mm"] = float(np.mean(abs_errors <= 2.0))
        out["pct_frames_within_5mm"] = float(np.mean(abs_errors <= 5.0))

        if shape_matches == len(abs_errors) and float(np.mean(abs_errors <= 2.0)) >= 0.90:
            out["status"] = "geometry_ready"
        elif float(np.mean(abs_errors <= 2.0)) >= 0.90:
            out["status"] = "z_ready_shape_warning"
        else:
            out["status"] = "geometry_warning"

        if make_example and len(matched_files) > 0:
            k = int(nonzero_idx[np.argmax(frame_pixels[nonzero_idx])])
            if k in matched_frames:
                idx = matched_frames.index(k)
                mask2d = mask[k]
                ok = make_overlay(
                    matched_files[idx],
                    mask2d,
                    FIGS / "43_phase3c_example_overlay.png",
                    "Phase 3C geometry-matched MR and SEG overlay",
                )
                out["example_overlay_created"] = "yes" if ok else "no"

        return out

    except Exception as e:
        out["status"] = "failed"
        out["error"] = str(e)[:300]
        return out


def md_table(csv_path, md_path, max_rows=50, max_cols=10):
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    small = df.head(max_rows).iloc[:, :max_cols]

    def clean(x):
        x = str(x).replace("|", "\\|").replace("\n", " ")
        if len(x) > 80:
            x = x[:77] + "..."
        return x

    lines = []
    lines.append("# " + csv_path.name)
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
    parser.add_argument("--max-cases", type=int, default=120)
    args = parser.parse_args()

    src = TABLES / "phase3b_tumor_peritumor_features_pilot.csv"
    if not src.exists():
        raise SystemExit("Missing Phase 3B table. Run Phase 3B first.")

    df = pd.read_csv(src, dtype=str).fillna("")

    if "decode_status" in df.columns:
        work = df[df["decode_status"].eq("success")].copy()
    else:
        work = df.copy()

    if work.empty:
        work = df.copy()

    work = work.head(args.max_cases)

    rows = []
    example_done = False

    for _, row in work.iterrows():
        result = process_one(row, make_example=(not example_done))
        if result.get("example_overlay_created", "") == "yes":
            example_done = True
        rows.append(result)

    qc = pd.DataFrame(rows)
    out_csv = TABLES / "phase3c_geometry_matching_qc.csv"
    qc.to_csv(out_csv, index=False)

    plt.figure(figsize=(8, 5))
    qc["status"].fillna("unknown").value_counts().sort_values().plot(kind="barh")
    plt.title("Phase 3C Geometry Matching Status")
    plt.xlabel("Number of cases")
    plt.tight_layout()
    plt.savefig(FIGS / "40_phase3c_match_status.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    vals = pd.to_numeric(qc.get("mean_abs_z_error_mm", pd.Series([])), errors="coerce").dropna()
    if len(vals) > 0:
        vals.plot(kind="hist", bins=30)
        plt.xlabel("Mean absolute z error, mm")
        plt.ylabel("Cases")
        plt.title("Phase 3C Mean Z-Matching Error")
    else:
        plt.text(0.5, 0.5, "No z-error values available", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "41_phase3c_z_error_hist.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    vals = pd.to_numeric(qc.get("pct_frames_within_2mm", pd.Series([])), errors="coerce").dropna()
    if len(vals) > 0:
        vals.plot(kind="hist", bins=20)
        plt.xlabel("Fraction of frames within 2 mm")
        plt.ylabel("Cases")
        plt.title("Phase 3C Frames Within 2 mm")
    else:
        plt.text(0.5, 0.5, "No within-2mm values available", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "42_phase3c_within_2mm.png", dpi=220)
    plt.close()

    if not (FIGS / "43_phase3c_example_overlay.png").exists():
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "No example overlay was created.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(FIGS / "43_phase3c_example_overlay.png", dpi=220)
        plt.close()

    md_table(out_csv, VIEWS / "phase3c_geometry_matching_qc.md")

    n = len(qc)
    n_ready = int((qc["status"] == "geometry_ready").sum()) if "status" in qc.columns else 0
    n_warn = int(qc["status"].astype(str).str.contains("warning").sum()) if "status" in qc.columns else 0
    n_failed = int((qc["status"] == "failed").sum()) if "status" in qc.columns else 0

    summary = [
        "# Phase 3C Geometry Matching QC Summary",
        "",
        "This phase checks whether decoded SEG frames can be matched to MR slices using DICOM z-position geometry.",
        "",
        "## Main outputs",
        "",
        "- reports/tables/phase3c_geometry_matching_qc.csv",
        "- reports/table_views/phase3c_geometry_matching_qc.md",
        "- reports/figures/40_phase3c_match_status.png",
        "- reports/figures/41_phase3c_z_error_hist.png",
        "- reports/figures/42_phase3c_within_2mm.png",
        "- reports/figures/43_phase3c_example_overlay.png",
        "",
        "## Main counts",
        "",
        "- Cases checked: " + str(n),
        "- Geometry-ready cases: " + str(n_ready),
        "- Warning cases: " + str(n_warn),
        "- Failed cases: " + str(n_failed),
        "",
        "## Interpretation",
        "",
        "This is a QC step before final biomarker extraction. Geometry-ready cases can move toward verified tumor volume, shape, radial signal, and vesselness extraction.",
        "",
        "## Next step",
        "",
        "Phase 3D should compute final verified tumor and peritumor biomarkers only for geometry-ready cases.",
    ]

    (REPORTS / "Phase3C_Geometry_Matching_QC_Summary.md").write_text("\n".join(summary))

    dash_path = REPORTS / "Dashboard.md"
    old = dash_path.read_text() if dash_path.exists() else "# ISPY2 4D Atlas Dashboard\n"
    add = "\n\n## Phase 3C table view\n\n- [Phase 3C geometry matching QC](table_views/phase3c_geometry_matching_qc.md)\n"
    if "Phase 3C geometry matching QC" not in old:
        dash_path.write_text(old.rstrip() + add)

    print("Phase 3C complete.")
    print("Cases checked:", n)
    print("Geometry-ready cases:", n_ready)
    print("Warning cases:", n_warn)
    print("Failed cases:", n_failed)


if __name__ == "__main__":
    main()
