from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pydicom
from scipy import ndimage as ndi
from skimage import exposure, morphology, measure

REPO = Path.home() / "ISPY2-3DGCNN"
REPORTS = REPO / "reports"
TABLES = REPORTS / "tables"
FIGS = REPORTS / "figures"
VIEWS = REPORTS / "table_views"

for p in [REPORTS, TABLES, FIGS, VIEWS]:
    p.mkdir(parents=True, exist_ok=True)

CANDIDATE_TABLE = TABLES / "phase7c_fractional_tumor_mask_candidate_table.csv"

if not CANDIDATE_TABLE.exists():
    raise SystemExit("Missing reports/tables/phase7c_fractional_tumor_mask_candidate_table.csv. Run Phase 7C first.")

def normalize01(img):
    img = img.astype(float)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((img - lo) / (hi - lo), 0, 1)

def read_mr_image(path):
    ds = pydicom.dcmread(path, force=True)
    img = ds.pixel_array.astype(float)

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)

    img = img * slope + intercept
    return normalize01(img)

def read_seg_array(path):
    ds = pydicom.dcmread(path, force=True)
    arr = np.asarray(ds.pixel_array)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 4:
        arr = np.squeeze(arr)
        if arr.ndim == 4:
            arr = arr[..., 0]

    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected SEG array shape: {arr.shape}")

    return arr.astype(float)

def preprocess_mri(img):
    img = normalize01(img)
    img = ndi.gaussian_filter(img, sigma=0.7)
    img = exposure.equalize_adapthist(img, clip_limit=0.02)
    img = normalize01(img)
    return img

def preprocess_mask(mask):
    mask = mask.astype(bool)

    if mask.sum() == 0:
        return mask

    mask = morphology.remove_small_objects(mask, min_size=20)
    mask = morphology.binary_closing(mask, morphology.disk(2))
    mask = ndi.binary_fill_holes(mask)

    labels = measure.label(mask)
    if labels.max() > 0:
        props = measure.regionprops(labels)
        largest = max(props, key=lambda p: p.area)
        mask = labels == largest.label

    return mask.astype(bool)

def crop_to_mask_or_center(img, mask, pad=40):
    h, w = img.shape

    if mask.sum() == 0:
        size = min(h, w)
        y0 = max(0, h // 2 - size // 2)
        x0 = max(0, w // 2 - size // 2)
        return img[y0:y0+size, x0:x0+size], mask[y0:y0+size, x0:x0+size]

    ys, xs = np.where(mask)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad)

    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]

def save_without_mask(raw, pre, out_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].imshow(raw, cmap="gray")
    axes[0].set_title("Raw MRI")
    axes[0].axis("off")

    axes[1].imshow(pre, cmap="gray")
    axes[1].set_title("Preprocessed MRI")
    axes[1].axis("off")

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def save_with_mask(raw, pre, raw_mask, pre_mask, out_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].imshow(raw, cmap="gray")
    if raw_mask.sum() > 0:
        axes[0].contour(raw_mask, linewidths=1.2)
    axes[0].set_title("Raw MRI + mask")
    axes[0].axis("off")

    axes[1].imshow(pre, cmap="gray")
    if pre_mask.sum() > 0:
        axes[1].contour(pre_mask, linewidths=1.2)
    axes[1].set_title("Preprocessed MRI + mask")
    axes[1].axis("off")

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def save_four_panel(raw, pre, raw_mask, pre_mask, out_path, title):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    axes[0, 0].imshow(raw, cmap="gray")
    axes[0, 0].set_title("Raw MRI, no mask")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(raw, cmap="gray")
    if raw_mask.sum() > 0:
        axes[0, 1].contour(raw_mask, linewidths=1.2)
    axes[0, 1].set_title("Raw MRI + raw mask")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(pre, cmap="gray")
    axes[1, 0].set_title("Preprocessed MRI, no mask")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pre, cmap="gray")
    if pre_mask.sum() > 0:
        axes[1, 1].contour(pre_mask, linewidths=1.2)
    axes[1, 1].set_title("Preprocessed MRI + mask")
    axes[1, 1].axis("off")

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def make_grid(paths, out_path, title, cols=2):
    paths = [Path(p) for p in paths if Path(p).exists()]

    if len(paths) == 0:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "No images created", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=220)
        plt.close()
        return

    rows = int(np.ceil(len(paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 5))
    axes = np.array(axes).reshape(-1)

    for ax, p in zip(axes, paths):
        img = plt.imread(p)
        ax.imshow(img)
        ax.set_title(p.name, fontsize=7)
        ax.axis("off")

    for ax in axes[len(paths):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def make_md(csv_path, md_path, title, max_rows=80, max_cols=12):
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    small = df.head(max_rows).iloc[:, :max_cols]

    def clean(x):
        x = str(x).replace("|", "\\|").replace("\n", " ")
        return x[:87] + "..." if len(x) > 90 else x

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
    parser.add_argument("--max-cases", type=int, default=40)
    parser.add_argument("--max-grid", type=int, default=12)
    args = parser.parse_args()

    candidates = pd.read_csv(CANDIDATE_TABLE, dtype=str).fillna("")
    work = candidates[candidates["recovery_status"].eq("fractional_tumor_candidate_pass")].copy()

    # Prefer cleaner masks first.
    if "selected_frame_fraction" in work.columns:
        work["selected_frame_fraction_num"] = pd.to_numeric(work["selected_frame_fraction"], errors="coerce")
        work = work.sort_values("selected_frame_fraction_num", ascending=True)

    work = work.head(args.max_cases)

    rows = []
    no_mask_paths = []
    with_mask_paths = []
    four_panel_paths = []

    for _, r in work.iterrows():
        case_no = len(rows) + 1

        row = {
            "case_no": case_no,
            "patient_folder": r.get("patient_folder", ""),
            "study_folder": r.get("study_folder", ""),
            "seg_series_uid": r.get("seg_series_uid", ""),
            "status": "not_started",
        }

        try:
            seg_file = r.get("seg_file", "")
            mr_file = r.get("matched_mr_file", "")
            threshold = float(r.get("selected_threshold", "nan"))
            frame_idx = int(float(r.get("selected_frame_index", 0)))

            if not Path(seg_file).exists():
                row["status"] = "missing_seg_file"
                rows.append(row)
                continue

            if not Path(mr_file).exists():
                row["status"] = "missing_mr_file"
                rows.append(row)
                continue

            seg_arr = read_seg_array(seg_file)
            raw = read_mr_image(mr_file)

            mask3d = seg_arr >= threshold

            if frame_idx < 0 or frame_idx >= mask3d.shape[0]:
                pix = mask3d.reshape(mask3d.shape[0], -1).sum(axis=1)
                frame_idx = int(np.argmax(pix))

            raw_mask = mask3d[frame_idx].astype(bool)

            if raw.shape != raw_mask.shape:
                row["status"] = "shape_mismatch"
                row["raw_shape"] = str(raw.shape)
                row["mask_shape"] = str(raw_mask.shape)
                rows.append(row)
                continue

            pre = preprocess_mri(raw)
            pre_mask = preprocess_mask(raw_mask)

            if pre_mask.sum() < 20:
                row["status"] = "mask_too_small"
                rows.append(row)
                continue

            # Crop only for visualization.
            raw_crop, raw_mask_crop = crop_to_mask_or_center(raw, raw_mask, pad=45)
            pre_crop, pre_mask_crop = crop_to_mask_or_center(pre, pre_mask, pad=45)

            title = f"Stage 10B case {case_no}: {row['patient_folder']}"

            no_mask_path = FIGS / f"stage10b_case_{case_no:03d}_without_mask.png"
            with_mask_path = FIGS / f"stage10b_case_{case_no:03d}_with_mask.png"
            four_panel_path = FIGS / f"stage10b_case_{case_no:03d}_four_panel.png"

            save_without_mask(raw_crop, pre_crop, no_mask_path, title)
            save_with_mask(raw_crop, pre_crop, raw_mask_crop, pre_mask_crop, with_mask_path, title)
            save_four_panel(raw_crop, pre_crop, raw_mask_crop, pre_mask_crop, four_panel_path, title)

            no_mask_paths.append(no_mask_path)
            with_mask_paths.append(with_mask_path)
            four_panel_paths.append(four_panel_path)

            row["status"] = "success"
            row["threshold"] = threshold
            row["frame_index"] = frame_idx
            row["raw_mask_pixels"] = int(raw_mask.sum())
            row["preprocessed_mask_pixels"] = int(pre_mask.sum())
            row["mask_fraction"] = float(pre_mask.sum() / pre_mask.size)
            row["without_mask_figure"] = str(no_mask_path)
            row["with_mask_figure"] = str(with_mask_path)
            row["four_panel_figure"] = str(four_panel_path)

        except Exception as e:
            row["status"] = "failed_exception"
            row["error"] = str(e)[:300]

        rows.append(row)

    out = pd.DataFrame(rows)
    out_csv = TABLES / "stage10b_preprocessed_visual_case_manifest.csv"
    out.to_csv(out_csv, index=False)

    make_md(
        out_csv,
        VIEWS / "stage10b_preprocessed_visual_case_manifest.md",
        "Stage 10B Preprocessed Visual Case Manifest"
    )

    make_grid(
        no_mask_paths[:args.max_grid],
        FIGS / "320_stage10b_without_mask_grid.png",
        "Stage 10B: Raw and preprocessed MRI without mask"
    )

    make_grid(
        with_mask_paths[:args.max_grid],
        FIGS / "321_stage10b_with_mask_grid.png",
        "Stage 10B: Raw and preprocessed MRI with mask"
    )

    make_grid(
        four_panel_paths[:args.max_grid],
        FIGS / "322_stage10b_four_panel_grid.png",
        "Stage 10B: Four-panel preprocessing summary"
    )

    plt.figure(figsize=(8, 5))
    out["status"].value_counts().sort_values().plot(kind="barh")
    plt.title("Stage 10B Visual Preprocessing Status")
    plt.xlabel("Cases")
    plt.tight_layout()
    plt.savefig(FIGS / "323_stage10b_status_counts.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    vals = pd.to_numeric(out.get("mask_fraction", pd.Series([])), errors="coerce").dropna()
    if len(vals):
        vals.plot(kind="hist", bins=20)
        plt.xlabel("Mask fraction")
        plt.ylabel("Cases")
        plt.title("Stage 10B Mask Fraction Distribution")
    else:
        plt.text(0.5, 0.5, "No mask fraction values", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "324_stage10b_mask_fraction_distribution.png", dpi=220)
    plt.close()

    n_total = len(out)
    n_success = int((out["status"] == "success").sum()) if n_total else 0

    report = []
    report.append("# Stage 10B Expanded Preprocessed Visual Output Report")
    report.append("")
    report.append("This stage expands the preprocessing visual output after Stage 10A.")
    report.append("")
    report.append("## Goal")
    report.append("")
    report.append("The goal is to create thesis-ready visual panels showing raw MRI, preprocessed MRI, raw MRI with tumor mask, and preprocessed MRI with preprocessed tumor mask.")
    report.append("")
    report.append("## Main outputs")
    report.append("")
    report.append("- reports/tables/stage10b_preprocessed_visual_case_manifest.csv")
    report.append("- reports/table_views/stage10b_preprocessed_visual_case_manifest.md")
    report.append("- reports/figures/320_stage10b_without_mask_grid.png")
    report.append("- reports/figures/321_stage10b_with_mask_grid.png")
    report.append("- reports/figures/322_stage10b_four_panel_grid.png")
    report.append("- reports/figures/323_stage10b_status_counts.png")
    report.append("- reports/figures/324_stage10b_mask_fraction_distribution.png")
    report.append("- reports/figures/stage10b_case_XXX_without_mask.png")
    report.append("- reports/figures/stage10b_case_XXX_with_mask.png")
    report.append("- reports/figures/stage10b_case_XXX_four_panel.png")
    report.append("")
    report.append("## Main counts")
    report.append("")
    report.append("- Cases attempted: " + str(n_total))
    report.append("- Successful preprocessing visual cases: " + str(n_success))
    report.append("")
    report.append("## Interpretation")
    report.append("")
    report.append("These figures support the visual methods section of the thesis. They show how MRI and tumor-mask candidates change after preprocessing.")
    report.append("")
    report.append("## Limitation")
    report.append("")
    report.append("These figures use recovered fractional tumor-mask candidates. They are suitable for visual methods, but the main pCR result should still be based on official FTV/MRI support data.")
    report.append("")
    report.append("## Next step")
    report.append("")
    report.append("Stage 10C should create the left/right breast-pair and contrast-transport obstacle visualization in a cleaner paper-style format.")

    (REPORTS / "Stage10B_Expanded_Preprocessed_Visual_Output_Report.md").write_text("\n".join(report))

    dash = REPORTS / "Dashboard.md"
    old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
    dash_add = (
        "\n\n## Stage 10B expanded preprocessing visuals\n\n"
        "- [Expanded preprocessing visual report](Stage10B_Expanded_Preprocessed_Visual_Output_Report.md)\n"
        "- [Preprocessed visual case manifest](table_views/stage10b_preprocessed_visual_case_manifest.md)\n"
    )
    if "Stage 10B expanded preprocessing visuals" not in old_dash:
        dash.write_text(old_dash.rstrip() + dash_add)

    readme = REPO / "README.md"
    old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
    readme_add = (
        "\n\n### Stage 10B: expanded preprocessing visual outputs\n\n"
        "Main outputs:\n\n"
        "- reports/Stage10B_Expanded_Preprocessed_Visual_Output_Report.md\n"
        "- reports/figures/320_stage10b_without_mask_grid.png\n"
        "- reports/figures/321_stage10b_with_mask_grid.png\n"
        "- reports/figures/322_stage10b_four_panel_grid.png\n"
    )
    if "Stage 10B: expanded preprocessing visual outputs" not in old_readme:
        readme.write_text(old_readme.rstrip() + readme_add)

    print("Stage 10B complete.")
    print("Cases attempted:", n_total)
    print("Successful cases:", n_success)

if __name__ == "__main__":
    main()
