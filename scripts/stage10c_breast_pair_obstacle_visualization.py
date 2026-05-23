from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pydicom
from scipy import ndimage as ndi
from skimage import exposure, morphology, measure, filters

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
    return normalize01(img)

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

def crop_to_breast_pair(img, mask, pad=30):
    """
    Keep a wide field around the mask, but do not crop too tightly.
    This makes the left/right breast-pair display more readable.
    """
    h, w = img.shape

    if mask.sum() == 0:
        return img, mask

    ys, xs = np.where(mask)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad)

    # keep full width to preserve left/right breast pairing
    x0 = 0
    x1 = w

    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]

def split_left_right(img, mask):
    h, w = img.shape
    mid = w // 2

    left_img = img[:, :mid]
    right_img = img[:, mid:]

    left_mask = mask[:, :mid]
    right_mask = mask[:, mid:]

    pair_img = np.concatenate([left_img, right_img], axis=1)
    pair_mask = np.concatenate([left_mask, right_mask], axis=1)

    return left_img, right_img, left_mask, right_mask, pair_img, pair_mask

def obstacle_proxy_map(img, mask):
    """
    This is a visualization proxy only.
    It represents how an abnormal tumor region may disturb local contrast-transport appearance.
    It is not direct blood-flow, milk-flow, or fluid-flow measurement.
    """
    mask = mask.astype(bool)

    if mask.sum() == 0:
        return normalize01(img)

    dist = ndi.distance_transform_edt(~mask)
    obstacle = np.exp(-dist / 18.0)

    grad_y, grad_x = np.gradient(img)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag = normalize01(grad_mag)

    proxy = normalize01(0.70 * img + 0.30 * grad_mag)
    proxy = proxy * (1.0 - 0.65 * obstacle)
    proxy[mask] = np.percentile(proxy, 5)

    return normalize01(proxy)

def vector_field_for_display(img, mask, step=24):
    h, w = img.shape
    yy, xx = np.mgrid[0:h:step, 0:w:step]

    mask = mask.astype(bool)
    dist = ndi.distance_transform_edt(~mask)
    obstacle = np.exp(-dist / 22.0)

    # arrows generally move from left to right, but shrink near tumor
    mag = 1.0 - 0.85 * obstacle[yy, xx]
    mag = np.clip(mag, 0.05, 1.0)

    u = mag
    v = np.zeros_like(u)

    return xx, yy, u, v

def save_breast_pair(img, mask, out_path, title):
    left_img, right_img, left_mask, right_mask, pair_img, pair_mask = split_left_right(img, mask)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(left_img, cmap="gray")
    if left_mask.sum() > 0:
        axes[0].contour(left_mask, linewidths=1.0)
    axes[0].set_title("Left half")
    axes[0].axis("off")

    axes[1].imshow(right_img, cmap="gray")
    if right_mask.sum() > 0:
        axes[1].contour(right_mask, linewidths=1.0)
    axes[1].set_title("Right half")
    axes[1].axis("off")

    axes[2].imshow(pair_img, cmap="gray")
    if pair_mask.sum() > 0:
        axes[2].contour(pair_mask, linewidths=1.2)
    axes[2].set_title("Single breast-pair image")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def save_obstacle_proxy(img, mask, out_path, title):
    proxy = obstacle_proxy_map(img, mask)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Preprocessed MRI")
    axes[0].axis("off")

    axes[1].imshow(img, cmap="gray")
    if mask.sum() > 0:
        axes[1].contour(mask, linewidths=1.2)
    axes[1].set_title("Tumor mask overlay")
    axes[1].axis("off")

    axes[2].imshow(proxy, cmap="gray")
    if mask.sum() > 0:
        axes[2].contour(mask, linewidths=1.0)
    axes[2].set_title("Contrast-transport obstacle proxy")
    axes[2].axis("off")

    fig.suptitle(title + "\nProxy only, not measured blood/milk/fluid flow", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def save_transport_field(img, mask, out_path, title):
    proxy = obstacle_proxy_map(img, mask)
    xx, yy, u, v = vector_field_for_display(img, mask, step=24)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(img, cmap="gray")
    if mask.sum() > 0:
        axes[0].contour(mask, linewidths=1.2)
    axes[0].set_title("Tumor region")
    axes[0].axis("off")

    axes[1].imshow(proxy, cmap="gray")
    axes[1].set_title("Obstacle proxy map")
    axes[1].axis("off")

    axes[2].imshow(img, cmap="gray")
    axes[2].quiver(xx, yy, u, v, angles="xy", scale_units="xy", scale=0.12)
    if mask.sum() > 0:
        axes[2].contour(mask, linewidths=1.2)
    axes[2].set_title("Toy transport field")
    axes[2].axis("off")

    fig.suptitle(title + "\nArrows shrink near tumor to visualize possible contrast-transport disturbance", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def make_grid(paths, out_path, title, cols=2):
    paths = [Path(p) for p in paths if Path(p).exists()]

    if len(paths) == 0:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "No images created", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=220)
        plt.close()
        return

    rows = int(np.ceil(len(paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 5))
    axes = np.array(axes).reshape(-1)

    for ax, p in zip(axes, paths):
        image = plt.imread(p)
        ax.imshow(image)
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

    if "selected_frame_fraction" in work.columns:
        work["selected_frame_fraction_num"] = pd.to_numeric(work["selected_frame_fraction"], errors="coerce")
        work = work.sort_values("selected_frame_fraction_num", ascending=True)

    work = work.head(args.max_cases)

    rows = []
    breast_pair_paths = []
    obstacle_paths = []
    transport_paths = []

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
            pre = preprocess_mri(raw)

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

            mask = preprocess_mask(raw_mask)

            if mask.sum() < 20:
                row["status"] = "mask_too_small"
                rows.append(row)
                continue

            # Keep width, crop only in y direction around tumor region.
            pre_crop, mask_crop = crop_to_breast_pair(pre, mask, pad=35)

            title = f"Stage 10C case {case_no}: {row['patient_folder']}"

            breast_pair_path = FIGS / f"stage10c_case_{case_no:03d}_breast_pair.png"
            obstacle_path = FIGS / f"stage10c_case_{case_no:03d}_obstacle_proxy.png"
            transport_path = FIGS / f"stage10c_case_{case_no:03d}_transport_field.png"

            save_breast_pair(pre_crop, mask_crop, breast_pair_path, title)
            save_obstacle_proxy(pre_crop, mask_crop, obstacle_path, title)
            save_transport_field(pre_crop, mask_crop, transport_path, title)

            breast_pair_paths.append(breast_pair_path)
            obstacle_paths.append(obstacle_path)
            transport_paths.append(transport_path)

            row["status"] = "success"
            row["threshold"] = threshold
            row["frame_index"] = frame_idx
            row["mask_pixels"] = int(mask.sum())
            row["mask_fraction"] = float(mask.sum() / mask.size)
            row["breast_pair_figure"] = str(breast_pair_path)
            row["obstacle_proxy_figure"] = str(obstacle_path)
            row["transport_field_figure"] = str(transport_path)

        except Exception as e:
            row["status"] = "failed_exception"
            row["error"] = str(e)[:300]

        rows.append(row)

    out = pd.DataFrame(rows)
    out_csv = TABLES / "stage10c_breast_pair_case_manifest.csv"
    out.to_csv(out_csv, index=False)

    make_md(
        out_csv,
        VIEWS / "stage10c_breast_pair_case_manifest.md",
        "Stage 10C Breast Pair Case Manifest"
    )

    make_grid(
        breast_pair_paths[:args.max_grid],
        FIGS / "330_stage10c_breast_pair_grid.png",
        "Stage 10C: Left/right breast-pair visual"
    )

    make_grid(
        obstacle_paths[:args.max_grid],
        FIGS / "331_stage10c_obstacle_proxy_grid.png",
        "Stage 10C: Contrast-transport obstacle proxy"
    )

    make_grid(
        transport_paths[:args.max_grid],
        FIGS / "332_stage10c_transport_field_grid.png",
        "Stage 10C: Toy transport field near tumor"
    )

    plt.figure(figsize=(8, 5))
    out["status"].value_counts().sort_values().plot(kind="barh")
    plt.title("Stage 10C Breast-Pair Visualization Status")
    plt.xlabel("Cases")
    plt.tight_layout()
    plt.savefig(FIGS / "333_stage10c_status_counts.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    vals = pd.to_numeric(out.get("mask_fraction", pd.Series([])), errors="coerce").dropna()
    if len(vals):
        vals.plot(kind="hist", bins=20)
        plt.xlabel("Mask fraction")
        plt.ylabel("Cases")
        plt.title("Stage 10C Mask Fraction")
    else:
        plt.text(0.5, 0.5, "No mask fraction values", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "334_stage10c_mask_fraction_distribution.png", dpi=220)
    plt.close()

    n_total = len(out)
    n_success = int((out["status"] == "success").sum()) if n_total else 0

    report = []
    report.append("# Stage 10C Breast-Pair and Contrast-Transport Visualization Report")
    report.append("")
    report.append("This stage creates paper-style left/right breast-pair visuals and contrast-transport obstacle proxy images.")
    report.append("")
    report.append("## Main outputs")
    report.append("")
    report.append("- reports/tables/stage10c_breast_pair_case_manifest.csv")
    report.append("- reports/table_views/stage10c_breast_pair_case_manifest.md")
    report.append("- reports/figures/330_stage10c_breast_pair_grid.png")
    report.append("- reports/figures/331_stage10c_obstacle_proxy_grid.png")
    report.append("- reports/figures/332_stage10c_transport_field_grid.png")
    report.append("- reports/figures/333_stage10c_status_counts.png")
    report.append("- reports/figures/334_stage10c_mask_fraction_distribution.png")
    report.append("- reports/figures/stage10c_case_XXX_breast_pair.png")
    report.append("- reports/figures/stage10c_case_XXX_obstacle_proxy.png")
    report.append("- reports/figures/stage10c_case_XXX_transport_field.png")
    report.append("")
    report.append("## Main counts")
    report.append("")
    report.append("- Cases attempted: " + str(n_total))
    report.append("- Successful breast-pair visual cases: " + str(n_success))
    report.append("")
    report.append("## Biological interpretation")
    report.append("")
    report.append("The obstacle visualization is a conceptual contrast-transport disturbance map. It is not a direct measurement of blood flow, milk flow, or fluid flow.")
    report.append("")
    report.append("## Thesis use")
    report.append("")
    report.append("Use these figures in the visual methods section to explain how a breast-pair image and tumor-region disturbance proxy were created.")
    report.append("")
    report.append("## Important limitation")
    report.append("")
    report.append("The scientific main result should still be based on official MRI/FTV support-data pCR prediction. These figures support the imaging-method explanation.")

    (REPORTS / "Stage10C_Breast_Pair_Obstacle_Visualization_Report.md").write_text("\n".join(report))

    dash = REPORTS / "Dashboard.md"
    old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
    dash_add = (
        "\n\n## Stage 10C breast-pair obstacle visualization\n\n"
        "- [Breast-pair visualization report](Stage10C_Breast_Pair_Obstacle_Visualization_Report.md)\n"
        "- [Breast-pair case manifest](table_views/stage10c_breast_pair_case_manifest.md)\n"
    )
    if "Stage 10C breast-pair obstacle visualization" not in old_dash:
        dash.write_text(old_dash.rstrip() + dash_add)

    readme = REPO / "README.md"
    old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
    readme_add = (
        "\n\n### Stage 10C: breast-pair and contrast-transport visualization\n\n"
        "Main outputs:\n\n"
        "- reports/Stage10C_Breast_Pair_Obstacle_Visualization_Report.md\n"
        "- reports/figures/330_stage10c_breast_pair_grid.png\n"
        "- reports/figures/331_stage10c_obstacle_proxy_grid.png\n"
        "- reports/figures/332_stage10c_transport_field_grid.png\n"
    )
    if "Stage 10C: breast-pair and contrast-transport visualization" not in old_readme:
        readme.write_text(old_readme.rstrip() + readme_add)

    print("Stage 10C complete.")
    print("Cases attempted:", n_total)
    print("Successful cases:", n_success)

if __name__ == "__main__":
    main()
