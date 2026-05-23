from pathlib import Path
import os
import subprocess
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pydicom
from scipy import ndimage as ndi
from skimage import exposure, morphology, measure, filters

REPO = Path.home() / "ISPY2-3DGCNN"
DATA_ROOT = Path.home() / "ispy2" / "ispy2"
REPORTS = REPO / "reports"
TABLES = REPORTS / "tables"
FIGS = REPORTS / "figures"
VIEWS = REPORTS / "table_views"

for p in [REPORTS, TABLES, FIGS, VIEWS]:
    p.mkdir(parents=True, exist_ok=True)

PHASE1_INVENTORY = TABLES / "ispy2_series_inventory.csv"
PHASE7C = TABLES / "phase7c_fractional_tumor_mask_candidate_table.csv"

if not PHASE1_INVENTORY.exists():
    raise SystemExit("Missing reports/tables/ispy2_series_inventory.csv. Run Phase 1 first.")

if not PHASE7C.exists():
    raise SystemExit("Missing reports/tables/phase7c_fractional_tumor_mask_candidate_table.csv. Run Phase 7C first.")

def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
        return out.strip()
    except Exception as e:
        return str(e)

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
    return ds, normalize01(img)

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
        raise RuntimeError(f"Unexpected SEG shape: {arr.shape}")

    return ds, arr.astype(float)

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
    mask = mask.astype(bool)

    labels = measure.label(mask)
    if labels.max() > 0:
        props = measure.regionprops(labels)
        biggest = max(props, key=lambda p: p.area)
        mask = labels == biggest.label

    return mask.astype(bool)

def split_breast_pair(img, mask):
    h, w = img.shape
    mid = w // 2

    left_img = img[:, :mid]
    right_img = img[:, mid:]

    left_mask = mask[:, :mid]
    right_mask = mask[:, mid:]

    # Keep left and right in natural image order.
    pair_img = np.concatenate([left_img, right_img], axis=1)
    pair_mask = np.concatenate([left_mask, right_mask], axis=1)

    return left_img, right_img, left_mask, right_mask, pair_img, pair_mask

def obstacle_flow_map(img, mask):
    mask = mask.astype(bool)
    dist = ndi.distance_transform_edt(~mask)
    dist_norm = normalize01(dist)

    # This is a visual hypothesis map, not real fluid-flow measurement.
    obstacle = np.exp(-dist / 20.0)
    flow_proxy = normalize01(img * (1.0 - 0.65 * obstacle))
    flow_proxy[mask] = 0.15

    return flow_proxy

def make_preprocess_figure(raw, pre, raw_mask, pre_mask, out_path, title):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    axes[0, 0].imshow(raw, cmap="gray")
    axes[0, 0].set_title("Raw MRI, no mask")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pre, cmap="gray")
    axes[0, 1].set_title("Preprocessed MRI, no mask")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(raw, cmap="gray")
    if raw_mask.sum() > 0:
        axes[1, 0].contour(raw_mask, linewidths=1.2)
    axes[1, 0].set_title("Raw MRI + raw recovered mask")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pre, cmap="gray")
    if pre_mask.sum() > 0:
        axes[1, 1].contour(pre_mask, linewidths=1.2)
    axes[1, 1].set_title("Preprocessed MRI + preprocessed mask")
    axes[1, 1].axis("off")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def make_segmentation_top_bottom(raw, pre, raw_mask, pre_mask, out_path, title):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    axes[0, 0].imshow(raw, cmap="gray")
    axes[0, 0].set_title("Top: raw MRI")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(raw, cmap="gray")
    axes[0, 1].contour(raw_mask, linewidths=1.2)
    axes[0, 1].set_title("Top: raw MRI + raw mask")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(pre, cmap="gray")
    axes[1, 0].set_title("Bottom: preprocessed MRI")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pre, cmap="gray")
    axes[1, 1].contour(pre_mask, linewidths=1.2)
    axes[1, 1].set_title("Bottom: preprocessed MRI + preprocessed mask")
    axes[1, 1].axis("off")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def make_breast_pair_figure(pre, pre_mask, out_path, title):
    left_img, right_img, left_mask, right_mask, pair_img, pair_mask = split_breast_pair(pre, pre_mask)
    flow_proxy = obstacle_flow_map(pair_img, pair_mask)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(left_img, cmap="gray")
    axes[0, 0].set_title("Left breast half")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(right_img, cmap="gray")
    axes[0, 1].set_title("Right breast half")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(pair_img, cmap="gray")
    axes[0, 2].set_title("Single breast-pair image")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(pair_img, cmap="gray")
    if pair_mask.sum() > 0:
        axes[1, 0].contour(pair_mask, linewidths=1.2)
    axes[1, 0].set_title("Breast pair + tumor mask")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(flow_proxy, cmap="gray")
    if pair_mask.sum() > 0:
        axes[1, 1].contour(pair_mask, linewidths=1.0)
    axes[1, 1].set_title("Contrast-transport obstacle proxy")
    axes[1, 1].axis("off")

    # Simple arrow field to show reduced movement near tumor.
    axes[1, 2].imshow(pair_img, cmap="gray")
    h, w = pair_img.shape
    yy, xx = np.mgrid[0:h:24, 0:w:24]
    mask_small = pair_mask[yy, xx]
    dist = ndi.distance_transform_edt(~pair_mask)
    mag = 1.0 - 0.8 * np.exp(-dist[yy, xx] / 25.0)
    mag[mask_small] = 0.05
    uu = mag
    vv = np.zeros_like(uu)
    axes[1, 2].quiver(xx, yy, uu, vv, angles="xy", scale_units="xy", scale=0.15)
    if pair_mask.sum() > 0:
        axes[1, 2].contour(pair_mask, linewidths=1.0)
    axes[1, 2].set_title("Toy flow field: lower near tumor")
    axes[1, 2].axis("off")

    fig.suptitle(title + "\nNote: obstacle map is a visualization, not measured milk/blood flow.", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
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

# ------------------------------------------------------
# 1. Dataset structure summary
# ------------------------------------------------------
inventory = pd.read_csv(PHASE1_INVENTORY, dtype=str).fillna("")

for c in ["dcm_count"]:
    if c in inventory.columns:
        inventory[c] = pd.to_numeric(inventory[c], errors="coerce").fillna(0).astype(int)

size_text = run_cmd('du -sh "$HOME/ispy2" | awk \'{print $1}\'')

summary_rows = [
    {"metric": "dataset_root", "value": str(DATA_ROOT)},
    {"metric": "dataset_size_du", "value": size_text},
    {"metric": "patients", "value": inventory["patient_folder"].nunique() if "patient_folder" in inventory.columns else ""},
    {"metric": "studies", "value": inventory["study_folder"].nunique() if "study_folder" in inventory.columns else ""},
    {"metric": "series", "value": len(inventory)},
    {"metric": "dicom_files_counted", "value": int(inventory["dcm_count"].sum()) if "dcm_count" in inventory.columns else ""},
]

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(TABLES / "stage10a_dataset_structure_summary.csv", index=False)

modality = inventory["Modality"].replace("", "UNKNOWN").value_counts().reset_index()
modality.columns = ["modality", "series_count"]
modality.to_csv(TABLES / "stage10a_modality_series_counts.csv", index=False)

plt.figure(figsize=(8, 5))
modality.head(10).set_index("modality")["series_count"].plot(kind="bar")
plt.title("Stage 10A Dataset Structure: Series by Modality")
plt.ylabel("Series count")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(FIGS / "310_stage10a_dataset_modality_counts.png", dpi=220)
plt.close()

# ------------------------------------------------------
# 2. Visual preprocessing from recovered Phase 7C masks
# ------------------------------------------------------
candidates = pd.read_csv(PHASE7C, dtype=str).fillna("")
work = candidates[candidates["recovery_status"].eq("fractional_tumor_candidate_pass")].copy()
work = work.head(12)

case_rows = []
preprocess_paths = []
seg_paths = []
pair_paths = []

for _, r in work.iterrows():
    case_no = len(case_rows) + 1

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
            case_rows.append(row)
            continue

        if not Path(mr_file).exists():
            row["status"] = "missing_mr_file"
            case_rows.append(row)
            continue

        _, seg_arr = read_seg_array(seg_file)
        _, raw_mri = read_mr_image(mr_file)

        mask3d = seg_arr >= threshold

        if frame_idx < 0 or frame_idx >= mask3d.shape[0]:
            frame_pixels = mask3d.reshape(mask3d.shape[0], -1).sum(axis=1)
            frame_idx = int(np.argmax(frame_pixels))

        raw_mask = mask3d[frame_idx].astype(bool)

        if raw_mri.shape != raw_mask.shape:
            row["status"] = "shape_mismatch"
            row["raw_mri_shape"] = str(raw_mri.shape)
            row["mask_shape"] = str(raw_mask.shape)
            case_rows.append(row)
            continue

        pre_mri = preprocess_mri(raw_mri)
        pre_mask = preprocess_mask(raw_mask)

        if pre_mask.sum() < 20:
            row["status"] = "mask_too_small_after_preprocess"
            case_rows.append(row)
            continue

        row["status"] = "success"
        row["threshold"] = threshold
        row["frame_index"] = frame_idx
        row["raw_mask_pixels"] = int(raw_mask.sum())
        row["preprocessed_mask_pixels"] = int(pre_mask.sum())
        row["mask_fraction"] = float(pre_mask.sum() / pre_mask.size)

        title = f"Stage 10A case {case_no}: {row['patient_folder']}"

        out_pre = FIGS / f"stage10a_case_{case_no:03d}_preprocess_without_with_mask.png"
        make_preprocess_figure(raw_mri, pre_mri, raw_mask, pre_mask, out_pre, title)
        preprocess_paths.append(out_pre)

        out_seg = FIGS / f"stage10a_case_{case_no:03d}_segmentation_top_bottom.png"
        make_segmentation_top_bottom(raw_mri, pre_mri, raw_mask, pre_mask, out_seg, title)
        seg_paths.append(out_seg)

        out_pair = FIGS / f"stage10a_case_{case_no:03d}_breast_pair_obstacle_map.png"
        make_breast_pair_figure(pre_mri, pre_mask, out_pair, title)
        pair_paths.append(out_pair)

    except Exception as e:
        row["status"] = "failed_exception"
        row["error"] = str(e)[:300]

    case_rows.append(row)

case_df = pd.DataFrame(case_rows)
case_df.to_csv(TABLES / "stage10a_visual_preprocessing_case_summary.csv", index=False)
make_md(TABLES / "stage10a_visual_preprocessing_case_summary.csv",
        VIEWS / "stage10a_visual_preprocessing_case_summary.md",
        "Stage 10A Visual Preprocessing Case Summary")

# Grid figures
def make_grid(paths, out_path, title):
    paths = [Path(p) for p in paths if Path(p).exists()]
    if not paths:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "No images created", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=220)
        plt.close()
        return

    cols = 2
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

make_grid(preprocess_paths, FIGS / "311_stage10a_preprocess_without_with_mask_grid.png",
          "Stage 10A: Raw/preprocessed MRI with and without mask")

make_grid(seg_paths, FIGS / "312_stage10a_segmentation_top_bottom_grid.png",
          "Stage 10A: Segmentation top raw, bottom preprocessed")

make_grid(pair_paths, FIGS / "313_stage10a_breast_pair_obstacle_grid.png",
          "Stage 10A: Single breast-pair and obstacle proxy")

# Status plot
plt.figure(figsize=(8, 5))
case_df["status"].value_counts().sort_values().plot(kind="barh")
plt.title("Stage 10A Visual Preprocessing Status")
plt.xlabel("Cases")
plt.tight_layout()
plt.savefig(FIGS / "314_stage10a_visual_status.png", dpi=220)
plt.close()

n_success = int((case_df["status"] == "success").sum())
n_total = len(case_df)

report = []
report.append("# Stage 10A Dataset Structure and Visual Preprocessing Report")
report.append("")
report.append("This stage creates dataset structure outputs and visual preprocessing examples from recovered tumor-mask candidates.")
report.append("")
report.append("## What was created")
report.append("")
report.append("- Dataset structure summary with size and DICOM counts")
report.append("- Raw MRI without mask and preprocessed MRI without mask")
report.append("- Raw MRI with raw recovered mask and preprocessed MRI with preprocessed mask")
report.append("- Top/bottom segmentation figure: raw MRI + raw mask, then preprocessed MRI + preprocessed mask")
report.append("- Single breast-pair visual from left and right halves")
report.append("- Contrast-transport obstacle proxy visualization")
report.append("")
report.append("## Main outputs")
report.append("")
report.append("- reports/tables/stage10a_dataset_structure_summary.csv")
report.append("- reports/tables/stage10a_visual_preprocessing_case_summary.csv")
report.append("- reports/figures/310_stage10a_dataset_modality_counts.png")
report.append("- reports/figures/311_stage10a_preprocess_without_with_mask_grid.png")
report.append("- reports/figures/312_stage10a_segmentation_top_bottom_grid.png")
report.append("- reports/figures/313_stage10a_breast_pair_obstacle_grid.png")
report.append("- reports/figures/314_stage10a_visual_status.png")
report.append("")
report.append("## Main counts")
report.append("")
report.append("- Cases attempted: " + str(n_total))
report.append("- Successful visual preprocessing cases: " + str(n_success))
report.append("")
report.append("## Important biological note")
report.append("")
report.append("The obstacle map is a visualization of contrast-transport/perfusion disturbance, not a direct measurement of blood, milk, or fluid flow.")
report.append("")
report.append("## Limitation")
report.append("")
report.append("This stage uses recovered fractional tumor-mask candidates. Exact replication of the attached example image/PDF requires the user to upload those files again.")
report.append("")
report.append("## Next step")
report.append("")
report.append("Use these visuals in the methods section of the thesis. For a publishable study, the main result should still be the official MRI/FTV support-data pCR prediction path.")

(REPORTS / "Stage10A_Dataset_Visual_Preprocessing_Report.md").write_text("\n".join(report))

# Dashboard/README
dash = REPORTS / "Dashboard.md"
old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
dash_add = (
    "\n\n## Stage 10A visual preprocessing\n\n"
    "- [Dataset and visual preprocessing report](Stage10A_Dataset_Visual_Preprocessing_Report.md)\n"
    "- [Visual preprocessing case summary](table_views/stage10a_visual_preprocessing_case_summary.md)\n"
)
if "Stage 10A visual preprocessing" not in old_dash:
    dash.write_text(old_dash.rstrip() + dash_add)

readme = REPO / "README.md"
old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
readme_add = (
    "\n\n### Stage 10A: dataset structure and visual preprocessing\n\n"
    "Main outputs:\n\n"
    "- reports/Stage10A_Dataset_Visual_Preprocessing_Report.md\n"
    "- reports/figures/311_stage10a_preprocess_without_with_mask_grid.png\n"
    "- reports/figures/312_stage10a_segmentation_top_bottom_grid.png\n"
    "- reports/figures/313_stage10a_breast_pair_obstacle_grid.png\n"
)
if "Stage 10A: dataset structure and visual preprocessing" not in old_readme:
    readme.write_text(old_readme.rstrip() + readme_add)

print("Stage 10A complete.")
print("Dataset size:", size_text)
print("Cases attempted:", n_total)
print("Successful cases:", n_success)
