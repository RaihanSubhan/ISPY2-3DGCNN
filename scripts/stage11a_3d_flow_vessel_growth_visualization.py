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

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


REPO = Path.home() / "ISPY2-3DGCNN"
REPORTS = REPO / "reports"
TABLES = REPORTS / "tables"
FIGS = REPORTS / "figures"
VIEWS = REPORTS / "table_views"

for p in [REPORTS, TABLES, FIGS, VIEWS]:
    p.mkdir(parents=True, exist_ok=True)

PHASE7C = TABLES / "phase7c_fractional_tumor_mask_candidate_table.csv"
PHASE9A_NODES = TABLES / "phase9a_longitudinal_visit_nodes.csv"
INVENTORY = TABLES / "ispy2_series_inventory.csv"

if not PHASE7C.exists():
    raise SystemExit("Missing reports/tables/phase7c_fractional_tumor_mask_candidate_table.csv. Run Phase 7C first.")

if not PHASE9A_NODES.exists():
    raise SystemExit("Missing reports/tables/phase9a_longitudinal_visit_nodes.csv. Run Phase 9A first.")


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
        raise RuntimeError(f"Unexpected SEG array shape: {arr.shape}")

    return ds, arr.astype(float)


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


def crop_full_width_around_mask(img, mask, pad=45):
    h, w = img.shape

    if mask.sum() == 0:
        return img, mask

    ys, xs = np.where(mask)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad)

    return img[y0:y1, :], mask[y0:y1, :]


def get_best_recovered_case():
    cand = pd.read_csv(PHASE7C, dtype=str).fillna("")
    work = cand[cand["recovery_status"].eq("fractional_tumor_candidate_pass")].copy()

    if work.empty:
        raise SystemExit("No Phase 7C fractional_tumor_candidate_pass cases found.")

    if "selected_frame_fraction" in work.columns:
        work["frac_num"] = pd.to_numeric(work["selected_frame_fraction"], errors="coerce")
        # Pick a visible but not large mask.
        work["display_score"] = work["frac_num"].apply(lambda x: x if pd.notna(x) and 0.0005 <= x <= 0.05 else 0)
        work = work.sort_values(["display_score", "patient_folder"], ascending=[False, True])

    return work.iloc[0].to_dict()


def load_case_image_and_mask(row):
    seg_file = row.get("seg_file", "")
    mr_file = row.get("matched_mr_file", "")
    threshold = float(row.get("selected_threshold", "nan"))
    frame_idx = int(float(row.get("selected_frame_index", 0)))

    _, seg_arr = read_seg_array(seg_file)
    _, raw = read_mr_image(mr_file)

    mask3d = seg_arr >= threshold

    if frame_idx < 0 or frame_idx >= mask3d.shape[0]:
        pix = mask3d.reshape(mask3d.shape[0], -1).sum(axis=1)
        frame_idx = int(np.argmax(pix))

    raw_mask = mask3d[frame_idx].astype(bool)

    if raw.shape != raw_mask.shape:
        raise RuntimeError(f"MR/mask shape mismatch: {raw.shape} vs {raw_mask.shape}")

    pre = preprocess_mri(raw)
    mask2d = preprocess_mask(raw_mask)

    return raw, pre, mask2d, mask3d, frame_idx


def flow_field(mask, direction=1, step=22):
    h, w = mask.shape
    yy, xx = np.mgrid[0:h:step, 0:w:step]

    mask = mask.astype(bool)

    if mask.sum() == 0:
        u = np.ones_like(xx, dtype=float) * direction
        v = np.zeros_like(u)
        return xx, yy, u, v

    cy, cx = ndi.center_of_mass(mask.astype(float))
    if not np.isfinite(cx):
        cx = w / 2
    if not np.isfinite(cy):
        cy = h / 2

    dist = ndi.distance_transform_edt(~mask)
    obstacle = np.exp(-dist[yy, xx] / 18.0)

    dy = yy - cy
    dx = xx - cx
    rr = np.sqrt(dx**2 + dy**2) + 1e-8

    # Base left-to-right or right-to-left motion.
    u = direction * (1.0 - 0.85 * obstacle)

    # Deflection around tumor.
    v = direction * 0.60 * obstacle * (dy / rr)

    # Do not show flow inside tumor.
    mask_sample = mask[yy, xx]
    u[mask_sample] = 0.0
    v[mask_sample] = 0.0

    return xx, yy, u, v


def save_bidirectional_flow(pre, mask, out_path, title):
    pre_crop, mask_crop = crop_full_width_around_mask(pre, mask, pad=50)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(pre_crop, cmap="gray")
    if mask_crop.sum() > 0:
        axes[0].contour(mask_crop, linewidths=1.3)
    axes[0].set_title("Breast pair + tumor region")
    axes[0].axis("off")

    axes[1].imshow(pre_crop, cmap="gray")
    x, y, u, v = flow_field(mask_crop, direction=1, step=22)
    axes[1].quiver(x, y, u, v, angles="xy", scale_units="xy", scale=0.12)
    if mask_crop.sum() > 0:
        axes[1].contour(mask_crop, linewidths=1.3)
    axes[1].set_title("Conceptual left-to-right transport")
    axes[1].axis("off")

    axes[2].imshow(pre_crop, cmap="gray")
    x, y, u, v = flow_field(mask_crop, direction=-1, step=22)
    axes[2].quiver(x, y, u, v, angles="xy", scale_units="xy", scale=0.12)
    if mask_crop.sum() > 0:
        axes[2].contour(mask_crop, linewidths=1.3)
    axes[2].set_title("Conceptual right-to-left transport")
    axes[2].axis("off")

    fig.suptitle(
        title + "\nVisualization proxy only: not measured blood, milk, or fluid flow",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_breast_shell(ax, alpha=0.10):
    u = np.linspace(0, 2 * np.pi, 80)
    v = np.linspace(0, np.pi, 40)
    x = 1.15 * np.outer(np.cos(u), np.sin(v))
    y = 0.85 * np.outer(np.sin(u), np.sin(v))
    z = 0.65 * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="gray", alpha=alpha, linewidth=0, shade=True)


def normalize_tumor_coords(mask3d, max_points=3500):
    coords = np.argwhere(mask3d.astype(bool))

    if coords.shape[0] == 0:
        return np.empty((0, 3))

    if coords.shape[0] > max_points:
        idx = np.linspace(0, coords.shape[0] - 1, max_points).astype(int)
        coords = coords[idx]

    z = coords[:, 0].astype(float)
    y = coords[:, 1].astype(float)
    x = coords[:, 2].astype(float)

    # Normalize into an ellipsoid-like coordinate space.
    x = (x - x.mean()) / max(mask3d.shape[2], 1) * 2.2
    y = (y - y.mean()) / max(mask3d.shape[1], 1) * 1.7
    z = (z - z.mean()) / max(mask3d.shape[0], 1) * 1.3

    # Put tumor slightly off center for visibility.
    x = x + 0.20
    y = y
    z = z + 0.05

    return np.vstack([x, y, z]).T


def draw_vessel_curves(ax, center=np.array([0.20, 0.0, 0.05])):
    # Conceptual vessel-like curves connected near tumor.
    starts = [
        np.array([-0.95, -0.45,  0.35]),
        np.array([-0.90,  0.35, -0.30]),
        np.array([ 0.85, -0.35,  0.25]),
        np.array([ 0.95,  0.40, -0.25]),
        np.array([ 0.00,  0.75,  0.45]),
        np.array([ 0.00, -0.75, -0.40]),
    ]

    for i, s in enumerate(starts):
        t = np.linspace(0, 1, 80)
        curve = (1 - t)[:, None] * s + t[:, None] * center

        # Add a small bend.
        bend = np.zeros_like(curve)
        bend[:, 1] = 0.10 * np.sin(2 * np.pi * t + i)
        bend[:, 2] = 0.08 * np.sin(np.pi * t + i / 2)

        c = curve + bend
        ax.plot(c[:, 0], c[:, 1], c[:, 2], linewidth=2, alpha=0.8)


def save_3d_tumor_vessel(mask3d, out_path, title):
    pts = normalize_tumor_coords(mask3d)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    plot_breast_shell(ax, alpha=0.11)

    if pts.shape[0] > 0:
        center = pts.mean(axis=0)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5, alpha=0.35)
        ax.scatter([center[0]], [center[1]], [center[2]], s=80)
        draw_vessel_curves(ax, center=center)
    else:
        draw_vessel_curves(ax)

    ax.set_title(title + "\nConceptual 3D tumor-vessel visualization", fontsize=11)
    ax.set_xlabel("left-right")
    ax.set_ylabel("anterior-posterior")
    ax.set_zlabel("slice/depth")
    ax.view_init(elev=18, azim=35)
    ax.set_box_aspect([1.3, 1.0, 0.8])
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def infer_patient_dates(patient, max_visits=4):
    if not INVENTORY.exists():
        return {}

    inv = pd.read_csv(INVENTORY, dtype=str).fillna("")
    if "patient_folder" not in inv.columns:
        return {}

    sub = inv[inv["patient_folder"].astype(str).eq(str(patient))].copy()
    if sub.empty:
        return {}

    date_cols = [c for c in sub.columns if "date" in c.lower()]
    dates = []

    for col in date_cols:
        for v in sub[col].dropna().astype(str).tolist():
            s = v.strip()
            if not s:
                continue
            dt = pd.NaT

            # DICOM date.
            if len(s) >= 8 and s[:8].isdigit():
                dt = pd.to_datetime(s[:8], format="%Y%m%d", errors="coerce")
            else:
                dt = pd.to_datetime(s, errors="coerce")

            if pd.notna(dt):
                dates.append(dt)

    dates = sorted(set(dates))

    if len(dates) == 0:
        return {}

    base = dates[0]
    out = {}

    labels = ["T0", "T1", "T2", "T3"]
    for i, d in enumerate(dates[:max_visits]):
        out[labels[i]] = {
            "date": d.strftime("%Y-%m-%d"),
            "days": int((d - base).days)
        }

    return out


def choose_growth_patient():
    nodes = pd.read_csv(PHASE9A_NODES, dtype=str).fillna("")
    for col in ["visit_index", "label_pCR", "ftv_volume", "sphericity", "longest_diameter", "bpe_5slice_mean"]:
        if col in nodes.columns:
            nodes[col] = pd.to_numeric(nodes[col], errors="coerce")

    rows = []
    for patient, sub in nodes.groupby("patient_folder"):
        sub = sub.sort_values("visit_index")
        n_visits = len(sub)
        n_vol = sub["ftv_volume"].notna().sum() if "ftv_volume" in sub.columns else 0
        vol_range = sub["ftv_volume"].max() - sub["ftv_volume"].min() if n_vol > 1 else 0
        label = int(sub["label_pCR"].dropna().iloc[0]) if sub["label_pCR"].notna().any() else -1

        rows.append({
            "patient_folder": patient,
            "n_visits": n_visits,
            "n_volume_values": int(n_vol),
            "volume_range": float(vol_range) if pd.notna(vol_range) else 0,
            "label_pCR": label,
        })

    score = pd.DataFrame(rows)
    if score.empty:
        raise SystemExit("No phase9a longitudinal nodes found.")

    score = score.sort_values(["n_visits", "n_volume_values", "volume_range"], ascending=[False, False, False])
    patient = score.iloc[0]["patient_folder"]

    sub = nodes[nodes["patient_folder"].astype(str).eq(str(patient))].sort_values("visit_index")
    return patient, sub, score


def ellipsoid_mesh(center, radii, n=35):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = radii[0] * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radii[1] * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    return x, y, z


def save_multivisit_growth(patient, sub, out_path):
    dates = infer_patient_dates(patient, max_visits=4)

    colors = {
        "T0": "red",
        "T1": "gold",
        "T2": "green",
        "T3": "deepskyblue"
    }

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    plot_breast_shell(ax, alpha=0.08)

    # Find volume scale.
    vols = pd.to_numeric(sub["ftv_volume"], errors="coerce")
    max_vol = np.nanmax(vols.values) if vols.notna().any() else 1.0
    if not np.isfinite(max_vol) or max_vol <= 0:
        max_vol = 1.0

    visit_rows = []

    # Single anatomical location, tiny offsets only for visibility.
    base_center = np.array([0.18, 0.02, 0.02])
    offsets = {
        "T0": np.array([-0.04, 0.00, 0.00]),
        "T1": np.array([-0.015, 0.01, 0.01]),
        "T2": np.array([0.015, -0.01, -0.005]),
        "T3": np.array([0.04, 0.00, 0.005]),
    }

    for _, row in sub.iterrows():
        visit = str(row["visit_label"])
        vol = pd.to_numeric(row["ftv_volume"], errors="coerce")

        if pd.isna(vol) or vol <= 0:
            continue

        relative_radius = (vol / max_vol) ** (1 / 3)
        radius = 0.22 * relative_radius

        sph = pd.to_numeric(row.get("sphericity", np.nan), errors="coerce")
        ld = pd.to_numeric(row.get("longest_diameter", np.nan), errors="coerce")

        # Simple ellipsoid. If sphericity is low, make it more elongated.
        if pd.notna(sph):
            elong = np.clip(1.4 - sph, 0.75, 1.6)
        else:
            elong = 1.0

        center = base_center + offsets.get(visit, np.array([0.0, 0.0, 0.0]))
        radii = np.array([radius * elong, radius, radius * 0.85])

        X, Y, Z = ellipsoid_mesh(center, radii)
        ax.plot_surface(X, Y, Z, color=colors.get(visit, "gray"), alpha=0.38, linewidth=0)

        if visit in dates:
            label = f"{visit}\n{dates[visit]['date']}\nDay {dates[visit]['days']}"
            date_value = dates[visit]["date"]
            day_value = dates[visit]["days"]
        else:
            label = f"{visit}\nDate unknown"
            date_value = "unknown"
            day_value = "unknown"

        ax.text(center[0], center[1] + 0.35, center[2] + 0.12, label, fontsize=9)

        visit_rows.append({
            "patient_folder": patient,
            "visit_label": visit,
            "visit_index": int(row["visit_index"]),
            "date": date_value,
            "days_from_first": day_value,
            "ftv_volume": float(vol),
            "color": colors.get(visit, "gray"),
            "visual_note": "tumor ellipsoid scaled from FTV volume; slight offsets are for visibility"
        })

    ax.set_title(
        f"Stage 11A multivisit 3D tumor growth comparison\nPatient: {patient}",
        fontsize=11
    )
    ax.set_xlabel("left-right")
    ax.set_ylabel("anterior-posterior")
    ax.set_zlabel("depth")
    ax.view_init(elev=18, azim=35)
    ax.set_box_aspect([1.3, 1.0, 0.8])

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    return pd.DataFrame(visit_rows)


def table_figure(df, out_path, title):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    if df.empty:
        ax.text(0.5, 0.5, "No visit rows", ha="center", va="center")
    else:
        display = df[["visit_label", "date", "days_from_first", "ftv_volume", "color"]].copy()
        display["ftv_volume"] = pd.to_numeric(display["ftv_volume"], errors="coerce").round(3)
        tbl = ax.table(
            cellText=display.values,
            colLabels=display.columns,
            loc="center",
            cellLoc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.4)

    ax.set_title(title)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-growth-patients", type=int, default=1)
    args = parser.parse_args()

    # Case-level flow/tumor-vessel visual.
    case = get_best_recovered_case()
    raw, pre, mask2d, mask3d, frame_idx = load_case_image_and_mask(case)

    patient_case = case.get("patient_folder", "")
    title = f"Stage 11A case: {patient_case}"

    save_bidirectional_flow(
        pre,
        mask2d,
        FIGS / "410_stage11a_bidirectional_flow_proxy.png",
        title
    )

    save_3d_tumor_vessel(
        mask3d,
        FIGS / "411_stage11a_3d_tumor_vessel_schematic.png",
        title
    )

    case_manifest = pd.DataFrame([{
        "patient_folder": patient_case,
        "study_folder": case.get("study_folder", ""),
        "seg_series_uid": case.get("seg_series_uid", ""),
        "selected_threshold": case.get("selected_threshold", ""),
        "selected_frame_index": frame_idx,
        "flow_proxy_figure": "reports/figures/410_stage11a_bidirectional_flow_proxy.png",
        "tumor_vessel_3d_figure": "reports/figures/411_stage11a_3d_tumor_vessel_schematic.png",
        "important_note": "Flow and vessel curves are conceptual visual proxies, not measured flow."
    }])
    case_manifest.to_csv(TABLES / "stage11a_selected_case_manifest.csv", index=False)

    # Multivisit growth visual.
    growth_patient, growth_nodes, growth_scores = choose_growth_patient()
    visit_manifest = save_multivisit_growth(
        growth_patient,
        growth_nodes,
        FIGS / "412_stage11a_multivisit_3d_tumor_growth.png"
    )
    visit_manifest.to_csv(TABLES / "stage11a_growth_patient_manifest.csv", index=False)

    table_figure(
        visit_manifest,
        FIGS / "413_stage11a_growth_visit_table.png",
        "Stage 11A multivisit tumor growth visit labels"
    )

    growth_scores.to_csv(TABLES / "stage11a_growth_patient_selection_scores.csv", index=False)

    # Combined storyboard.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.reshape(-1)

    for ax, path, ttl in [
        (axes[0], FIGS / "410_stage11a_bidirectional_flow_proxy.png", "A. Bidirectional conceptual transport"),
        (axes[1], FIGS / "411_stage11a_3d_tumor_vessel_schematic.png", "B. 3D tumor-vessel schematic"),
        (axes[2], FIGS / "412_stage11a_multivisit_3d_tumor_growth.png", "C. Multivisit 3D tumor growth"),
        (axes[3], FIGS / "413_stage11a_growth_visit_table.png", "D. Visit dates / days")
    ]:
        if Path(path).exists():
            img = plt.imread(path)
            ax.imshow(img)
            ax.set_title(ttl, fontsize=9)
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, f"Missing: {path.name}", ha="center", va="center")
            ax.axis("off")

    fig.suptitle("Stage 11A flow, vessel, and multivisit tumor-growth visual package", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGS / "414_stage11a_combined_visual_storyboard.png", dpi=220)
    plt.close()

    # Claims and limits table.
    claims = pd.DataFrame([
        {
            "visual_component": "bidirectional flow field",
            "what_it_shows": "conceptual contrast-transport direction around a tumor obstacle",
            "safe_wording": "contrast-transport disturbance proxy",
            "unsafe_wording": "measured blood, milk, or fluid flow"
        },
        {
            "visual_component": "3D tumor-vessel schematic",
            "what_it_shows": "tumor inside transparent breast shell connected to conceptual vessel-like curves",
            "safe_wording": "schematic vessel/tumor relationship",
            "unsafe_wording": "validated vessel segmentation or measured vascular network"
        },
        {
            "visual_component": "multivisit 3D growth",
            "what_it_shows": "FTV-based tumor-size change across visits in one transparent breast shell",
            "safe_wording": "FTV-scaled longitudinal tumor growth visualization",
            "unsafe_wording": "registered voxel-level tumor growth map"
        },
        {
            "visual_component": "dates/days",
            "what_it_shows": "DICOM date-derived labels when available, otherwise date unknown",
            "safe_wording": "visit date labels are inferred from available metadata",
            "unsafe_wording": "definitive treatment-day schedule unless clinically verified"
        }
    ])
    claims.to_csv(TABLES / "stage11a_visual_claims_and_limits.csv", index=False)

    make_md(
        TABLES / "stage11a_selected_case_manifest.csv",
        VIEWS / "stage11a_selected_case_manifest.md",
        "Stage 11A Selected Case Manifest"
    )
    make_md(
        TABLES / "stage11a_growth_patient_manifest.csv",
        VIEWS / "stage11a_growth_patient_manifest.md",
        "Stage 11A Growth Patient Manifest"
    )
    make_md(
        TABLES / "stage11a_visual_claims_and_limits.csv",
        VIEWS / "stage11a_visual_claims_and_limits.md",
        "Stage 11A Visual Claims and Limits"
    )

    # Report.
    report = []
    report.append("# Stage 11A 3D Flow, Vessel, and Multivisit Tumor-Growth Visualization Report")
    report.append("")
    report.append("This stage extends the visual-methods work beyond the four earlier outputs.")
    report.append("")
    report.append("## What was created")
    report.append("")
    report.append("- Bidirectional conceptual contrast-transport flow field around a tumor mask")
    report.append("- 3D transparent breast shell with recovered tumor mask and conceptual vessel-like curves")
    report.append("- Multivisit FTV-scaled 3D tumor growth visualization in one transparent breast shape")
    report.append("- Visit label table with date/day labels when metadata is available")
    report.append("")
    report.append("## Main outputs")
    report.append("")
    report.append("- reports/figures/410_stage11a_bidirectional_flow_proxy.png")
    report.append("- reports/figures/411_stage11a_3d_tumor_vessel_schematic.png")
    report.append("- reports/figures/412_stage11a_multivisit_3d_tumor_growth.png")
    report.append("- reports/figures/413_stage11a_growth_visit_table.png")
    report.append("- reports/figures/414_stage11a_combined_visual_storyboard.png")
    report.append("- reports/tables/stage11a_selected_case_manifest.csv")
    report.append("- reports/tables/stage11a_growth_patient_manifest.csv")
    report.append("- reports/tables/stage11a_visual_claims_and_limits.csv")
    report.append("")
    report.append("## Selected cases")
    report.append("")
    report.append("- Flow/tumor-vessel case patient: " + str(patient_case))
    report.append("- Multivisit growth patient: " + str(growth_patient))
    report.append("")
    report.append("## Important scientific limitation")
    report.append("")
    report.append("These visuals are conceptual and thesis-facing. They do not measure true blood flow, milk flow, or fluid flow. The correct scientific wording is contrast-transport or enhancement disturbance proxy.")
    report.append("")
    report.append("## How to use this in the thesis")
    report.append("")
    report.append("Use these figures to explain the proposed biological intuition and future graph-learning idea. The main validated pilot result remains the official MRI/FTV support-data pCR prediction model.")
    report.append("")
    report.append("## Next step")
    report.append("")
    report.append("Stage 11B should create a publication-style single figure that combines the best panels into one polished thesis figure.")

    (REPORTS / "Stage11A_3D_Flow_Vessel_Growth_Visualization_Report.md").write_text("\n".join(report))

    # Dashboard / README.
    dash = REPORTS / "Dashboard.md"
    old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
    dash_add = (
        "\n\n## Stage 11A 3D flow-vessel-growth visualization\n\n"
        "- [3D flow, vessel, and tumor-growth report](Stage11A_3D_Flow_Vessel_Growth_Visualization_Report.md)\n"
        "- [Selected case manifest](table_views/stage11a_selected_case_manifest.md)\n"
        "- [Visual claims and limits](table_views/stage11a_visual_claims_and_limits.md)\n"
    )
    if "Stage 11A 3D flow-vessel-growth visualization" not in old_dash:
        dash.write_text(old_dash.rstrip() + dash_add)

    readme = REPO / "README.md"
    old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
    readme_add = (
        "\n\n### Stage 11A: 3D flow-vessel-growth visualization\n\n"
        "Main outputs:\n\n"
        "- reports/Stage11A_3D_Flow_Vessel_Growth_Visualization_Report.md\n"
        "- reports/figures/410_stage11a_bidirectional_flow_proxy.png\n"
        "- reports/figures/411_stage11a_3d_tumor_vessel_schematic.png\n"
        "- reports/figures/412_stage11a_multivisit_3d_tumor_growth.png\n"
    )
    if "Stage 11A: 3D flow-vessel-growth visualization" not in old_readme:
        readme.write_text(old_readme.rstrip() + readme_add)

    print("Stage 11A complete.")
    print("Flow/tumor-vessel patient:", patient_case)
    print("Growth patient:", growth_patient)
    print("Important note: visuals are conceptual proxies, not measured fluid flow.")


if __name__ == "__main__":
    main()
