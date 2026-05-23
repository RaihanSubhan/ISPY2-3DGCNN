from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

import pydicom
from scipy import ndimage as ndi
from skimage import exposure, morphology, measure

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


REPO = Path.home() / "ISPY2-3DGCNN"
REPORTS = REPO / "reports"
TABLES = REPORTS / "tables"
FIGS = REPORTS / "figures"
VIEWS = REPORTS / "table_views"

for p in [REPORTS, TABLES, FIGS, VIEWS]:
    p.mkdir(parents=True, exist_ok=True)

PHASE7C = TABLES / "phase7c_fractional_tumor_mask_candidate_table.csv"
STAGE11A_SELECTED = TABLES / "stage11a_selected_case_manifest.csv"
STAGE11A_GROWTH = TABLES / "stage11a_growth_patient_manifest.csv"
PHASE9A_NODES = TABLES / "phase9a_longitudinal_visit_nodes.csv"

if not PHASE7C.exists():
    raise SystemExit("Missing reports/tables/phase7c_fractional_tumor_mask_candidate_table.csv. Run Phase 7C first.")

if not PHASE9A_NODES.exists():
    raise SystemExit("Missing reports/tables/phase9a_longitudinal_visit_nodes.csv. Run Phase 9A first.")


# -----------------------------
# Basic helpers
# -----------------------------
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


def crop_full_width_around_mask(img, mask, pad=55):
    h, w = img.shape

    if mask.sum() == 0:
        return img, mask

    ys, xs = np.where(mask)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad)

    return img[y0:y1, :], mask[y0:y1, :]


def find_selected_case():
    candidates = pd.read_csv(PHASE7C, dtype=str).fillna("")
    candidates = candidates[candidates["recovery_status"].eq("fractional_tumor_candidate_pass")].copy()

    if candidates.empty:
        raise SystemExit("No fractional_tumor_candidate_pass cases found.")

    if STAGE11A_SELECTED.exists():
        selected = pd.read_csv(STAGE11A_SELECTED, dtype=str).fillna("")
        if not selected.empty and "seg_series_uid" in selected.columns:
            seg_uid = selected.iloc[0].get("seg_series_uid", "")
            match = candidates[candidates["seg_series_uid"].astype(str).eq(str(seg_uid))]
            if not match.empty:
                return match.iloc[0].to_dict()

    # fallback: choose a small, visible mask
    if "selected_frame_fraction" in candidates.columns:
        candidates["frac_num"] = pd.to_numeric(candidates["selected_frame_fraction"], errors="coerce")
        candidates["score"] = candidates["frac_num"].apply(
            lambda x: x if pd.notna(x) and 0.0005 <= x <= 0.05 else 0
        )
        candidates = candidates.sort_values(["score", "patient_folder"], ascending=[False, True])

    return candidates.iloc[0].to_dict()


def load_case_image_and_mask(case):
    seg_file = case.get("seg_file", "")
    mr_file = case.get("matched_mr_file", "")
    threshold = float(case.get("selected_threshold", "nan"))
    frame_idx = int(float(case.get("selected_frame_index", 0)))

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


# -----------------------------
# 2D animated transport proxy
# -----------------------------
def compute_particle_positions(frame, n_frames, h, w, cy, cx, direction=1, n_particles=18):
    # particles span the y axis, and move across x axis
    y0 = np.linspace(0.18 * h, 0.82 * h, n_particles)

    phase = frame / max(n_frames - 1, 1)

    if direction == 1:
        x = phase * w
    else:
        x = (1.0 - phase) * w

    x_arr = np.full_like(y0, x, dtype=float)

    dx = x_arr - cx
    dy = y0 - cy
    rr2 = dx**2 + dy**2

    sigma = max(22.0, min(h, w) / 7.0)
    obstacle = np.exp(-rr2 / (2 * sigma**2))

    # Deflect around tumor.
    y_deflect = y0 + np.sign(dy + 1e-8) * obstacle * 45.0
    speed_alpha = np.clip(1.0 - 0.70 * obstacle, 0.25, 1.0)

    return x_arr, y_deflect, speed_alpha


def make_bidirectional_transport_animation(pre, mask, out_gif, out_png, title, frames=70):
    img, mask_crop = crop_full_width_around_mask(pre, mask, pad=60)
    h, w = img.shape

    if mask_crop.sum() > 0:
        cy, cx = ndi.center_of_mass(mask_crop.astype(float))
    else:
        cy, cx = h / 2, w / 2

    fig, ax = plt.subplots(figsize=(10, 4.5))

    def update(frame):
        ax.clear()
        ax.imshow(img, cmap="gray")

        if mask_crop.sum() > 0:
            ax.contour(mask_crop, linewidths=1.4)

        x1, y1, a1 = compute_particle_positions(frame, frames, h, w, cy, cx, direction=1)
        x2, y2, a2 = compute_particle_positions(frame, frames, h, w, cy, cx, direction=-1)

        ax.scatter(x1, y1, s=32, alpha=0.70, label="left-to-right proxy")
        ax.scatter(x2, y2, s=32, alpha=0.70, marker="x", label="right-to-left proxy")

        # Direction arrows
        ax.arrow(0.12*w, 0.08*h, 0.28*w, 0, head_width=8, head_length=12, length_includes_head=True)
        ax.arrow(0.88*w, 0.12*h, -0.28*w, 0, head_width=8, head_length=12, length_includes_head=True)

        ax.set_title(
            title + "\nAnimated conceptual contrast-transport proxy; not measured blood/milk/fluid flow",
            fontsize=10
        )
        ax.axis("off")
        ax.legend(loc="lower right", fontsize=7)

    anim = FuncAnimation(fig, update, frames=frames, interval=100)
    anim.save(out_gif, writer=PillowWriter(fps=10))

    update(frames // 2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)


# -----------------------------
# 3D breast shell, tumor, vessel, growth
# -----------------------------
def ellipsoid_surface(center, radii, n=40):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = radii[0] * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radii[1] * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    return x, y, z


def plot_breast_shell(ax, alpha=0.12):
    X, Y, Z = ellipsoid_surface(
        center=np.array([0.0, 0.0, 0.0]),
        radii=np.array([1.20, 0.88, 0.70]),
        n=45
    )
    ax.plot_surface(X, Y, Z, color="lightgray", alpha=alpha, linewidth=0, shade=True)


def plot_tumor_ellipsoid(ax, center, radii, color="red", alpha=0.55):
    X, Y, Z = ellipsoid_surface(center=center, radii=radii, n=35)
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)


def draw_vessel_curves(ax, center=np.array([0.18, 0.02, 0.05])):
    starts = [
        np.array([-1.00, -0.45,  0.35]),
        np.array([-0.95,  0.40, -0.30]),
        np.array([ 0.95, -0.38,  0.25]),
        np.array([ 1.05,  0.45, -0.22]),
        np.array([ 0.00,  0.78,  0.45]),
        np.array([ 0.00, -0.78, -0.40]),
    ]

    for i, s in enumerate(starts):
        t = np.linspace(0, 1, 100)
        curve = (1 - t)[:, None] * s + t[:, None] * center

        bend = np.zeros_like(curve)
        bend[:, 1] = 0.10 * np.sin(2 * np.pi * t + i)
        bend[:, 2] = 0.08 * np.sin(np.pi * t + i / 2)

        c = curve + bend
        ax.plot(c[:, 0], c[:, 1], c[:, 2], linewidth=2.1, alpha=0.85)


def setup_3d_axes(ax, title, azim):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("left-right")
    ax.set_ylabel("anterior-posterior")
    ax.set_zlabel("depth")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.95, 0.95)
    ax.set_zlim(-0.75, 0.75)
    ax.view_init(elev=18, azim=azim)
    ax.set_box_aspect([1.3, 1.0, 0.8])


def make_tumor_vessel_animation(out_gif, out_png, title, frames=80):
    fig = plt.figure(figsize=(8.5, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    center = np.array([0.20, 0.02, 0.05])
    radii = np.array([0.23, 0.16, 0.13])

    def update(frame):
        ax.clear()
        azim = 35 + 360 * frame / frames
        plot_breast_shell(ax, alpha=0.12)
        draw_vessel_curves(ax, center=center)
        plot_tumor_ellipsoid(ax, center=center, radii=radii, color="red", alpha=0.55)
        ax.text(center[0], center[1] + 0.33, center[2] + 0.20, "tumor\nschematic", fontsize=8)
        setup_3d_axes(
            ax,
            title + "\nRotating schematic: vessel-like curves are conceptual",
            azim
        )

    anim = FuncAnimation(fig, update, frames=frames, interval=100)
    anim.save(out_gif, writer=PillowWriter(fps=10))

    update(frames // 5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)


def load_growth_manifest():
    if STAGE11A_GROWTH.exists():
        g = pd.read_csv(STAGE11A_GROWTH, dtype=str).fillna("")
        if not g.empty:
            return g

    nodes = pd.read_csv(PHASE9A_NODES, dtype=str).fillna("")
    for c in ["visit_index", "label_pCR", "ftv_volume"]:
        if c in nodes.columns:
            nodes[c] = pd.to_numeric(nodes[c], errors="coerce")

    best_patient = None
    best_score = -1
    for patient, sub in nodes.groupby("patient_folder"):
        sub = sub.sort_values("visit_index")
        n_visits = len(sub)
        n_vol = sub["ftv_volume"].notna().sum()
        vol_range = sub["ftv_volume"].max() - sub["ftv_volume"].min()
        score = n_visits * 10 + n_vol * 5 + (vol_range if pd.notna(vol_range) else 0)
        if score > best_score:
            best_score = score
            best_patient = patient

    sub = nodes[nodes["patient_folder"].eq(best_patient)].sort_values("visit_index")

    rows = []
    for _, r in sub.iterrows():
        visit = r.get("visit_label", "")
        rows.append({
            "patient_folder": best_patient,
            "visit_label": visit,
            "visit_index": int(r.get("visit_index", 0)),
            "date": "unknown",
            "days_from_first": "unknown",
            "ftv_volume": r.get("ftv_volume", ""),
            "color": {"T0": "red", "T1": "gold", "T2": "green", "T3": "deepskyblue"}.get(visit, "gray"),
            "visual_note": "fallback from Phase 9A nodes"
        })

    return pd.DataFrame(rows)


def make_growth_animation(out_gif, out_png, out_table_png, frames=100):
    g = load_growth_manifest().copy()

    g["visit_index"] = pd.to_numeric(g["visit_index"], errors="coerce")
    g["ftv_volume"] = pd.to_numeric(g["ftv_volume"], errors="coerce")
    g = g.dropna(subset=["visit_index", "ftv_volume"]).sort_values("visit_index")

    if g.empty:
        raise SystemExit("Growth manifest has no usable volume rows.")

    patient = str(g.iloc[0]["patient_folder"])
    max_vol = float(g["ftv_volume"].max())
    if max_vol <= 0:
        max_vol = 1.0

    fig = plt.figure(figsize=(8.8, 7.4))
    ax = fig.add_subplot(111, projection="3d")

    visit_colors = {
        "T0": "red",
        "T1": "gold",
        "T2": "green",
        "T3": "deepskyblue"
    }

    offsets = {
        "T0": np.array([-0.05, 0.00, 0.00]),
        "T1": np.array([-0.02, 0.01, 0.01]),
        "T2": np.array([0.02, -0.01, -0.005]),
        "T3": np.array([0.05, 0.00, 0.005]),
    }

    base_center = np.array([0.18, 0.02, 0.04])

    def update(frame):
        ax.clear()
        azim = 35 + 360 * frame / frames

        # Reveal visits over time.
        reveal = min(len(g), 1 + int(frame / max(frames / len(g), 1)))

        plot_breast_shell(ax, alpha=0.10)

        for idx, (_, row) in enumerate(g.iloc[:reveal].iterrows()):
            visit = str(row["visit_label"])
            vol = float(row["ftv_volume"])
            color = visit_colors.get(visit, str(row.get("color", "gray")))

            radius = 0.28 * (vol / max_vol) ** (1 / 3)
            center = base_center + offsets.get(visit, np.array([0.0, 0.0, 0.0]))

            radii = np.array([radius * 1.10, radius * 0.85, radius * 0.75])
            plot_tumor_ellipsoid(ax, center=center, radii=radii, color=color, alpha=0.45)

            date = row.get("date", "unknown")
            day = row.get("days_from_first", "unknown")
            label = f"{visit}\n{date}\nDay {day}"
            ax.text(center[0], center[1] + 0.35 + 0.04 * idx, center[2] + 0.16, label, fontsize=7)

        setup_3d_axes(
            ax,
            f"Animated multivisit FTV-scaled tumor growth\nPatient: {patient}",
            azim
        )

        legend_items = []
        for _, row in g.iterrows():
            visit = str(row["visit_label"])
            color = visit_colors.get(visit, str(row.get("color", "gray")))
            legend_items.append(Line2D([0], [0], marker="o", color="w", label=visit,
                                       markerfacecolor=color, markersize=8))
        ax.legend(handles=legend_items, loc="upper left", fontsize=7)

    anim = FuncAnimation(fig, update, frames=frames, interval=100)
    anim.save(out_gif, writer=PillowWriter(fps=10))

    update(frames - 1)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)

    # table PNG
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    display = g[["visit_label", "date", "days_from_first", "ftv_volume", "color"]].copy()
    display["ftv_volume"] = pd.to_numeric(display["ftv_volume"], errors="coerce").round(3)

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title(f"Animated growth visit table: {patient}")
    plt.tight_layout()
    plt.savefig(out_table_png, dpi=220)
    plt.close(fig)

    return g, patient


def make_static_storyboard(paths, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.reshape(-1)

    titles = [
        "A. Animated transport proxy still",
        "B. Rotating tumor-vessel schematic still",
        "C. Multivisit tumor growth still",
        "D. Visit date/day table"
    ]

    for ax, path, title in zip(axes, paths, titles):
        if Path(path).exists():
            img = plt.imread(path)
            ax.imshow(img)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, f"Missing:\n{Path(path).name}", ha="center", va="center")
            ax.axis("off")

    fig.suptitle("Stage 11B corrected animated 3D visualization package", fontsize=13)
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
    parser.add_argument("--frames", type=int, default=80)
    args = parser.parse_args()

    case = find_selected_case()
    raw, pre, mask2d, mask3d, frame_idx = load_case_image_and_mask(case)

    patient_case = case.get("patient_folder", "")

    transport_gif = FIGS / "420_stage11b_bidirectional_transport_animation.gif"
    transport_png = FIGS / "420_stage11b_bidirectional_transport_static.png"

    vessel_gif = FIGS / "421_stage11b_rotating_tumor_vessel_animation.gif"
    vessel_png = FIGS / "421_stage11b_rotating_tumor_vessel_static.png"

    growth_gif = FIGS / "422_stage11b_multivisit_growth_animation.gif"
    growth_png = FIGS / "422_stage11b_multivisit_growth_static.png"
    growth_table_png = FIGS / "422_stage11b_multivisit_growth_table.png"

    storyboard_png = FIGS / "423_stage11b_publication_static_storyboard.png"
    summary_png = FIGS / "424_stage11b_animation_file_summary.png"

    make_bidirectional_transport_animation(
        pre,
        mask2d,
        transport_gif,
        transport_png,
        f"Stage 11B transport proxy: {patient_case}",
        frames=args.frames
    )

    make_tumor_vessel_animation(
        vessel_gif,
        vessel_png,
        f"Stage 11B tumor-vessel schematic: {patient_case}",
        frames=args.frames
    )

    growth_manifest, growth_patient = make_growth_animation(
        growth_gif,
        growth_png,
        growth_table_png,
        frames=args.frames
    )

    make_static_storyboard(
        [transport_png, vessel_png, growth_png, growth_table_png],
        storyboard_png
    )

    # Animation manifest.
    manifest = pd.DataFrame([
        {
            "file": "reports/figures/420_stage11b_bidirectional_transport_animation.gif",
            "type": "animated GIF",
            "what_it_shows": "conceptual left-to-right and right-to-left contrast-transport disturbance around tumor",
            "safe_wording": "conceptual transport proxy",
            "unsafe_wording": "measured blood, milk, or fluid flow"
        },
        {
            "file": "reports/figures/421_stage11b_rotating_tumor_vessel_animation.gif",
            "type": "animated GIF",
            "what_it_shows": "rotating transparent breast shell with schematic tumor and vessel-like curves",
            "safe_wording": "schematic tumor-vessel relationship",
            "unsafe_wording": "validated vessel segmentation"
        },
        {
            "file": "reports/figures/422_stage11b_multivisit_growth_animation.gif",
            "type": "animated GIF",
            "what_it_shows": "FTV-scaled multivisit tumor-size visualization with dates/days",
            "safe_wording": "FTV-scaled longitudinal tumor growth visualization",
            "unsafe_wording": "registered voxel-level growth map"
        },
        {
            "file": "reports/figures/423_stage11b_publication_static_storyboard.png",
            "type": "static PNG",
            "what_it_shows": "publication-style still panels from the animations",
            "safe_wording": "visual methods figure",
            "unsafe_wording": "direct physiological proof"
        },
    ])
    manifest.to_csv(TABLES / "stage11b_animation_manifest.csv", index=False)

    # Visual claims table.
    claims = pd.DataFrame([
        {
            "claim": "The GIFs show moving flow.",
            "decision": "allowed only as a conceptual visualization",
            "safe_sentence": "The animation visualizes a conceptual contrast-transport disturbance around the tumor region."
        },
        {
            "claim": "The GIFs show blood or milk flow.",
            "decision": "do not claim",
            "safe_sentence": "The data do not directly measure blood flow, milk flow, or fluid flow."
        },
        {
            "claim": "The tumor is connected with vessels.",
            "decision": "allowed only as schematic",
            "safe_sentence": "The 3D scene shows a schematic tumor-vessel relationship, not validated vessel segmentation."
        },
        {
            "claim": "The tumor grows/shrinks across visits.",
            "decision": "allowed as FTV-scaled visualization",
            "safe_sentence": "The 3D tumor sizes are scaled from visit-level FTV features and dates when metadata are available."
        },
    ])
    claims.to_csv(TABLES / "stage11b_visual_claims_and_limits.csv", index=False)

    growth_manifest.to_csv(TABLES / "stage11b_growth_animation_visit_manifest.csv", index=False)

    make_md(
        TABLES / "stage11b_animation_manifest.csv",
        VIEWS / "stage11b_animation_manifest.md",
        "Stage 11B Animation Manifest"
    )
    make_md(
        TABLES / "stage11b_visual_claims_and_limits.csv",
        VIEWS / "stage11b_visual_claims_and_limits.md",
        "Stage 11B Visual Claims and Limits"
    )
    make_md(
        TABLES / "stage11b_growth_animation_visit_manifest.csv",
        VIEWS / "stage11b_growth_animation_visit_manifest.md",
        "Stage 11B Growth Animation Visit Manifest"
    )

    # Summary figure.
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    txt = (
        "Stage 11B animated visualization fix\n\n"
        "Fixed issue:\n"
        "Stage 11A was a static PNG; Stage 11B creates animated GIFs.\n\n"
        "Created animations:\n"
        "1. Bidirectional transport proxy GIF\n"
        "2. Rotating tumor-vessel schematic GIF\n"
        "3. Multivisit FTV-scaled tumor growth GIF\n\n"
        "Important wording:\n"
        "Conceptual contrast-transport proxy only.\n"
        "Not measured blood, milk, or fluid flow.\n\n"
        f"Flow/tumor-vessel patient: {patient_case}\n"
        f"Growth patient: {growth_patient}"
    )
    ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=13)
    plt.tight_layout()
    plt.savefig(summary_png, dpi=220)
    plt.close()

    # Report.
    report = []
    report.append("# Stage 11B Animated 3D Visualization Fix Report")
    report.append("")
    report.append("This stage fixes the Stage 11A static-image problem by creating animated GIFs.")
    report.append("")
    report.append("## Why this was needed")
    report.append("")
    report.append("Stage 11A created a static PNG. A PNG cannot show 3D movement. Stage 11B creates animated GIFs that rotate the 3D scenes and animate the conceptual transport proxy.")
    report.append("")
    report.append("## Main outputs")
    report.append("")
    report.append("- reports/figures/420_stage11b_bidirectional_transport_animation.gif")
    report.append("- reports/figures/421_stage11b_rotating_tumor_vessel_animation.gif")
    report.append("- reports/figures/422_stage11b_multivisit_growth_animation.gif")
    report.append("- reports/figures/423_stage11b_publication_static_storyboard.png")
    report.append("- reports/figures/424_stage11b_animation_file_summary.png")
    report.append("- reports/tables/stage11b_animation_manifest.csv")
    report.append("- reports/tables/stage11b_visual_claims_and_limits.csv")
    report.append("- reports/tables/stage11b_growth_animation_visit_manifest.csv")
    report.append("")
    report.append("## Selected cases")
    report.append("")
    report.append("- Transport/tumor-vessel case: " + str(patient_case))
    report.append("- Multivisit growth patient: " + str(growth_patient))
    report.append("")
    report.append("## Important scientific limitation")
    report.append("")
    report.append("These animations are conceptual visualizations. They do not measure true blood flow, milk flow, or fluid flow. Use the wording: contrast-transport disturbance proxy.")
    report.append("")
    report.append("## How to use in thesis")
    report.append("")
    report.append("Use the GIFs for presentation. Use the static storyboard PNG for the paper or thesis document.")
    report.append("")
    report.append("## Next step")
    report.append("")
    report.append("Stage 11C should create one final polished still figure for the thesis from the best Stage 11B panels.")

    (REPORTS / "Stage11B_Animated_3D_Visualization_Fix_Report.md").write_text("\n".join(report))

    # Dashboard / README.
    dash = REPORTS / "Dashboard.md"
    old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
    dash_add = (
        "\n\n## Stage 11B animated 3D visualization fix\n\n"
        "- [Animated 3D visualization fix report](Stage11B_Animated_3D_Visualization_Fix_Report.md)\n"
        "- [Animation manifest](table_views/stage11b_animation_manifest.md)\n"
        "- [Visual claims and limits](table_views/stage11b_visual_claims_and_limits.md)\n"
    )
    if "Stage 11B animated 3D visualization fix" not in old_dash:
        dash.write_text(old_dash.rstrip() + dash_add)

    readme = REPO / "README.md"
    old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
    readme_add = (
        "\n\n### Stage 11B: animated 3D visualization fix\n\n"
        "Main outputs:\n\n"
        "- reports/Stage11B_Animated_3D_Visualization_Fix_Report.md\n"
        "- reports/figures/420_stage11b_bidirectional_transport_animation.gif\n"
        "- reports/figures/421_stage11b_rotating_tumor_vessel_animation.gif\n"
        "- reports/figures/422_stage11b_multivisit_growth_animation.gif\n"
    )
    if "Stage 11B: animated 3D visualization fix" not in old_readme:
        readme.write_text(old_readme.rstrip() + readme_add)

    print("Stage 11B complete.")
    print("Transport/tumor-vessel patient:", patient_case)
    print("Growth patient:", growth_patient)
    print("Created GIF:", transport_gif)
    print("Created GIF:", vessel_gif)
    print("Created GIF:", growth_gif)
    print("Important: conceptual proxy only, not measured flow.")


if __name__ == "__main__":
    main()
