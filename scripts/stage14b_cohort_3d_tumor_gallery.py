"""
Stage 14B: Cohort 3D tumor-model gallery.

Renders the real-mask-derived 3D tumor models for several cohort patients in one
figure (a grid of 3D tumor surfaces + spatial node graphs), so the paper can show
that the 3D tumor pipeline generalises across patients, not just one case.

Same method/wording as Stage 13/14A: spatial tumor-node graphs from recovered
fractional masks. Edges are spatial links, not vessels or measured flow.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import pydicom
from scipy import ndimage as ndi
from skimage import measure, morphology
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

REPO = Path.home() / "ISPY2-3DGCNN"
REPORTS = REPO / "reports"
TABLES = REPORTS / "tables"
FIGS = REPORTS / "figures"
for p in [REPORTS, TABLES, FIGS]:
    p.mkdir(parents=True, exist_ok=True)

PHASE7C = TABLES / "phase7c_fractional_tumor_mask_candidate_table.csv"
MERGE = TABLES / "stage14a_cohort_label_merge.csv"


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def read_seg_array(path):
    ds = pydicom.dcmread(path, force=True)
    arr = np.squeeze(np.asarray(ds.pixel_array))
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 4:
        arr = np.squeeze(arr)
        if arr.ndim == 4:
            arr = arr[..., 0]
    if arr.ndim != 3:
        raise RuntimeError(f"bad shape {arr.shape}")
    return arr.astype(float)

def largest_component_3d(mask):
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return mask
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)

def clean_mask3d(mask):
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return mask
    mask = morphology.remove_small_objects(mask, min_size=30)
    mask = largest_component_3d(mask)
    out = np.zeros_like(mask, dtype=bool)
    for z in range(mask.shape[0]):
        out[z] = ndi.binary_fill_holes(mask[z])
    mask = out
    if mask.sum() > 0:
        mask = morphology.binary_closing(mask, morphology.ball(1))
    return largest_component_3d(mask).astype(bool)

def crop_bbox_3d(mask, pad=2):
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return mask
    c = np.argwhere(mask)
    z0, y0, x0 = c.min(axis=0)
    z1, y1, x1 = c.max(axis=0) + 1
    z0, y0, x0 = max(0, z0 - pad), max(0, y0 - pad), max(0, x0 - pad)
    z1 = min(mask.shape[0], z1 + pad); y1 = min(mask.shape[1], y1 + pad); x1 = min(mask.shape[2], x1 + pad)
    return mask[z0:z1, y0:y1, x0:x1]

def ensure_usable_3d(mask3d):
    mask3d = clean_mask3d(crop_bbox_3d(mask3d, pad=2))
    if mask3d.sum() == 0:
        return mask3d
    c = np.argwhere(mask3d)
    if int(c[:, 0].max() - c[:, 0].min() + 1) >= 4:
        return mask3d
    return mask3d  # keep as-is; gallery tolerates thin masks

def quality_score(mask3d):
    if mask3d.sum() == 0:
        return -999
    c = np.argwhere(mask3d)
    z = int(c[:, 0].max() - c[:, 0].min() + 1)
    y = int(c[:, 1].max() - c[:, 1].min() + 1)
    x = int(c[:, 2].max() - c[:, 2].min() + 1)
    occ = mask3d.sum() / max(z * y * x, 1)
    ratio = max(z, y, x) / max(min(z, y, x), 1)
    s = min(mask3d.sum(), 80000) / 1000.0
    s += 25 if z >= 4 else -20
    s += 25 if x > 5 and y > 5 else -30
    s += 20 if 0.02 <= occ <= 0.90 else -15
    s += 20 if ratio <= 15 else -25
    return float(s)

def make_mesh(mask3d, max_faces=12000):
    padded = np.pad(mask3d.astype(bool), 1, mode="constant", constant_values=False)
    verts, faces, _, _ = measure.marching_cubes(padded.astype(float), level=0.5)
    mesh_xyz = np.vstack([verts[:, 2], verts[:, 1], verts[:, 0]]).T
    center = mesh_xyz.mean(axis=0)
    scale = max(np.ptp(mesh_xyz[:, 0]), np.ptp(mesh_xyz[:, 1]), np.ptp(mesh_xyz[:, 2]), 1.0)
    def transform(raw):
        out = (raw - center) / scale
        out[:, 0] *= 1.10; out[:, 1] *= 0.85; out[:, 2] *= 0.75
        return out
    coords = transform(mesh_xyz)
    if faces.shape[0] > max_faces:
        faces = faces[::int(np.ceil(faces.shape[0] / max_faces))]
    return coords, faces, transform

def tumor_graph(mask3d, transform, max_nodes=40, knn=4):
    vz = np.argwhere(mask3d.astype(bool))
    raw = np.vstack([vz[:, 2] + 1, vz[:, 1] + 1, vz[:, 0] + 1]).T.astype(float)
    xyz = transform(raw)
    n = xyz.shape[0]
    k = min(max_nodes, max(8, int(np.sqrt(n))));  k = min(k, n)
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(xyz)
    centers = km.cluster_centers_
    depth = ndi.distance_transform_edt(mask3d.astype(bool))
    vdepth = depth[vz[:, 0], vz[:, 1], vz[:, 2]]
    p60 = np.percentile(vdepth, 60)
    region = []
    for cid in range(k):
        idx = np.where(km.labels_ == cid)[0]
        region.append("core" if np.mean(vdepth[idx]) > p60 else "periphery")
    nbrs = NearestNeighbors(n_neighbors=min(knn + 1, len(centers))).fit(centers)
    dist, ind = nbrs.kneighbors(centers)
    edges = set()
    for i in range(len(centers)):
        for j in ind[i][1:]:
            edges.add(tuple(sorted([int(i), int(j)])))
    return centers, list(edges), region

def render_panel(ax, mask3d, title):
    coords, faces, transform = make_mesh(mask3d)
    mesh = Poly3DCollection(coords[faces], alpha=0.30)
    mesh.set_facecolor("#d6455f"); mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)
    centers, edges, region = tumor_graph(mask3d, transform)
    for a, b in edges:
        ax.plot([centers[a, 0], centers[b, 0]], [centers[a, 1], centers[b, 1]],
                [centers[a, 2], centers[b, 2]], color="black", linewidth=0.8, alpha=0.5)
    cols = ["#1f77b4" if r == "core" else "#ff7f0e" for r in region]
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=cols, s=18, depthshade=True)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    lim = 0.6
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.view_init(elev=18, azim=35)

def best_mask_for_patient(cand, pid):
    sub = cand[cand["patient_folder"].eq(pid)]
    best = None
    for _, r in sub.iterrows():
        seg = r.get("seg_file", ""); thr = safe_float(r.get("selected_threshold", ""))
        try:
            if not seg or not Path(seg).exists():
                continue
            m = ensure_usable_3d(read_seg_array(seg) >= thr)
            if m.sum() == 0:
                continue
            q = quality_score(m)
            if best is None or q > best[0]:
                best = (q, m)
        except Exception:
            continue
    return None if best is None else best[1]

def pick_patients(n_panels):
    if MERGE.exists():
        mg = pd.read_csv(MERGE)
        labeled = mg[mg["has_label"] == True]["patient_folder"].tolist() if "has_label" in mg else []
        unl = mg[(mg.get("has_label") != True)].sort_values("tumor_voxels", ascending=False)["patient_folder"].tolist() \
              if "tumor_voxels" in mg else mg["patient_folder"].tolist()
        chosen = labeled + [p for p in unl if p not in labeled]
        return chosen[:n_panels]
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", default="", help="comma list; default = labeled + largest unlabeled")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    if not PHASE7C.exists():
        raise SystemExit("Missing phase7c table.")
    cand = pd.read_csv(PHASE7C, dtype=str).fillna("")
    cand = cand[cand["recovery_status"].eq("fractional_tumor_candidate_pass")]

    pats = [p.strip() for p in args.patients.split(",") if p.strip()] or pick_patients(args.n)
    pats = pats[:args.n]
    print(f"[info] rendering 3D tumor models for: {pats}")

    masks = []
    for pid in pats:
        m = best_mask_for_patient(cand, pid)
        if m is not None and m.sum() >= 8:
            masks.append((pid, m))
            print(f"  {pid}: voxels={int(m.sum())}")
        else:
            print(f"  {pid}: skipped (no usable mask)")

    if not masks:
        raise SystemExit("No masks to render.")

    ncol = 3
    nrow = int(np.ceil(len(masks) / ncol))
    fig = plt.figure(figsize=(4.2 * ncol, 4.2 * nrow))
    for i, (pid, m) in enumerate(masks, 1):
        ax = fig.add_subplot(nrow, ncol, i, projection="3d")
        render_panel(ax, m, pid)
    fig.suptitle("Stage 14B: real-mask-derived 3D tumor models across the ISPY2 cohort\n"
                 "(blue = core nodes, orange = periphery; edges are spatial links, not vessels)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIGS / "710_stage14b_cohort_3d_tumor_gallery.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[done] wrote {out}")

    rep = REPORTS / "Stage14B_Cohort_3D_Tumor_Gallery_Report.md"
    rep.write_text(
        "# Stage 14B Cohort 3D Tumor-Model Gallery\n\n"
        f"Rendered real-mask-derived 3D tumor models for {len(masks)} cohort patients in one figure.\n"
        "Each panel shows a marching-cubes tumor surface plus its spatial tumor-node graph\n"
        "(blue = core, orange = periphery). Edges are spatial links, not vessels or measured flow.\n\n"
        "## Patients shown\n" + "\n".join(f"- {pid} (voxels={int(m.sum())})" for pid, m in masks) + "\n\n"
        "## Output\n- reports/figures/710_stage14b_cohort_3d_tumor_gallery.png\n\n"
        "## Limitation\nRecovered fractional masks, not expert segmentations. Pilot feasibility work.\n")
    print(f"[done] wrote {rep}")

if __name__ == "__main__":
    main()
