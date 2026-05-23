from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
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
VIEWS = REPORTS / "table_views"
INTERACTIVE = REPORTS / "interactive"

for p in [REPORTS, TABLES, FIGS, VIEWS, INTERACTIVE]:
    p.mkdir(parents=True, exist_ok=True)

PHASE7C = TABLES / "phase7c_fractional_tumor_mask_candidate_table.csv"

if not PHASE7C.exists():
    raise SystemExit("Missing reports/tables/phase7c_fractional_tumor_mask_candidate_table.csv. Run Phase 7C first.")


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


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def largest_component_3d(mask):
    mask = mask.astype(bool)

    if mask.sum() == 0:
        return mask

    lab, n = ndi.label(mask)

    if n == 0:
        return mask

    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    keep = int(np.argmax(sizes)) + 1

    return lab == keep


def fill_holes_slice_wise(mask):
    out = np.zeros_like(mask, dtype=bool)
    for z in range(mask.shape[0]):
        out[z] = ndi.binary_fill_holes(mask[z])
    return out


def clean_mask3d(mask):
    mask = mask.astype(bool)

    if mask.sum() == 0:
        return mask

    mask = morphology.remove_small_objects(mask, min_size=30)
    mask = largest_component_3d(mask)
    mask = fill_holes_slice_wise(mask)

    try:
        mask = morphology.binary_closing(mask, morphology.ball(1))
    except Exception:
        pass

    mask = largest_component_3d(mask)

    return mask.astype(bool)


def crop_bbox_3d(mask, pad=2):
    mask = mask.astype(bool)

    if mask.sum() == 0:
        return mask

    coords = np.argwhere(mask)

    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1

    z0 = max(0, z0 - pad)
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)

    z1 = min(mask.shape[0], z1 + pad)
    y1 = min(mask.shape[1], y1 + pad)
    x1 = min(mask.shape[2], x1 + pad)

    return mask[z0:z1, y0:y1, x0:x1]


def pseudo_3d_from_2d(mask2d, n_slices=13):
    """
    If the recovered mask is too flat, build a pseudo-3D tumor
    from the real 2D tumor outline. This is better than a fake sphere,
    but it must be labeled as pseudo-3D.
    """
    mask2d = mask2d.astype(bool)

    if mask2d.sum() == 0:
        return np.zeros((n_slices,) + mask2d.shape, dtype=bool)

    d = ndi.distance_transform_edt(mask2d)
    max_d = max(float(d.max()), 1.0)

    slices = []
    center = (n_slices - 1) / 2.0

    for i in range(n_slices):
        dz = abs(i - center) / max(center, 1.0)
        shrink = dz * max_d * 0.80

        sl = d > shrink
        sl = morphology.remove_small_objects(sl, min_size=20)
        sl = ndi.binary_fill_holes(sl)

        slices.append(sl.astype(bool))

    return np.stack(slices, axis=0)


def ensure_usable_3d(mask3d):
    """
    Try to use the recovered 3D mask. If it is too thin,
    build pseudo-3D from the largest real 2D mask outline.
    """
    mask3d = crop_bbox_3d(mask3d, pad=2)
    mask3d = clean_mask3d(mask3d)

    if mask3d.sum() == 0:
        return mask3d, "empty"

    coords = np.argwhere(mask3d)
    z_extent = int(coords[:, 0].max() - coords[:, 0].min() + 1)

    if z_extent >= 4:
        return mask3d, "real_recovered_3d_mask"

    frame_pixels = mask3d.reshape(mask3d.shape[0], -1).sum(axis=1)
    z = int(np.argmax(frame_pixels))
    mask2d = mask3d[z]

    pseudo = pseudo_3d_from_2d(mask2d, n_slices=13)
    pseudo = clean_mask3d(pseudo)

    return pseudo, "pseudo_3d_from_real_2d_mask_outline"


def quality_metrics(mask3d):
    if mask3d.sum() == 0:
        return {
            "voxels": 0,
            "z_extent": 0,
            "y_extent": 0,
            "x_extent": 0,
            "bbox_occupancy": 0,
            "shape_ratio": 999,
            "quality_score": -999
        }

    coords = np.argwhere(mask3d)

    z_extent = int(coords[:, 0].max() - coords[:, 0].min() + 1)
    y_extent = int(coords[:, 1].max() - coords[:, 1].min() + 1)
    x_extent = int(coords[:, 2].max() - coords[:, 2].min() + 1)

    bbox_vol = max(z_extent * y_extent * x_extent, 1)
    occupancy = float(mask3d.sum() / bbox_vol)
    ratio = max(z_extent, y_extent, x_extent) / max(min(z_extent, y_extent, x_extent), 1)

    score = 0.0
    score += min(mask3d.sum(), 80000) / 1000.0
    score += 25 if z_extent >= 4 else -20
    score += 25 if x_extent > 5 and y_extent > 5 else -30
    score += 20 if 0.02 <= occupancy <= 0.90 else -15
    score += 20 if ratio <= 15 else -25

    return {
        "voxels": int(mask3d.sum()),
        "z_extent": z_extent,
        "y_extent": y_extent,
        "x_extent": x_extent,
        "bbox_occupancy": occupancy,
        "shape_ratio": ratio,
        "quality_score": float(score)
    }


def build_candidate_masks(max_cases=200):
    cand = pd.read_csv(PHASE7C, dtype=str).fillna("")
    cand = cand[cand["recovery_status"].eq("fractional_tumor_candidate_pass")].copy()

    if cand.empty:
        raise SystemExit("No fractional_tumor_candidate_pass cases found in Phase 7C.")

    rows = []
    masks = {}

    for i, r in cand.head(max_cases).iterrows():
        seg_file = r.get("seg_file", "")
        threshold = safe_float(r.get("selected_threshold", ""))

        row = {
            "patient_folder": r.get("patient_folder", ""),
            "study_folder": r.get("study_folder", ""),
            "seg_series_uid": r.get("seg_series_uid", ""),
            "seg_file": seg_file,
            "selected_threshold": threshold,
            "status": "not_started"
        }

        try:
            if not Path(seg_file).exists():
                row["status"] = "missing_seg_file"
                rows.append(row)
                continue

            _, arr = read_seg_array(seg_file)
            raw_mask = arr >= threshold

            mask3d, mode = ensure_usable_3d(raw_mask)
            metrics = quality_metrics(mask3d)

            row.update(metrics)
            row["mask_mode"] = mode
            row["status"] = "success"

            masks[row["seg_series_uid"]] = mask3d

        except Exception as e:
            row["status"] = "failed_exception"
            row["error"] = str(e)[:200]

        rows.append(row)

    q = pd.DataFrame(rows)
    q.to_csv(TABLES / "stage13a_candidate_mask_quality.csv", index=False)

    ok = q[q["status"].eq("success")].copy()
    ok["quality_score"] = pd.to_numeric(ok["quality_score"], errors="coerce")
    ok = ok.sort_values("quality_score", ascending=False)

    if ok.empty:
        raise SystemExit("No successful masks after cleaning.")

    best = ok.iloc[0].to_dict()
    best_mask = masks[best["seg_series_uid"]]

    return best, best_mask, q


def make_mesh(mask3d, max_faces=15000):
    padded = np.pad(mask3d.astype(bool), 1, mode="constant", constant_values=False)
    verts, faces, normals, values = measure.marching_cubes(padded.astype(float), level=0.5)

    # marching_cubes gives z,y,x. Convert to x,y,z raw.
    mesh_xyz = np.vstack([verts[:, 2], verts[:, 1], verts[:, 0]]).T

    center = mesh_xyz.mean(axis=0)
    scale = max(np.ptp(mesh_xyz[:, 0]), np.ptp(mesh_xyz[:, 1]), np.ptp(mesh_xyz[:, 2]), 1.0)

    def transform(raw_xyz):
        out = (raw_xyz - center) / scale
        out[:, 0] *= 1.10
        out[:, 1] *= 0.85
        out[:, 2] *= 0.75
        out += np.array([0.18, 0.02, 0.04])
        return out

    coords = transform(mesh_xyz)

    # Downsample faces only for plotting if needed.
    if faces.shape[0] > max_faces:
        step = int(np.ceil(faces.shape[0] / max_faces))
        faces = faces[::step]

    return coords, faces, transform


def create_tumor_graph(mask3d, transform, max_nodes=40, knn=4):
    voxel_zyx = np.argwhere(mask3d.astype(bool))

    if voxel_zyx.shape[0] < 8:
        raise RuntimeError("Too few tumor voxels to create graph.")

    # raw xyz with +1 because mesh was padded before transform
    raw_xyz = np.vstack([
        voxel_zyx[:, 2] + 1,
        voxel_zyx[:, 1] + 1,
        voxel_zyx[:, 0] + 1
    ]).T.astype(float)

    xyz = transform(raw_xyz)

    n_vox = xyz.shape[0]
    n_clusters = min(max_nodes, max(8, int(np.sqrt(n_vox))))
    n_clusters = min(n_clusters, n_vox)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_id = kmeans.fit_predict(xyz)

    centers = kmeans.cluster_centers_

    # distance transform gives internal depth, a useful node feature
    depth_map = ndi.distance_transform_edt(mask3d.astype(bool))
    voxel_depth = depth_map[voxel_zyx[:, 0], voxel_zyx[:, 1], voxel_zyx[:, 2]]

    tumor_center = centers.mean(axis=0)

    node_rows = []
    for cid in range(n_clusters):
        idx = np.where(cluster_id == cid)[0]
        pts = xyz[idx]

        centroid = pts.mean(axis=0)
        radial = float(np.linalg.norm(centroid - tumor_center))

        node_rows.append({
            "node_id": int(cid),
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "centroid_z": float(centroid[2]),
            "voxel_count": int(len(idx)),
            "volume_fraction": float(len(idx) / n_vox),
            "mean_internal_depth": float(np.mean(voxel_depth[idx])),
            "radial_distance": radial,
            "graph_region": "core" if np.mean(voxel_depth[idx]) > np.percentile(voxel_depth, 60) else "periphery"
        })

    nodes = pd.DataFrame(node_rows)

    # kNN edges
    nbrs = NearestNeighbors(n_neighbors=min(knn + 1, len(centers))).fit(centers)
    distances, indices = nbrs.kneighbors(centers)

    edge_set = set()
    edge_rows = []

    for i in range(len(centers)):
        for j_idx, d in zip(indices[i][1:], distances[i][1:]):
            a = int(i)
            b = int(j_idx)
            pair = tuple(sorted([a, b]))

            if pair in edge_set:
                continue

            edge_set.add(pair)

            edge_rows.append({
                "source": pair[0],
                "target": pair[1],
                "edge_distance": float(d),
                "edge_type": "spatial_knn"
            })

    edges = pd.DataFrame(edge_rows)

    return nodes, edges, xyz


def plot_breast_shell(ax, alpha=0.08):
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)

    x = 1.25 * np.outer(np.cos(u), np.sin(v))
    y = 0.90 * np.outer(np.sin(u), np.sin(v))
    z = 0.75 * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, color="lightgray", alpha=alpha, linewidth=0, shade=True)


def add_mesh(ax, coords, faces, color="red", alpha=0.45):
    mesh = Poly3DCollection(coords[faces], alpha=alpha)
    mesh.set_facecolor(color)
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)


def add_graph(ax, nodes, edges, color_by="graph_region"):
    pos = nodes.set_index("node_id")[["centroid_x", "centroid_y", "centroid_z"]].to_dict("index")

    # edges
    for _, e in edges.iterrows():
        s = int(e["source"])
        t = int(e["target"])
        if s not in pos or t not in pos:
            continue

        xs = [pos[s]["centroid_x"], pos[t]["centroid_x"]]
        ys = [pos[s]["centroid_y"], pos[t]["centroid_y"]]
        zs = [pos[s]["centroid_z"], pos[t]["centroid_z"]]

        ax.plot(xs, ys, zs, color="black", linewidth=1.1, alpha=0.55)

    # nodes
    if color_by == "graph_region":
        colors = ["gold" if r == "core" else "deepskyblue" for r in nodes["graph_region"]]
    else:
        vals = pd.to_numeric(nodes[color_by], errors="coerce")
        colors = vals

    sizes = 35 + 180 * pd.to_numeric(nodes["volume_fraction"], errors="coerce").fillna(0).values

    ax.scatter(
        nodes["centroid_x"],
        nodes["centroid_y"],
        nodes["centroid_z"],
        s=sizes,
        c=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.95
    )


def setup_ax(ax, title, azim=35):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("left-right")
    ax.set_ylabel("anterior-posterior")
    ax.set_zlabel("depth")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.95, 0.95)
    ax.set_zlim(-0.75, 0.75)
    ax.view_init(elev=18, azim=azim)
    ax.set_box_aspect([1.3, 1.0, 0.8])


def save_static_graph(coords, faces, nodes, edges, best, out_path, title_extra=""):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    plot_breast_shell(ax, alpha=0.08)
    add_mesh(ax, coords, faces, color="red", alpha=0.35)
    add_graph(ax, nodes, edges)

    setup_ax(
        ax,
        "Real-mask-derived 3D tumor graph\n"
        + f"Patient: {best.get('patient_folder', '')} "
        + title_extra,
        azim=35
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_zoom_graph(coords, faces, nodes, edges, best, out_path):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    add_mesh(ax, coords, faces, color="red", alpha=0.30)
    add_graph(ax, nodes, edges)

    ax.set_title("Tumor graph zoom: nodes and spatial edges", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    pad = 0.15

    ax.set_xlim(mins[0] - pad, maxs[0] + pad)
    ax.set_ylim(mins[1] - pad, maxs[1] + pad)
    ax.set_zlim(mins[2] - pad, maxs[2] + pad)

    ax.view_init(elev=22, azim=45)
    ax.set_box_aspect([1.1, 0.9, 0.8])

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_feature_map(nodes, edges, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))

    x = nodes["radial_distance"]
    y = nodes["mean_internal_depth"]
    s = 80 + 800 * nodes["volume_fraction"]

    colors = ["gold" if r == "core" else "deepskyblue" for r in nodes["graph_region"]]

    ax.scatter(x, y, s=s, c=colors, edgecolor="black", alpha=0.85)

    for _, row in nodes.iterrows():
        ax.text(row["radial_distance"], row["mean_internal_depth"], str(int(row["node_id"])), fontsize=7)

    ax.set_xlabel("Radial distance from tumor graph center")
    ax.set_ylabel("Mean internal depth")
    ax.set_title("Tumor graph node feature map\ncore vs periphery")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_rotating_gif(coords, faces, nodes, edges, best, out_gif, out_png, frames=60):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(i):
        ax.clear()
        azim = 35 + 360 * i / frames
        plot_breast_shell(ax, alpha=0.08)
        add_mesh(ax, coords, faces, color="red", alpha=0.35)
        add_graph(ax, nodes, edges)
        setup_ax(
            ax,
            f"Rotating real 3D tumor graph\nPatient: {best.get('patient_folder', '')}",
            azim=azim
        )

    anim = FuncAnimation(fig, update, frames=frames, interval=100)
    anim.save(out_gif, writer=PillowWriter(fps=10))

    update(frames // 5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)


def save_plotly_html(coords, faces, nodes, edges, out_html):
    try:
        import plotly.graph_objects as go
    except Exception:
        return False

    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]

    mesh = go.Mesh3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        i=i,
        j=j,
        k=k,
        color="red",
        opacity=0.25,
        name="tumor surface"
    )

    edge_x = []
    edge_y = []
    edge_z = []

    pos = nodes.set_index("node_id")[["centroid_x", "centroid_y", "centroid_z"]].to_dict("index")

    for _, e in edges.iterrows():
        s = int(e["source"])
        t = int(e["target"])

        if s not in pos or t not in pos:
            continue

        edge_x += [pos[s]["centroid_x"], pos[t]["centroid_x"], None]
        edge_y += [pos[s]["centroid_y"], pos[t]["centroid_y"], None]
        edge_z += [pos[s]["centroid_z"], pos[t]["centroid_z"], None]

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(color="black", width=3),
        name="graph edges"
    )

    node_color = [1 if r == "core" else 0 for r in nodes["graph_region"]]

    node_trace = go.Scatter3d(
        x=nodes["centroid_x"],
        y=nodes["centroid_y"],
        z=nodes["centroid_z"],
        mode="markers+text",
        text=nodes["node_id"].astype(str),
        marker=dict(
            size=6 + 18 * nodes["volume_fraction"],
            color=node_color,
            colorscale="Viridis",
            line=dict(color="black", width=1)
        ),
        name="graph nodes"
    )

    fig = go.Figure(data=[mesh, edge_trace, node_trace])
    fig.update_layout(
        title="Interactive real-mask-derived 3D tumor graph",
        scene=dict(
            xaxis_title="left-right",
            yaxis_title="anterior-posterior",
            zaxis_title="depth"
        ),
        width=1100,
        height=800
    )

    fig.write_html(out_html)
    return True


def make_md(csv_path, md_path, title, max_rows=100, max_cols=12):
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
    parser.add_argument("--nodes", type=int, default=40)
    parser.add_argument("--knn", type=int, default=4)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--max-cases", type=int, default=200)
    args = parser.parse_args()

    best, mask3d, quality = build_candidate_masks(max_cases=args.max_cases)

    coords, faces, transform = make_mesh(mask3d)
    nodes, edges, voxel_xyz = create_tumor_graph(mask3d, transform, max_nodes=args.nodes, knn=args.knn)

    nodes["patient_folder"] = best.get("patient_folder", "")
    nodes["seg_series_uid"] = best.get("seg_series_uid", "")
    edges["patient_folder"] = best.get("patient_folder", "")
    edges["seg_series_uid"] = best.get("seg_series_uid", "")

    nodes.to_csv(TABLES / "stage13a_tumor_graph_nodes.csv", index=False)
    edges.to_csv(TABLES / "stage13a_tumor_graph_edges.csv", index=False)

    selected = pd.DataFrame([{
        "patient_folder": best.get("patient_folder", ""),
        "study_folder": best.get("study_folder", ""),
        "seg_series_uid": best.get("seg_series_uid", ""),
        "seg_file": best.get("seg_file", ""),
        "selected_threshold": best.get("selected_threshold", ""),
        "mask_mode": best.get("mask_mode", ""),
        "voxels": best.get("voxels", ""),
        "z_extent": best.get("z_extent", ""),
        "y_extent": best.get("y_extent", ""),
        "x_extent": best.get("x_extent", ""),
        "bbox_occupancy": best.get("bbox_occupancy", ""),
        "shape_ratio": best.get("shape_ratio", ""),
        "quality_score": best.get("quality_score", ""),
        "n_graph_nodes": len(nodes),
        "n_graph_edges": len(edges)
    }])
    selected.to_csv(TABLES / "stage13a_selected_real_tumor_graph_case.csv", index=False)

    summary = pd.DataFrame([{
        "patient_folder": best.get("patient_folder", ""),
        "mask_mode": best.get("mask_mode", ""),
        "n_graph_nodes": len(nodes),
        "n_graph_edges": len(edges),
        "core_nodes": int((nodes["graph_region"] == "core").sum()),
        "periphery_nodes": int((nodes["graph_region"] == "periphery").sum()),
        "mean_edge_distance": float(edges["edge_distance"].mean()) if len(edges) else np.nan,
        "main_output_png": "reports/figures/600_stage13a_real_tumor_surface_graph.png",
        "interactive_html": "reports/interactive/604_stage13a_interactive_tumor_graph.html"
    }])
    summary.to_csv(TABLES / "stage13a_tumor_graph_summary.csv", index=False)

    claims = pd.DataFrame([
        {
            "component": "tumor surface",
            "safe_claim": "The tumor surface is derived from a recovered fractional tumor mask.",
            "limit": "It is not a manual clinical segmentation unless clinically verified."
        },
        {
            "component": "graph nodes",
            "safe_claim": "Graph nodes are spatial clusters of tumor voxels.",
            "limit": "They are computational graph regions, not histologic regions."
        },
        {
            "component": "graph edges",
            "safe_claim": "Edges connect nearby tumor graph nodes in 3D space.",
            "limit": "Edges are not measured vessels."
        },
        {
            "component": "pseudo-3D warning",
            "safe_claim": "If the recovered mask is too thin, pseudo-3D is built from the real 2D outline.",
            "limit": "Pseudo-3D is a visualization approximation and must be labeled clearly."
        }
    ])
    claims.to_csv(TABLES / "stage13a_tumor_graph_claims_and_limits.csv", index=False)

    save_static_graph(
        coords,
        faces,
        nodes,
        edges,
        best,
        FIGS / "600_stage13a_real_tumor_surface_graph.png",
        title_extra="| mask-derived"
    )

    save_zoom_graph(
        coords,
        faces,
        nodes,
        edges,
        best,
        FIGS / "601_stage13a_tumor_graph_zoom.png"
    )

    save_feature_map(
        nodes,
        edges,
        FIGS / "602_stage13a_graph_node_feature_map.png"
    )

    save_rotating_gif(
        coords,
        faces,
        nodes,
        edges,
        best,
        FIGS / "603_stage13a_rotating_tumor_graph.gif",
        FIGS / "603_stage13a_rotating_tumor_graph_static.png",
        frames=args.frames
    )

    html_ok = save_plotly_html(
        coords,
        faces,
        nodes,
        edges,
        INTERACTIVE / "604_stage13a_interactive_tumor_graph.html"
    )

    # schema figure
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    txt = (
        "Stage 13A real 3D tumor graph\n\n"
        "Input: recovered tumor mask\n"
        "↓\n"
        "Clean 3D mask / pseudo-3D if too thin\n"
        "↓\n"
        "Marching-cubes tumor surface mesh\n"
        "↓\n"
        "K-means tumor voxel clusters = graph nodes\n"
        "↓\n"
        "k-nearest-neighbor spatial links = graph edges\n\n"
        "This is a tumor graph, not a vessel graph.\n"
        "Edges are spatial graph links, not measured blood/milk/fluid flow."
    )
    ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGS / "605_stage13a_tumor_graph_schema.png", dpi=220)
    plt.close()

    make_md(TABLES / "stage13a_candidate_mask_quality.csv", VIEWS / "stage13a_candidate_mask_quality.md", "Stage 13A Candidate Mask Quality")
    make_md(TABLES / "stage13a_selected_real_tumor_graph_case.csv", VIEWS / "stage13a_selected_real_tumor_graph_case.md", "Stage 13A Selected Real Tumor Graph Case")
    make_md(TABLES / "stage13a_tumor_graph_nodes.csv", VIEWS / "stage13a_tumor_graph_nodes.md", "Stage 13A Tumor Graph Nodes")
    make_md(TABLES / "stage13a_tumor_graph_edges.csv", VIEWS / "stage13a_tumor_graph_edges.md", "Stage 13A Tumor Graph Edges")
    make_md(TABLES / "stage13a_tumor_graph_summary.csv", VIEWS / "stage13a_tumor_graph_summary.md", "Stage 13A Tumor Graph Summary")
    make_md(TABLES / "stage13a_tumor_graph_claims_and_limits.csv", VIEWS / "stage13a_tumor_graph_claims_and_limits.md", "Stage 13A Tumor Graph Claims and Limits")

    report = []
    report.append("# Stage 13A Real 3D Tumor Graph From Recovered Mask Report")
    report.append("")
    report.append("This stage fixes the problem that earlier 3D tumor visuals looked schematic or unrealistic.")
    report.append("")
    report.append("## What this stage does")
    report.append("")
    report.append("- Uses the recovered tumor mask instead of an artificial ellipsoid.")
    report.append("- Builds a real 3D tumor surface mesh using marching cubes.")
    report.append("- Converts tumor voxels into graph nodes using spatial clustering.")
    report.append("- Connects nodes using 3D nearest-neighbor edges.")
    report.append("- Creates static, animated, and interactive visual outputs.")
    report.append("")
    report.append("## Main outputs")
    report.append("")
    report.append("- reports/tables/stage13a_tumor_graph_nodes.csv")
    report.append("- reports/tables/stage13a_tumor_graph_edges.csv")
    report.append("- reports/figures/600_stage13a_real_tumor_surface_graph.png")
    report.append("- reports/figures/601_stage13a_tumor_graph_zoom.png")
    report.append("- reports/figures/602_stage13a_graph_node_feature_map.png")
    report.append("- reports/figures/603_stage13a_rotating_tumor_graph.gif")
    report.append("- reports/figures/605_stage13a_tumor_graph_schema.png")
    report.append("- reports/interactive/604_stage13a_interactive_tumor_graph.html")
    report.append("")
    report.append("## Selected tumor graph case")
    report.append("")
    report.append("- Patient: " + str(best.get("patient_folder", "")))
    report.append("- Mask mode: " + str(best.get("mask_mode", "")))
    report.append("- Tumor voxels: " + str(best.get("voxels", "")))
    report.append("- Graph nodes: " + str(len(nodes)))
    report.append("- Graph edges: " + str(len(edges)))
    report.append("- Interactive HTML created: " + str(html_ok))
    report.append("")
    report.append("## Important interpretation")
    report.append("")
    report.append("This is now a true tumor graph visualization because the graph is built from recovered tumor-mask voxels. It is not a fake ellipsoid.")
    report.append("")
    report.append("## Limitation")
    report.append("")
    report.append("The mask is still a recovered fractional mask. If mask_mode is pseudo_3d_from_real_2d_mask_outline, then the 3D thickness is a visualization approximation based on the real 2D outline.")
    report.append("")
    report.append("## Safe wording")
    report.append("")
    report.append("Use: real-mask-derived tumor graph, spatial tumor-node graph, and contrast-transport proxy.")
    report.append("")
    report.append("Avoid: measured blood flow, milk flow, vessel graph, or clinically validated 3D tumor segmentation.")
    report.append("")
    report.append("## Next step")
    report.append("")
    report.append("Use Stage 13A tumor graph figures as the main 3D tumor graph visuals. Do not use earlier ellipsoid figures as final tumor-shape evidence.")

    (REPORTS / "Stage13A_Real_3D_Tumor_Graph_From_Mask_Report.md").write_text("\n".join(report))

    # Dashboard / README
    dash = REPORTS / "Dashboard.md"
    old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
    dash_add = (
        "\n\n## Stage 13A real 3D tumor graph\n\n"
        "- [Real 3D tumor graph report](Stage13A_Real_3D_Tumor_Graph_From_Mask_Report.md)\n"
        "- [Tumor graph nodes](table_views/stage13a_tumor_graph_nodes.md)\n"
        "- [Tumor graph edges](table_views/stage13a_tumor_graph_edges.md)\n"
        "- [Interactive tumor graph](interactive/604_stage13a_interactive_tumor_graph.html)\n"
    )
    if "Stage 13A real 3D tumor graph" not in old_dash:
        dash.write_text(old_dash.rstrip() + dash_add)

    readme = REPO / "README.md"
    old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
    readme_add = (
        "\n\n### Stage 13A: real 3D tumor graph from recovered mask\n\n"
        "Main outputs:\n\n"
        "- reports/Stage13A_Real_3D_Tumor_Graph_From_Mask_Report.md\n"
        "- reports/figures/600_stage13a_real_tumor_surface_graph.png\n"
        "- reports/figures/603_stage13a_rotating_tumor_graph.gif\n"
        "- reports/interactive/604_stage13a_interactive_tumor_graph.html\n"
    )
    if "Stage 13A: real 3D tumor graph from recovered mask" not in old_readme:
        readme.write_text(old_readme.rstrip() + readme_add)

    print("Stage 13A complete.")
    print("Selected patient:", best.get("patient_folder", ""))
    print("Mask mode:", best.get("mask_mode", ""))
    print("Tumor voxels:", best.get("voxels", ""))
    print("Graph nodes:", len(nodes))
    print("Graph edges:", len(edges))
    print("Interactive HTML created:", html_ok)
    print("Use this stage as the real tumor graph output.")


if __name__ == "__main__":
    main()
