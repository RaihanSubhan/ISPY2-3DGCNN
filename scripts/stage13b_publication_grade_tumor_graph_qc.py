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

STAGE13A_SELECTED = TABLES / "stage13a_selected_real_tumor_graph_case.csv"
PHASE7C = TABLES / "phase7c_fractional_tumor_mask_candidate_table.csv"

if not STAGE13A_SELECTED.exists() and not PHASE7C.exists():
    raise SystemExit("Missing Stage 13A selected case or Phase 7C candidate table.")


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


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


def pseudo_3d_from_2d(mask2d, n_slices=13):
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


def ensure_3d_mask(mask3d):
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
    pseudo = pseudo_3d_from_2d(mask3d[z], n_slices=13)
    pseudo = clean_mask3d(pseudo)

    return pseudo, "pseudo_3d_from_real_2d_mask_outline"


def load_selected_case():
    if STAGE13A_SELECTED.exists():
        df = pd.read_csv(STAGE13A_SELECTED, dtype=str).fillna("")
        if len(df) > 0:
            return df.iloc[0].to_dict()

    cand = pd.read_csv(PHASE7C, dtype=str).fillna("")
    cand = cand[cand["recovery_status"].eq("fractional_tumor_candidate_pass")].copy()
    if cand.empty:
        raise SystemExit("No Phase 7C pass cases found.")
    return cand.iloc[0].to_dict()


def reconstruct_mask_from_case(case):
    seg_file = case.get("seg_file", "")
    threshold = safe_float(case.get("selected_threshold", ""))

    if not Path(seg_file).exists():
        raise SystemExit(f"SEG file does not exist on Cradle: {seg_file}")

    _, arr = read_seg_array(seg_file)
    raw_mask = arr >= threshold
    mask3d, mode = ensure_3d_mask(raw_mask)

    return mask3d, mode


def quality_metrics(mask3d):
    if mask3d.sum() == 0:
        return {
            "voxels": 0,
            "z_extent": 0,
            "y_extent": 0,
            "x_extent": 0,
            "bbox_occupancy": 0,
            "shape_ratio": 999,
        }

    coords = np.argwhere(mask3d)
    z_extent = int(coords[:, 0].max() - coords[:, 0].min() + 1)
    y_extent = int(coords[:, 1].max() - coords[:, 1].min() + 1)
    x_extent = int(coords[:, 2].max() - coords[:, 2].min() + 1)

    bbox_vol = max(z_extent * y_extent * x_extent, 1)
    occupancy = float(mask3d.sum() / bbox_vol)
    ratio = max(z_extent, y_extent, x_extent) / max(min(z_extent, y_extent, x_extent), 1)

    return {
        "voxels": int(mask3d.sum()),
        "z_extent": z_extent,
        "y_extent": y_extent,
        "x_extent": x_extent,
        "bbox_occupancy": occupancy,
        "shape_ratio": ratio,
    }


def laplacian_smooth(vertices, faces, iterations=3, lam=0.30):
    if len(vertices) > 60000:
        return vertices

    neighbors = [[] for _ in range(len(vertices))]

    for tri in faces:
        a, b, c = tri
        neighbors[a].extend([b, c])
        neighbors[b].extend([a, c])
        neighbors[c].extend([a, b])

    vertices = vertices.copy()

    for _ in range(iterations):
        new_vertices = vertices.copy()
        for i, nb in enumerate(neighbors):
            if not nb:
                continue
            nb = list(set(nb))
            mean_nb = vertices[nb].mean(axis=0)
            new_vertices[i] = vertices[i] * (1 - lam) + mean_nb * lam
        vertices = new_vertices

    return vertices


def make_smooth_mesh(mask3d, sigma=1.0, max_faces=18000):
    padded = np.pad(mask3d.astype(float), 1, mode="constant", constant_values=0)
    smooth = ndi.gaussian_filter(padded, sigma=sigma)

    max_value = float(smooth.max())
    level = 0.35
    if max_value <= level:
        level = max_value * 0.5

    verts, faces, normals, values = measure.marching_cubes(smooth, level=level)

    # skimage returns z, y, x. Convert to x, y, z.
    raw_xyz = np.vstack([verts[:, 2], verts[:, 1], verts[:, 0]]).T

    raw_center = raw_xyz.mean(axis=0)
    raw_scale = max(np.ptp(raw_xyz[:, 0]), np.ptp(raw_xyz[:, 1]), np.ptp(raw_xyz[:, 2]), 1.0)

    def transform(raw):
        out = (raw - raw_center) / raw_scale
        out[:, 0] *= 1.10
        out[:, 1] *= 0.85
        out[:, 2] *= 0.75
        out += np.array([0.18, 0.02, 0.04])
        return out

    coords = transform(raw_xyz)

    coords = laplacian_smooth(coords, faces, iterations=3, lam=0.25)

    if faces.shape[0] > max_faces:
        step = int(np.ceil(faces.shape[0] / max_faces))
        faces = faces[::step]

    return coords, faces, transform


def mesh_area(coords, faces):
    tri = coords[faces]
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    cross = np.cross(a, b)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    return float(area.sum())


def create_graph(mask3d, transform, max_nodes=50, knn=5):
    voxel_zyx = np.argwhere(mask3d.astype(bool))
    if voxel_zyx.shape[0] < 10:
        raise RuntimeError("Too few voxels for graph construction.")

    raw_xyz = np.vstack([
        voxel_zyx[:, 2] + 1,
        voxel_zyx[:, 1] + 1,
        voxel_zyx[:, 0] + 1
    ]).T.astype(float)

    xyz = transform(raw_xyz)

    n_vox = xyz.shape[0]
    n_clusters = min(max_nodes, max(12, int(np.sqrt(n_vox))))
    n_clusters = min(n_clusters, n_vox)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_id = kmeans.fit_predict(xyz)
    centers = kmeans.cluster_centers_

    depth_map = ndi.distance_transform_edt(mask3d.astype(bool))
    voxel_depth = depth_map[voxel_zyx[:, 0], voxel_zyx[:, 1], voxel_zyx[:, 2]]

    tumor_center = centers.mean(axis=0)

    node_rows = []
    for cid in range(n_clusters):
        idx = np.where(cluster_id == cid)[0]
        pts = xyz[idx]
        centroid = pts.mean(axis=0)

        depth = float(np.mean(voxel_depth[idx]))
        radial = float(np.linalg.norm(centroid - tumor_center))

        node_rows.append({
            "node_id": int(cid),
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "centroid_z": float(centroid[2]),
            "voxel_count": int(len(idx)),
            "volume_fraction": float(len(idx) / n_vox),
            "mean_internal_depth": depth,
            "radial_distance": radial,
            "graph_region": "core" if depth > np.percentile(voxel_depth, 60) else "periphery"
        })

    nodes = pd.DataFrame(node_rows)

    nbrs = NearestNeighbors(n_neighbors=min(knn + 1, len(centers))).fit(centers)
    distances, indices = nbrs.kneighbors(centers)

    edge_set = set()
    edge_rows = []

    for i in range(len(centers)):
        for j_idx, d in zip(indices[i][1:], distances[i][1:]):
            pair = tuple(sorted([int(i), int(j_idx)]))
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

    # Ensure connected graph.
    edges = ensure_connected_graph(nodes, edges)

    degrees = compute_degrees(nodes, edges)
    nodes["degree"] = nodes["node_id"].map(degrees).fillna(0).astype(int)

    return nodes, edges


def compute_degrees(nodes, edges):
    deg = {int(n): 0 for n in nodes["node_id"].tolist()}
    for _, e in edges.iterrows():
        s = int(e["source"])
        t = int(e["target"])
        deg[s] = deg.get(s, 0) + 1
        deg[t] = deg.get(t, 0) + 1
    return deg


def connected_components(nodes, edges):
    node_ids = [int(x) for x in nodes["node_id"].tolist()]
    adj = {n: set() for n in node_ids}

    for _, e in edges.iterrows():
        s = int(e["source"])
        t = int(e["target"])
        adj[s].add(t)
        adj[t].add(s)

    seen = set()
    comps = []

    for n in node_ids:
        if n in seen:
            continue
        stack = [n]
        comp = []
        seen.add(n)

        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)

        comps.append(comp)

    return comps


def ensure_connected_graph(nodes, edges):
    edges = edges.copy()
    comps = connected_components(nodes, edges)

    if len(comps) <= 1:
        return edges

    pos = nodes.set_index("node_id")[["centroid_x", "centroid_y", "centroid_z"]].to_dict("index")

    while len(comps) > 1:
        c0 = comps[0]
        c1 = comps[1]

        best = None
        for a in c0:
            for b in c1:
                pa = np.array([pos[a]["centroid_x"], pos[a]["centroid_y"], pos[a]["centroid_z"]])
                pb = np.array([pos[b]["centroid_x"], pos[b]["centroid_y"], pos[b]["centroid_z"]])
                d = float(np.linalg.norm(pa - pb))
                if best is None or d < best[2]:
                    best = (a, b, d)

        edges = pd.concat([
            edges,
            pd.DataFrame([{
                "source": int(best[0]),
                "target": int(best[1]),
                "edge_distance": float(best[2]),
                "edge_type": "connectivity_fix"
            }])
        ], ignore_index=True)

        comps = connected_components(nodes, edges)

    return edges


def graph_qc(nodes, edges, mask_metrics, surface_area):
    deg = compute_degrees(nodes, edges)
    degree_values = np.array(list(deg.values()))

    comps = connected_components(nodes, edges)

    n = len(nodes)
    e = len(edges)
    density = float((2 * e) / max(n * (n - 1), 1))

    edge_dist = pd.to_numeric(edges["edge_distance"], errors="coerce")

    rows = [
        {"metric": "graph_nodes", "value": n},
        {"metric": "graph_edges", "value": e},
        {"metric": "connected_components", "value": len(comps)},
        {"metric": "mean_degree", "value": float(degree_values.mean())},
        {"metric": "min_degree", "value": int(degree_values.min())},
        {"metric": "max_degree", "value": int(degree_values.max())},
        {"metric": "graph_density", "value": density},
        {"metric": "mean_edge_distance", "value": float(edge_dist.mean())},
        {"metric": "max_edge_distance", "value": float(edge_dist.max())},
        {"metric": "surface_area_proxy", "value": surface_area},
    ]

    for k, v in mask_metrics.items():
        rows.append({"metric": "mask_" + k, "value": v})

    return pd.DataFrame(rows)


def plot_breast_shell(ax, alpha=0.06):
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)

    x = 1.25 * np.outer(np.cos(u), np.sin(v))
    y = 0.90 * np.outer(np.sin(u), np.sin(v))
    z = 0.75 * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, color="lightgray", alpha=alpha, linewidth=0, shade=True)


def add_mesh(ax, coords, faces, color="red", alpha=0.50):
    mesh = Poly3DCollection(coords[faces], alpha=alpha)
    mesh.set_facecolor(color)
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)


def add_graph(ax, nodes, edges, alpha_edges=0.65):
    pos = nodes.set_index("node_id")[["centroid_x", "centroid_y", "centroid_z"]].to_dict("index")

    for _, e in edges.iterrows():
        s = int(e["source"])
        t = int(e["target"])

        if s not in pos or t not in pos:
            continue

        xs = [pos[s]["centroid_x"], pos[t]["centroid_x"]]
        ys = [pos[s]["centroid_y"], pos[t]["centroid_y"]]
        zs = [pos[s]["centroid_z"], pos[t]["centroid_z"]]

        color = "black" if e.get("edge_type", "") == "spatial_knn" else "darkred"
        ax.plot(xs, ys, zs, color=color, linewidth=1.2, alpha=alpha_edges)

    colors = ["gold" if r == "core" else "deepskyblue" for r in nodes["graph_region"]]
    sizes = 35 + 260 * pd.to_numeric(nodes["volume_fraction"], errors="coerce").fillna(0).values

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


def save_publication_panel(coords, faces, nodes, edges, qc, out_path):
    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    plot_breast_shell(ax1, alpha=0.06)
    add_mesh(ax1, coords, faces, color="red", alpha=0.55)
    setup_ax(ax1, "A. Smoothed real-mask-derived tumor surface", azim=35)

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    plot_breast_shell(ax2, alpha=0.06)
    add_mesh(ax2, coords, faces, color="red", alpha=0.25)
    add_graph(ax2, nodes, edges)
    setup_ax(ax2, "B. Tumor graph over surface", azim=35)

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    add_mesh(ax3, coords, faces, color="red", alpha=0.20)
    add_graph(ax3, nodes, edges)
    setup_ax(ax3, "C. Tumor graph zoom", azim=55)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")

    qmap = dict(zip(qc["metric"], qc["value"]))
    txt = (
        "D. Graph QC\n\n"
        f"Nodes: {qmap.get('graph_nodes', '')}\n"
        f"Edges: {qmap.get('graph_edges', '')}\n"
        f"Connected components: {qmap.get('connected_components', '')}\n"
        f"Mean degree: {float(qmap.get('mean_degree', 0)):.2f}\n"
        f"Mean edge distance: {float(qmap.get('mean_edge_distance', 0)):.3f}\n"
        f"Tumor voxels: {qmap.get('mask_voxels', '')}\n"
        f"Mask z/y/x extent: {qmap.get('mask_z_extent', '')}/"
        f"{qmap.get('mask_y_extent', '')}/"
        f"{qmap.get('mask_x_extent', '')}\n\n"
        "Gold nodes = core\n"
        "Blue nodes = periphery\n\n"
        "Edges are spatial graph links,\nnot measured vessels."
    )
    ax4.text(0.05, 0.95, txt, va="top", ha="left", fontsize=12)

    fig.suptitle("Stage 13B publication-grade real 3D tumor graph", fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_degree_plot(nodes, edges, out_path):
    plt.figure(figsize=(8, 5))
    degree = pd.to_numeric(nodes["degree"], errors="coerce")
    degree.plot(kind="hist", bins=range(int(degree.min()), int(degree.max()) + 2))
    plt.xlabel("Node degree")
    plt.ylabel("Number of nodes")
    plt.title("Stage 13B Tumor Graph Degree Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_edge_plot(edges, out_path):
    plt.figure(figsize=(8, 5))
    pd.to_numeric(edges["edge_distance"], errors="coerce").plot(kind="hist", bins=20)
    plt.xlabel("3D edge distance")
    plt.ylabel("Number of edges")
    plt.title("Stage 13B Tumor Graph Edge-Length Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_node_feature_plot(nodes, out_path):
    plt.figure(figsize=(8, 6))
    colors = ["gold" if r == "core" else "deepskyblue" for r in nodes["graph_region"]]

    plt.scatter(
        nodes["radial_distance"],
        nodes["mean_internal_depth"],
        s=80 + 900 * nodes["volume_fraction"],
        c=colors,
        edgecolor="black",
        alpha=0.85
    )

    for _, row in nodes.iterrows():
        plt.text(row["radial_distance"], row["mean_internal_depth"], str(int(row["node_id"])), fontsize=7)

    plt.xlabel("Radial distance from tumor graph center")
    plt.ylabel("Mean internal depth")
    plt.title("Stage 13B Tumor Graph Node Feature Map")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_rotating_gif(coords, faces, nodes, edges, out_gif, out_png, frames=45):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(i):
        ax.clear()
        azim = 35 + 360 * i / frames
        plot_breast_shell(ax, alpha=0.06)
        add_mesh(ax, coords, faces, color="red", alpha=0.28)
        add_graph(ax, nodes, edges)
        setup_ax(ax, "Rotating publication-grade 3D tumor graph", azim=azim)

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
        out_html.write_text("<html><body><p>Plotly not available.</p></body></html>")
        return False

    mesh = go.Mesh3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color="red",
        opacity=0.25,
        name="Smoothed tumor surface"
    )

    pos = nodes.set_index("node_id")[["centroid_x", "centroid_y", "centroid_z"]].to_dict("index")

    ex, ey, ez = [], [], []
    for _, e in edges.iterrows():
        s = int(e["source"])
        t = int(e["target"])
        if s not in pos or t not in pos:
            continue
        ex += [pos[s]["centroid_x"], pos[t]["centroid_x"], None]
        ey += [pos[s]["centroid_y"], pos[t]["centroid_y"], None]
        ez += [pos[s]["centroid_z"], pos[t]["centroid_z"], None]

    edge_trace = go.Scatter3d(
        x=ex,
        y=ey,
        z=ez,
        mode="lines",
        line=dict(color="black", width=3),
        name="Spatial graph edges"
    )

    node_colors = [1 if r == "core" else 0 for r in nodes["graph_region"]]

    node_trace = go.Scatter3d(
        x=nodes["centroid_x"],
        y=nodes["centroid_y"],
        z=nodes["centroid_z"],
        mode="markers+text",
        text=nodes["node_id"].astype(str),
        marker=dict(
            size=6 + 28 * nodes["volume_fraction"],
            color=node_colors,
            colorscale="Viridis",
            line=dict(color="black", width=1)
        ),
        name="Tumor graph nodes"
    )

    fig = go.Figure(data=[mesh, edge_trace, node_trace])
    fig.update_layout(
        title="Interactive publication-grade real 3D tumor graph",
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
    parser.add_argument("--nodes", type=int, default=50)
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--frames", type=int, default=45)
    parser.add_argument("--sigma", type=float, default=1.0)
    args = parser.parse_args()

    case = load_selected_case()
    mask3d, mask_mode = reconstruct_mask_from_case(case)

    mask_metrics = quality_metrics(mask3d)

    coords, faces, transform = make_smooth_mesh(mask3d, sigma=args.sigma)
    surface_area = mesh_area(coords, faces)

    nodes, edges = create_graph(mask3d, transform, max_nodes=args.nodes, knn=args.knn)

    qc = graph_qc(nodes, edges, mask_metrics, surface_area)

    patient = case.get("patient_folder", "")
    seg_uid = case.get("seg_series_uid", "")

    nodes["patient_folder"] = patient
    nodes["seg_series_uid"] = seg_uid
    nodes["mask_mode"] = mask_mode

    edges["patient_folder"] = patient
    edges["seg_series_uid"] = seg_uid
    edges["mask_mode"] = mask_mode

    nodes.to_csv(TABLES / "stage13b_publication_tumor_graph_nodes.csv", index=False)
    edges.to_csv(TABLES / "stage13b_publication_tumor_graph_edges.csv", index=False)
    qc.to_csv(TABLES / "stage13b_graph_qc_metrics.csv", index=False)

    # Tensor-ready CSVs.
    tensor_features = nodes[[
        "node_id",
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "voxel_count",
        "volume_fraction",
        "mean_internal_depth",
        "radial_distance",
        "degree",
        "graph_region"
    ]].copy()
    tensor_features["graph_region_core"] = (tensor_features["graph_region"] == "core").astype(int)
    tensor_features.to_csv(TABLES / "stage13b_graph_tensor_features.csv", index=False)

    edge_index = edges[["source", "target", "edge_distance", "edge_type"]].copy()
    edge_index.to_csv(TABLES / "stage13b_graph_tensor_edge_index.csv", index=False)

    selected = pd.DataFrame([{
        "patient_folder": patient,
        "seg_series_uid": seg_uid,
        "mask_mode": mask_mode,
        "selected_threshold": case.get("selected_threshold", ""),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "surface_area_proxy": surface_area,
        "mesh_vertices": len(coords),
        "mesh_faces": len(faces),
        "publication_panel": "reports/figures/610_stage13b_publication_tumor_graph_panel.png",
        "interactive_html": "reports/interactive/616_stage13b_publication_tumor_graph.html"
    }])
    selected.to_csv(TABLES / "stage13b_selected_publication_graph_case.csv", index=False)

    claims = pd.DataFrame([
        {
            "component": "tumor surface",
            "safe_claim": "Smoothed surface is derived from a recovered tumor mask.",
            "limit": "It is not a manual expert segmentation unless clinically verified."
        },
        {
            "component": "tumor graph nodes",
            "safe_claim": "Nodes are spatial clusters of tumor voxels.",
            "limit": "Nodes are computational regions, not histologic compartments."
        },
        {
            "component": "graph edges",
            "safe_claim": "Edges connect nearby tumor regions in 3D space.",
            "limit": "Edges are not vessels and not measured fluid pathways."
        },
        {
            "component": "publication figure",
            "safe_claim": "Figure is suitable for methods/future-work explanation.",
            "limit": "Do not claim clinical validation from this figure alone."
        }
    ])
    claims.to_csv(TABLES / "stage13b_publication_graph_claims.csv", index=False)

    save_publication_panel(
        coords,
        faces,
        nodes,
        edges,
        qc,
        FIGS / "610_stage13b_publication_tumor_graph_panel.png"
    )

    save_degree_plot(
        nodes,
        edges,
        FIGS / "611_stage13b_node_degree_distribution.png"
    )

    save_edge_plot(
        edges,
        FIGS / "612_stage13b_edge_length_distribution.png"
    )

    save_node_feature_plot(
        nodes,
        FIGS / "613_stage13b_node_feature_map.png"
    )

    save_rotating_gif(
        coords,
        faces,
        nodes,
        edges,
        FIGS / "614_stage13b_rotating_publication_tumor_graph.gif",
        FIGS / "614_stage13b_rotating_publication_tumor_graph_static.png",
        frames=args.frames
    )

    html_ok = save_plotly_html(
        coords,
        faces,
        nodes,
        edges,
        INTERACTIVE / "616_stage13b_publication_tumor_graph.html"
    )

    # Schema.
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    text = (
        "Stage 13B publication-grade tumor graph\n\n"
        "Recovered tumor mask\n"
        "↓\n"
        "3D mask cleaning and smoothing\n"
        "↓\n"
        "Marching-cubes tumor surface\n"
        "↓\n"
        "Voxel clustering into tumor graph nodes\n"
        "↓\n"
        "Spatial k-nearest-neighbor graph edges\n\n"
        "Graph edges are spatial links, not vessels.\n"
        "Use this figure for 3DGCNN methods/future work."
    )
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGS / "615_stage13b_publication_graph_schema.png", dpi=220)
    plt.close()

    make_md(TABLES / "stage13b_selected_publication_graph_case.csv", VIEWS / "stage13b_selected_publication_graph_case.md", "Stage 13B Selected Publication Graph Case")
    make_md(TABLES / "stage13b_publication_tumor_graph_nodes.csv", VIEWS / "stage13b_publication_tumor_graph_nodes.md", "Stage 13B Tumor Graph Nodes")
    make_md(TABLES / "stage13b_publication_tumor_graph_edges.csv", VIEWS / "stage13b_publication_tumor_graph_edges.md", "Stage 13B Tumor Graph Edges")
    make_md(TABLES / "stage13b_graph_qc_metrics.csv", VIEWS / "stage13b_graph_qc_metrics.md", "Stage 13B Graph QC Metrics")
    make_md(TABLES / "stage13b_graph_tensor_features.csv", VIEWS / "stage13b_graph_tensor_features.md", "Stage 13B Graph Tensor Features")
    make_md(TABLES / "stage13b_graph_tensor_edge_index.csv", VIEWS / "stage13b_graph_tensor_edge_index.md", "Stage 13B Graph Tensor Edge Index")
    make_md(TABLES / "stage13b_publication_graph_claims.csv", VIEWS / "stage13b_publication_graph_claims.md", "Stage 13B Publication Graph Claims")

    qmap = dict(zip(qc["metric"], qc["value"]))

    report = []
    report.append("# Stage 13B Publication-Grade 3D Tumor Graph QC Report")
    report.append("")
    report.append("This stage refines the Stage 13A real 3D tumor graph into a cleaner publication-grade graph and adds graph quality control.")
    report.append("")
    report.append("## What was improved")
    report.append("")
    report.append("- Smoothed the real-mask-derived tumor surface.")
    report.append("- Rebuilt graph nodes with spatial clustering.")
    report.append("- Ensured graph connectivity.")
    report.append("- Added graph QC metrics, degree distribution, and edge-length distribution.")
    report.append("- Added tensor-ready node and edge tables for future 3DGCNN experiments.")
    report.append("")
    report.append("## Main outputs")
    report.append("")
    report.append("- reports/figures/610_stage13b_publication_tumor_graph_panel.png")
    report.append("- reports/figures/611_stage13b_node_degree_distribution.png")
    report.append("- reports/figures/612_stage13b_edge_length_distribution.png")
    report.append("- reports/figures/613_stage13b_node_feature_map.png")
    report.append("- reports/figures/614_stage13b_rotating_publication_tumor_graph.gif")
    report.append("- reports/figures/615_stage13b_publication_graph_schema.png")
    report.append("- reports/interactive/616_stage13b_publication_tumor_graph.html")
    report.append("- reports/tables/stage13b_publication_tumor_graph_nodes.csv")
    report.append("- reports/tables/stage13b_publication_tumor_graph_edges.csv")
    report.append("- reports/tables/stage13b_graph_qc_metrics.csv")
    report.append("- reports/tables/stage13b_graph_tensor_features.csv")
    report.append("- reports/tables/stage13b_graph_tensor_edge_index.csv")
    report.append("")
    report.append("## Selected graph")
    report.append("")
    report.append("- Patient: " + str(patient))
    report.append("- Mask mode: " + str(mask_mode))
    report.append("- Graph nodes: " + str(len(nodes)))
    report.append("- Graph edges: " + str(len(edges)))
    report.append("- Connected components: " + str(qmap.get("connected_components", "")))
    report.append("- Mean degree: " + str(qmap.get("mean_degree", "")))
    report.append("- Mean edge distance: " + str(qmap.get("mean_edge_distance", "")))
    report.append("- Interactive HTML created: " + str(html_ok))
    report.append("")
    report.append("## Interpretation")
    report.append("")
    report.append("This is the cleanest current 3D tumor graph output. It should replace earlier ellipsoid or schematic tumor-shape figures when discussing the 3DGCNN direction.")
    report.append("")
    report.append("## Limitation")
    report.append("")
    report.append("The tumor graph is derived from a recovered fractional mask, not a manual expert segmentation. Graph edges are spatial links, not vessels or measured flow.")
    report.append("")
    report.append("## Next step")
    report.append("")
    report.append("Use Stage 13B as the final 3D tumor graph visualization. If more work is needed, the next scientific step is to train a graph model on multiple patient tumor graphs, not just one visual case.")

    (REPORTS / "Stage13B_Publication_Grade_3D_Tumor_Graph_QC_Report.md").write_text("\n".join(report))

    # Dashboard / README
    dash = REPORTS / "Dashboard.md"
    old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
    dash_add = (
        "\n\n## Stage 13B publication-grade 3D tumor graph QC\n\n"
        "- [Publication-grade 3D tumor graph QC report](Stage13B_Publication_Grade_3D_Tumor_Graph_QC_Report.md)\n"
        "- [Graph QC metrics](table_views/stage13b_graph_qc_metrics.md)\n"
        "- [Interactive publication tumor graph](interactive/616_stage13b_publication_tumor_graph.html)\n"
    )
    if "Stage 13B publication-grade 3D tumor graph QC" not in old_dash:
        dash.write_text(old_dash.rstrip() + dash_add)

    readme = REPO / "README.md"
    old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
    readme_add = (
        "\n\n### Stage 13B: publication-grade 3D tumor graph QC\n\n"
        "Main outputs:\n\n"
        "- reports/Stage13B_Publication_Grade_3D_Tumor_Graph_QC_Report.md\n"
        "- reports/figures/610_stage13b_publication_tumor_graph_panel.png\n"
        "- reports/interactive/616_stage13b_publication_tumor_graph.html\n"
    )
    if "Stage 13B: publication-grade 3D tumor graph QC" not in old_readme:
        readme.write_text(old_readme.rstrip() + readme_add)

    print("Stage 13B complete.")
    print("Patient:", patient)
    print("Mask mode:", mask_mode)
    print("Graph nodes:", len(nodes))
    print("Graph edges:", len(edges))
    print("Connected components:", qmap.get("connected_components", ""))
    print("Interactive HTML created:", html_ok)
    print("Use Stage 13B as the clean final 3D tumor graph figure.")


if __name__ == "__main__":
    main()
