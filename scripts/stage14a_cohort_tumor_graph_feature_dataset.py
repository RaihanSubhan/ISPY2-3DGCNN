"""
Stage 14A: Cohort-level 3D tumor-graph feature dataset.

Expands the Stage 13 single-case 3D tumor graph (ISPY2-109623) to the whole
recovered-mask cohort: builds a 3D tumor graph per patient, extracts graph-level
features, merges pCR labels, and reports how many LABELED graph cases exist.
That count is the gate for the temporal 3DGCNN idea.

Graph edges are spatial kNN links, not vessels or measured flow. Masks are
recovered fractional masks, not expert segmentations. Pilot feasibility only.

Graph-building functions are copied from stage13a so the method is identical.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pydicom
from scipy import ndimage as ndi
from skimage import measure, morphology

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

import networkx as nx


REPO = Path.home() / "ISPY2-3DGCNN"
REPORTS = REPO / "reports"
TABLES = REPORTS / "tables"
FIGS = REPORTS / "figures"
VIEWS = REPORTS / "table_views"

for p in [REPORTS, TABLES, FIGS, VIEWS]:
    p.mkdir(parents=True, exist_ok=True)

PHASE7C = TABLES / "phase7c_fractional_tumor_mask_candidate_table.csv"
DEFAULT_LABELS = TABLES / "phase4c_pcr_label_template.csv"


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
    if n <= 1:
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
    if mask.sum() > 0:
        mask = morphology.binary_closing(mask, morphology.ball(1))
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
        return {"voxels": 0, "z_extent": 0, "y_extent": 0, "x_extent": 0,
                "bbox_occupancy": 0, "shape_ratio": 999, "quality_score": -999}
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
    return {"voxels": int(mask3d.sum()), "z_extent": z_extent, "y_extent": y_extent,
            "x_extent": x_extent, "bbox_occupancy": occupancy, "shape_ratio": ratio,
            "quality_score": float(score)}


def mesh_transform(mask3d):
    padded = np.pad(mask3d.astype(bool), 1, mode="constant", constant_values=False)
    verts, _, _, _ = measure.marching_cubes(padded.astype(float), level=0.5)
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
    return transform


def create_tumor_graph(mask3d, transform, max_nodes=50, knn=5):
    voxel_zyx = np.argwhere(mask3d.astype(bool))
    if voxel_zyx.shape[0] < 8:
        raise RuntimeError("Too few tumor voxels to create graph.")
    raw_xyz = np.vstack([voxel_zyx[:, 2] + 1, voxel_zyx[:, 1] + 1, voxel_zyx[:, 0] + 1]).T.astype(float)
    xyz = transform(raw_xyz)
    n_vox = xyz.shape[0]
    n_clusters = min(max_nodes, max(8, int(np.sqrt(n_vox))))
    n_clusters = min(n_clusters, n_vox)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_id = kmeans.fit_predict(xyz)
    centers = kmeans.cluster_centers_
    depth_map = ndi.distance_transform_edt(mask3d.astype(bool))
    voxel_depth = depth_map[voxel_zyx[:, 0], voxel_zyx[:, 1], voxel_zyx[:, 2]]
    tumor_center = centers.mean(axis=0)
    depth_p60 = np.percentile(voxel_depth, 60)
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
            "graph_region": "core" if np.mean(voxel_depth[idx]) > depth_p60 else "periphery",
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
            edge_rows.append({"source": pair[0], "target": pair[1],
                              "edge_distance": float(d), "edge_type": "spatial_knn"})
    edges = pd.DataFrame(edge_rows)
    return nodes, edges


def graph_level_features(nodes, edges):
    n_nodes = len(nodes)
    n_edges = len(edges)
    g = nx.Graph()
    g.add_nodes_from(nodes["node_id"].tolist())
    if n_edges:
        g.add_edges_from(edges[["source", "target"]].itertuples(index=False, name=None))
    if g.number_of_nodes():
        degrees = np.array([d for _, d in g.degree()], dtype=float)
        n_components = nx.number_connected_components(g)
        largest_cc = max((len(c) for c in nx.connected_components(g)), default=0)
        density = nx.density(g) if g.number_of_nodes() > 1 else 0.0
    else:
        degrees = np.array([0.0])
        n_components = 0
        largest_cc = 0
        density = 0.0
    vf = nodes["volume_fraction"].to_numpy() if n_nodes else np.array([1.0])
    vf = vf[vf > 0]
    vf_entropy = float(-np.sum(vf * np.log(vf))) if vf.size else 0.0
    edist = edges["edge_distance"].to_numpy() if n_edges else np.array([0.0])
    return {
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "mean_degree": float(degrees.mean()),
        "max_degree": float(degrees.max()),
        "graph_density": float(density),
        "n_connected_components": int(n_components),
        "largest_cc_fraction": float(largest_cc / n_nodes) if n_nodes else 0.0,
        "mean_edge_distance": float(edist.mean()),
        "std_edge_distance": float(edist.std()),
        "mean_internal_depth": float(nodes["mean_internal_depth"].mean()) if n_nodes else 0.0,
        "max_internal_depth": float(nodes["mean_internal_depth"].max()) if n_nodes else 0.0,
        "mean_radial_distance": float(nodes["radial_distance"].mean()) if n_nodes else 0.0,
        "std_radial_distance": float(nodes["radial_distance"].std()) if n_nodes else 0.0,
        "core_node_fraction": float((nodes["graph_region"] == "core").mean()) if n_nodes else 0.0,
        "node_size_entropy": vf_entropy,
    }


def normalize_pcr(series):
    pos = {"1", "1.0", "pcr", "yes", "y", "true", "responder", "complete"}
    neg = {"0", "0.0", "non-pcr", "no", "n", "false", "non_responder", "incomplete", "npcr"}
    out = []
    for v in series:
        s = str(v).strip().lower()
        if s in pos:
            out.append(1.0)
        elif s in neg:
            out.append(0.0)
        else:
            fv = safe_float(v)
            out.append(1.0 if fv == 1 else 0.0 if fv == 0 else np.nan)
    return out


def build_cohort(label_table, max_patients=0, max_nodes=50, knn=5):
    if not PHASE7C.exists():
        raise SystemExit("Missing reports/tables/phase7c_fractional_tumor_mask_candidate_table.csv. Run Phase 7C first.")
    cand = pd.read_csv(PHASE7C, dtype=str).fillna("")
    cand = cand[cand["recovery_status"].eq("fractional_tumor_candidate_pass")].copy()
    if cand.empty:
        raise SystemExit("No fractional_tumor_candidate_pass cases in Phase 7C.")
    patients = sorted(cand["patient_folder"].unique())
    if max_patients and max_patients > 0:
        patients = patients[:max_patients]
    print(f"[info] recovered-mask patients to process: {len(patients)}")
    all_nodes, all_edges, feat_rows = [], [], []
    for pi, pid in enumerate(patients, 1):
        sub = cand[cand["patient_folder"].eq(pid)]
        best = None
        for _, r in sub.iterrows():
            seg_file = r.get("seg_file", "")
            thr = safe_float(r.get("selected_threshold", ""))
            try:
                if not seg_file or not Path(seg_file).exists():
                    continue
                _, arr = read_seg_array(seg_file)
                mask3d, mode = ensure_usable_3d(arr >= thr)
                if mask3d.sum() == 0:
                    continue
                q = quality_metrics(mask3d)["quality_score"]
                if best is None or q > best[0]:
                    best = (q, mask3d, mode, r.get("seg_series_uid", ""))
            except Exception as e:
                print(f"[warn] {pid}: mask read failed ({str(e)[:80]})")
                continue
        if best is None:
            feat_rows.append({"patient_folder": pid, "status": "no_usable_mask"})
            print(f"[{pi}/{len(patients)}] {pid}: no usable mask")
            continue
        q_score, mask3d, mode, seg_uid = best
        try:
            transform = mesh_transform(mask3d)
            nodes, edges = create_tumor_graph(mask3d, transform, max_nodes=max_nodes, knn=knn)
        except Exception as e:
            feat_rows.append({"patient_folder": pid, "status": f"graph_failed:{str(e)[:60]}"})
            print(f"[{pi}/{len(patients)}] {pid}: graph build failed")
            continue
        nodes.insert(0, "patient_folder", pid)
        edges.insert(0, "patient_folder", pid)
        all_nodes.append(nodes)
        all_edges.append(edges)
        feats = {"patient_folder": pid, "status": "success", "mask_mode": mode,
                 "seg_series_uid": seg_uid, "mask_quality_score": float(q_score),
                 "tumor_voxels": int(mask3d.sum())}
        feats.update(graph_level_features(nodes, edges))
        feat_rows.append(feats)
        print(f"[{pi}/{len(patients)}] {pid}: nodes={feats['n_nodes']} edges={feats['n_edges']} "
              f"mode={mode} voxels={feats['tumor_voxels']}")
    nodes_df = pd.concat(all_nodes, ignore_index=True) if all_nodes else pd.DataFrame()
    edges_df = pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame()
    feats_df = pd.DataFrame(feat_rows)
    nodes_df.to_csv(TABLES / "stage14a_cohort_tumor_graph_nodes.csv", index=False)
    edges_df.to_csv(TABLES / "stage14a_cohort_tumor_graph_edges.csv", index=False)
    feats_df.to_csv(TABLES / "stage14a_cohort_graph_features.csv", index=False)
    ok = feats_df[feats_df["status"].eq("success")].copy()
    label_path = Path(label_table)
    if label_path.exists():
        lab = pd.read_csv(label_path, dtype=str).fillna("")
        lab_col = "pCR_label" if "pCR_label" in lab.columns else lab.columns[1]
        keep = ["patient_folder", lab_col]
        for extra in ["response_label", "tumor_subtype", "treatment_arm"]:
            if extra in lab.columns:
                keep.append(extra)
        lab = lab[keep].drop_duplicates("patient_folder")
        merged = ok.merge(lab, on="patient_folder", how="left")
        merged["pCR_label_numeric"] = normalize_pcr(merged[lab_col])
    else:
        print(f"[warn] label table not found: {label_path}")
        merged = ok.copy()
        merged["pCR_label_numeric"] = np.nan
    merged["has_label"] = merged["pCR_label_numeric"].notna()
    merged.to_csv(TABLES / "stage14a_cohort_label_merge.csv", index=False)
    return feats_df, merged


def write_summary(feats_df, merged, max_nodes, knn):
    n_total = int((feats_df["status"] == "success").sum()) if "status" in feats_df else 0
    n_real = int((feats_df.get("mask_mode") == "real_recovered_3d_mask").sum()) if "mask_mode" in feats_df else 0
    n_pseudo = int((feats_df.get("mask_mode") == "pseudo_3d_from_real_2d_mask_outline").sum()) if "mask_mode" in feats_df else 0
    n_labeled = int(merged["has_label"].sum()) if "has_label" in merged else 0
    n_pos = int((merged["pCR_label_numeric"] == 1).sum()) if "pCR_label_numeric" in merged else 0
    n_neg = int((merged["pCR_label_numeric"] == 0).sum()) if "pCR_label_numeric" in merged else 0
    if n_labeled >= 40 and min(n_pos, n_neg) >= 12:
        verdict = "ENOUGH_for_pilot_graph_feature_modeling"
        verdict_text = ("Enough labeled tumor graphs to attempt classical graph-feature modelling as a "
                        "pilot experiment. A graph-feature SVM/RandomForest is now defensible.")
    elif n_labeled >= 20 and min(n_pos, n_neg) >= 6:
        verdict = "BORDERLINE_exploratory_only"
        verdict_text = ("Barely enough labeled tumor graphs for a strictly exploratory graph-feature "
                        "baseline with heavy cross-validation. Keep the FTV SVM as the main result.")
    else:
        verdict = "NOT_ENOUGH_keep_3dgcnn_as_future_work"
        verdict_text = ("NOT enough labeled tumor graphs for graph-feature modelling or a 3DGCNN. This is "
                        "the empirical justification for keeping the temporal 3DGCNN as future work. The "
                        "cohort tumor-graph dataset is a feasibility asset, ready to grow with more labels.")
    md = f"""# Stage 14A Cohort Tumor-Graph Feature Dataset Report

Expands the Stage 13 single-case 3D tumor graph (ISPY2-109623) into a cohort-level
tumor-graph feature dataset from the recovered fractional tumor masks. Each patient's
best-quality recovered mask becomes a spatial tumor-node graph with the same method as
Stage 13 (marching-cubes surface, KMeans nodes, spatial kNN edges).

## Graph settings
- Max nodes per graph: {max_nodes}
- kNN per node: {knn}

## Cohort counts
- Patients with a usable tumor graph: {n_total}
- Built from a real recovered 3D mask: {n_real}
- Built from a pseudo-3D extrusion of a real 2D outline: {n_pseudo}

## Labeled cohort (the decision gate)
- Patients with a tumor graph AND a pCR label: {n_labeled}
- pCR-positive graphs: {n_pos}
- non-pCR graphs: {n_neg}

## Decision
- Verdict code: **{verdict}**
- {verdict_text}

## Main outputs
- reports/tables/stage14a_cohort_graph_features.csv
- reports/tables/stage14a_cohort_label_merge.csv
- reports/tables/stage14a_cohort_tumor_graph_nodes.csv
- reports/tables/stage14a_cohort_tumor_graph_edges.csv
- reports/figures/700_stage14a_cohort_graph_overview.png

## Safe wording
Use: cohort tumor-graph feature dataset, spatial tumor-node graph, recovered-mask-derived graph.
Avoid: measured blood flow, vessel graph, clinically validated 3D tumor segmentation.

## Limitation
Tumor graphs come from recovered fractional masks, not expert segmentations. Graph edges are
spatial links, not vessels or measured flow. Pilot feasibility work.
"""
    (REPORTS / "Stage14A_Cohort_Tumor_Graph_Feature_Dataset_Report.md").write_text(md)
    if "has_label" in merged and len(merged):
        view_cols = [c for c in ["patient_folder", "n_nodes", "n_edges", "mean_degree",
                                  "mean_edge_distance", "tumor_voxels", "mask_mode",
                                  "pCR_label_numeric", "has_label"] if c in merged.columns]
        try:
            sub = merged[view_cols]
            lines = ["| " + " | ".join(view_cols) + " |",
                     "| " + " | ".join(["---"] * len(view_cols)) + " |"]
            for _, r in sub.iterrows():
                lines.append("| " + " | ".join(str(r[c]) for c in view_cols) + " |")
            (VIEWS / "stage14a_cohort_label_merge.md").write_text("\n".join(lines))
        except Exception as e:
            print(f"[warn] table view skipped: {str(e)[:80]}")
    print("\n" + "=" * 60)
    print(f"COHORT TUMOR GRAPHS BUILT : {n_total}")
    print(f"LABELED GRAPH CASES       : {n_labeled}  (pCR+={n_pos}, pCR-={n_neg})")
    print(f"VERDICT                   : {verdict}")
    print("=" * 60)
    return verdict


def make_overview_figure(feats_df, merged):
    ok = feats_df[feats_df.get("status").eq("success")].copy() if "status" in feats_df else feats_df.copy()
    if ok.empty:
        return
    lab_map = {}
    if "patient_folder" in merged and "pCR_label_numeric" in merged:
        lab_map = dict(zip(merged["patient_folder"], merged["pCR_label_numeric"]))
    ok = ok.sort_values("n_nodes", ascending=False).reset_index(drop=True)
    colors = []
    for pid in ok["patient_folder"]:
        v = lab_map.get(pid, np.nan)
        colors.append("#2ca02c" if v == 1 else "#d62728" if v == 0 else "#9e9e9e")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(ok)), ok["n_edges"], color=colors)
    ax.set_xticks(range(len(ok)))
    ax.set_xticklabels(ok["patient_folder"], rotation=90, fontsize=6)
    ax.set_ylabel("graph edges")
    ax.set_title("Stage 14A cohort tumor graphs  (green=pCR+, red=pCR-, grey=unlabeled)")
    fig.tight_layout()
    fig.savefig(FIGS / "700_stage14a_cohort_graph_overview.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-table", default=str(DEFAULT_LABELS))
    ap.add_argument("--max-patients", type=int, default=0)
    ap.add_argument("--max-nodes", type=int, default=50)
    ap.add_argument("--knn", type=int, default=5)
    args = ap.parse_args()
    feats_df, merged = build_cohort(args.label_table, args.max_patients, args.max_nodes, args.knn)
    make_overview_figure(feats_df, merged)
    write_summary(feats_df, merged, args.max_nodes, args.knn)


if __name__ == "__main__":
    main()
