from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path.home() / "ISPY2-3DGCNN"
REPORTS = REPO / "reports"
TABLES = REPORTS / "tables"
FIGS = REPORTS / "figures"
VIEWS = REPORTS / "table_views"

for p in [REPORTS, TABLES, FIGS, VIEWS]:
    p.mkdir(parents=True, exist_ok=True)

DATASET = TABLES / "phase8c_ftv_modeling_dataset.csv"

if not DATASET.exists():
    raise SystemExit("Missing reports/tables/phase8c_ftv_modeling_dataset.csv. Run Phase 8C first.")

df = pd.read_csv(DATASET, dtype=str).fillna("")

if "patient_folder" not in df.columns or "label" not in df.columns:
    raise SystemExit("phase8c_ftv_modeling_dataset.csv must contain patient_folder and label.")

visit_map = [
    {
        "visit_label": "T0",
        "visit_index": 0,
        "volume_col": "VOLUME_TUM_BLU_V10",
        "sphericity_col": "SPHERICITY_T0",
        "ld_col": "LD_T0",
        "bpe_col": "BPE_5slice_mean_T0",
    },
    {
        "visit_label": "T1",
        "visit_index": 1,
        "volume_col": "VOLUME_TUM_BLU_V20",
        "sphericity_col": "SPHERICITY_T1",
        "ld_col": "LD_T1",
        "bpe_col": "BPE_5slice_mean_T1",
    },
    {
        "visit_label": "T2",
        "visit_index": 2,
        "volume_col": "VOLUME_TUM_BLU_V30",
        "sphericity_col": "SPHERICITY_T2",
        "ld_col": "LD_T2",
        "bpe_col": "BPE_5slice_mean_T2",
    },
    {
        "visit_label": "T3",
        "visit_index": 3,
        "volume_col": "VOLUME_TUM_BLU_V40",
        "sphericity_col": "SPHERICITY_T3",
        "ld_col": "LD_T3",
        "bpe_col": "BPE_5slice_mean_T3",
    },
]

def num(row, col):
    if col not in row.index:
        return np.nan
    return pd.to_numeric(row[col], errors="coerce")

node_rows = []

for _, row in df.iterrows():
    patient = str(row["patient_folder"])
    label = pd.to_numeric(row["label"], errors="coerce")

    for v in visit_map:
        volume = num(row, v["volume_col"])
        sphericity = num(row, v["sphericity_col"])
        ld = num(row, v["ld_col"])
        bpe = num(row, v["bpe_col"])

        values = [volume, sphericity, ld, bpe]
        has_any = any(pd.notna(x) for x in values)

        if not has_any:
            continue

        node_rows.append({
            "node_id": f"{patient}_{v['visit_label']}",
            "patient_folder": patient,
            "visit_label": v["visit_label"],
            "visit_index": v["visit_index"],
            "label_pCR": int(label) if pd.notna(label) else "",
            "ftv_volume": volume,
            "sphericity": sphericity,
            "longest_diameter": ld,
            "bpe_5slice_mean": bpe,
            "has_volume": int(pd.notna(volume)),
            "has_sphericity": int(pd.notna(sphericity)),
            "has_longest_diameter": int(pd.notna(ld)),
            "has_bpe": int(pd.notna(bpe)),
        })

nodes = pd.DataFrame(node_rows)
nodes.to_csv(TABLES / "phase9a_longitudinal_visit_nodes.csv", index=False)

# Temporal edge list
edge_rows = []
delta_rows = []

for patient, sub in nodes.groupby("patient_folder"):
    sub = sub.sort_values("visit_index").reset_index(drop=True)

    for i in range(len(sub) - 1):
        src = sub.iloc[i]
        dst = sub.iloc[i + 1]

        vol_src = pd.to_numeric(src["ftv_volume"], errors="coerce")
        vol_dst = pd.to_numeric(dst["ftv_volume"], errors="coerce")
        sph_src = pd.to_numeric(src["sphericity"], errors="coerce")
        sph_dst = pd.to_numeric(dst["sphericity"], errors="coerce")
        ld_src = pd.to_numeric(src["longest_diameter"], errors="coerce")
        ld_dst = pd.to_numeric(dst["longest_diameter"], errors="coerce")
        bpe_src = pd.to_numeric(src["bpe_5slice_mean"], errors="coerce")
        bpe_dst = pd.to_numeric(dst["bpe_5slice_mean"], errors="coerce")

        def diff(a, b):
            if pd.notna(a) and pd.notna(b):
                return b - a
            return np.nan

        def pct(a, b):
            if pd.notna(a) and pd.notna(b) and abs(a) > 1e-8:
                return 100.0 * (b - a) / a
            return np.nan

        edge_rows.append({
            "edge_id": f"{src['node_id']}__to__{dst['node_id']}",
            "patient_folder": patient,
            "source_node": src["node_id"],
            "target_node": dst["node_id"],
            "source_visit": src["visit_label"],
            "target_visit": dst["visit_label"],
            "source_visit_index": int(src["visit_index"]),
            "target_visit_index": int(dst["visit_index"]),
            "time_delta": int(dst["visit_index"] - src["visit_index"]),
            "label_pCR": int(src["label_pCR"]),
            "delta_ftv_volume": diff(vol_src, vol_dst),
            "pct_delta_ftv_volume": pct(vol_src, vol_dst),
            "delta_sphericity": diff(sph_src, sph_dst),
            "pct_delta_sphericity": pct(sph_src, sph_dst),
            "delta_longest_diameter": diff(ld_src, ld_dst),
            "pct_delta_longest_diameter": pct(ld_src, ld_dst),
            "delta_bpe_5slice_mean": diff(bpe_src, bpe_dst),
            "pct_delta_bpe_5slice_mean": pct(bpe_src, bpe_dst),
        })

    # patient-level deltas using first and last available visit
    if len(sub) >= 2:
        first = sub.iloc[0]
        last = sub.iloc[-1]

        vol_first = pd.to_numeric(first["ftv_volume"], errors="coerce")
        vol_last = pd.to_numeric(last["ftv_volume"], errors="coerce")
        sph_first = pd.to_numeric(first["sphericity"], errors="coerce")
        sph_last = pd.to_numeric(last["sphericity"], errors="coerce")
        ld_first = pd.to_numeric(first["longest_diameter"], errors="coerce")
        ld_last = pd.to_numeric(last["longest_diameter"], errors="coerce")
        bpe_first = pd.to_numeric(first["bpe_5slice_mean"], errors="coerce")
        bpe_last = pd.to_numeric(last["bpe_5slice_mean"], errors="coerce")

        delta_rows.append({
            "patient_folder": patient,
            "label_pCR": int(first["label_pCR"]),
            "first_visit": first["visit_label"],
            "last_visit": last["visit_label"],
            "n_visits": len(sub),
            "ftv_volume_first": vol_first,
            "ftv_volume_last": vol_last,
            "ftv_volume_delta_first_to_last": diff(vol_first, vol_last),
            "ftv_volume_pct_delta_first_to_last": pct(vol_first, vol_last),
            "sphericity_delta_first_to_last": diff(sph_first, sph_last),
            "sphericity_pct_delta_first_to_last": pct(sph_first, sph_last),
            "longest_diameter_delta_first_to_last": diff(ld_first, ld_last),
            "longest_diameter_pct_delta_first_to_last": pct(ld_first, ld_last),
            "bpe_delta_first_to_last": diff(bpe_first, bpe_last),
            "bpe_pct_delta_first_to_last": pct(bpe_first, bpe_last),
        })

edges = pd.DataFrame(edge_rows)
deltas = pd.DataFrame(delta_rows)

edges.to_csv(TABLES / "phase9a_temporal_edges.csv", index=False)
deltas.to_csv(TABLES / "phase9a_temporal_delta_features.csv", index=False)

# Patient graph summary
summary_rows = []
for patient, sub in nodes.groupby("patient_folder"):
    sub = sub.sort_values("visit_index")
    label = int(sub["label_pCR"].iloc[0])
    summary_rows.append({
        "patient_folder": patient,
        "label_pCR": label,
        "n_visit_nodes": len(sub),
        "first_visit": sub["visit_label"].iloc[0],
        "last_visit": sub["visit_label"].iloc[-1],
        "has_T0": int((sub["visit_label"] == "T0").any()),
        "has_T1": int((sub["visit_label"] == "T1").any()),
        "has_T2": int((sub["visit_label"] == "T2").any()),
        "has_T3": int((sub["visit_label"] == "T3").any()),
        "n_temporal_edges": max(0, len(sub) - 1),
    })

graph_summary = pd.DataFrame(summary_rows)
graph_summary.to_csv(TABLES / "phase9a_patient_graph_summary.csv", index=False)

# Markdown helper
def make_md(csv_path, md_path, title, max_rows=80, max_cols=12):
    data = pd.read_csv(csv_path, dtype=str).fillna("")
    small = data.head(max_rows).iloc[:, :max_cols]

    def clean(x):
        x = str(x).replace("|", "\\|").replace("\n", " ")
        if len(x) > 90:
            x = x[:87] + "..."
        return x

    lines = []
    lines.append("# " + title)
    lines.append("")
    lines.append("Rows: " + str(len(data)))
    lines.append("Columns: " + str(len(data.columns)))
    lines.append("")
    lines.append("| " + " | ".join(clean(c) for c in small.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(small.columns)) + " |")

    for _, r in small.iterrows():
        lines.append("| " + " | ".join(clean(v) for v in r.tolist()) + " |")

    md_path.write_text("\n".join(lines))

make_md(TABLES / "phase9a_longitudinal_visit_nodes.csv", VIEWS / "phase9a_longitudinal_visit_nodes.md", "Phase 9A Longitudinal Visit Nodes")
make_md(TABLES / "phase9a_temporal_edges.csv", VIEWS / "phase9a_temporal_edges.md", "Phase 9A Temporal Edges")
make_md(TABLES / "phase9a_patient_graph_summary.csv", VIEWS / "phase9a_patient_graph_summary.md", "Phase 9A Patient Graph Summary")
make_md(TABLES / "phase9a_temporal_delta_features.csv", VIEWS / "phase9a_temporal_delta_features.md", "Phase 9A Temporal Delta Features")

# Figures
# 280 visit availability heatmap
patients = list(graph_summary["patient_folder"])
visit_cols = ["has_T0", "has_T1", "has_T2", "has_T3"]
mat = graph_summary[visit_cols].values.astype(float)

plt.figure(figsize=(8, max(4, len(patients) * 0.25)))
plt.imshow(mat, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
plt.yticks(range(len(patients)), patients, fontsize=7)
plt.xticks(range(4), ["T0", "T1", "T2", "T3"])
plt.colorbar(label="visit available")
plt.title("Phase 9A Visit Availability")
plt.tight_layout()
plt.savefig(FIGS / "280_phase9a_visit_availability.png", dpi=220)
plt.close()

# 281 FTV trajectories
plt.figure(figsize=(9, 6))
for patient, sub in nodes.groupby("patient_folder"):
    sub = sub.sort_values("visit_index")
    y = pd.to_numeric(sub["ftv_volume"], errors="coerce")
    x = sub["visit_index"]
    label = int(sub["label_pCR"].iloc[0])
    marker = "o" if label == 1 else "s"
    plt.plot(x, y, marker=marker, alpha=0.75)
plt.xticks([0, 1, 2, 3], ["T0", "T1", "T2", "T3"])
plt.xlabel("Visit")
plt.ylabel("FTV / tumor volume feature")
plt.title("Phase 9A FTV Trajectories by Patient")
plt.tight_layout()
plt.savefig(FIGS / "281_phase9a_ftv_trajectories.png", dpi=220)
plt.close()

# 282 mean FTV by label
plt.figure(figsize=(9, 6))
if len(nodes) > 0:
    temp = nodes.copy()
    temp["ftv_volume"] = pd.to_numeric(temp["ftv_volume"], errors="coerce")
    temp["label_pCR"] = pd.to_numeric(temp["label_pCR"], errors="coerce")
    grouped = temp.groupby(["visit_index", "label_pCR"])["ftv_volume"].mean().reset_index()

    for lab, sub in grouped.groupby("label_pCR"):
        name = "pCR" if int(lab) == 1 else "non-pCR"
        plt.plot(sub["visit_index"], sub["ftv_volume"], marker="o", label=name)

    plt.xticks([0, 1, 2, 3], ["T0", "T1", "T2", "T3"])
    plt.xlabel("Visit")
    plt.ylabel("Mean FTV / tumor volume feature")
    plt.title("Phase 9A Mean FTV Trajectory by Label")
    plt.legend()
else:
    plt.text(0.5, 0.5, "No nodes available", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "282_phase9a_mean_ftv_by_label.png", dpi=220)
plt.close()

# 283 graph schema
plt.figure(figsize=(12, 7))
plt.axis("off")
text = (
    "Phase 9A graph-ready structure\n\n"
    "Patient graph\n"
    "  Node T0: volume, sphericity, longest diameter, BPE\n"
    "       ↓ temporal edge with delta features\n"
    "  Node T1: volume, sphericity, longest diameter, BPE\n"
    "       ↓\n"
    "  Node T2\n"
    "       ↓\n"
    "  Node T3\n\n"
    "Target label: pCR / non-pCR\n\n"
    "Next model idea\n"
    "  Classical baseline: delta features\n"
    "  Future model: temporal GNN over visit nodes"
)
plt.text(0.5, 0.5, text, ha="center", va="center", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "283_phase9a_graph_schema.png", dpi=220)
plt.close()

n_patients = graph_summary["patient_folder"].nunique()
n_nodes = len(nodes)
n_edges = len(edges)
n_delta = len(deltas)
n_pos = int((graph_summary["label_pCR"] == 1).sum())
n_neg = int((graph_summary["label_pCR"] == 0).sum())

report = []
report.append("# Phase 9A Longitudinal FTV Graph Dataset Report")
report.append("")
report.append("This phase converts the Phase 8C FTV modeling table into a graph-ready longitudinal patient-visit dataset.")
report.append("")
report.append("## Main outputs")
report.append("")
report.append("- reports/tables/phase9a_longitudinal_visit_nodes.csv")
report.append("- reports/tables/phase9a_temporal_edges.csv")
report.append("- reports/tables/phase9a_patient_graph_summary.csv")
report.append("- reports/tables/phase9a_temporal_delta_features.csv")
report.append("- reports/figures/280_phase9a_visit_availability.png")
report.append("- reports/figures/281_phase9a_ftv_trajectories.png")
report.append("- reports/figures/282_phase9a_mean_ftv_by_label.png")
report.append("- reports/figures/283_phase9a_graph_schema.png")
report.append("")
report.append("## Main counts")
report.append("")
report.append("- Patient graphs: " + str(n_patients))
report.append("- Visit nodes: " + str(n_nodes))
report.append("- Temporal edges: " + str(n_edges))
report.append("- Patient delta rows: " + str(n_delta))
report.append("- pCR positive graphs: " + str(n_pos))
report.append("- non-pCR graphs: " + str(n_neg))
report.append("")
report.append("## Interpretation")
report.append("")
report.append("The project now has a graph-ready representation. Each patient is represented by visit nodes and temporal edges. This is the correct bridge between the current FTV support-data model and the future temporal graph learning model.")
report.append("")
report.append("## Limitation")
report.append("")
report.append("This graph dataset is still small because it uses the 13 labeled patients from Phase 8C. It is ready for method design and pilot testing, not final clinical claims.")
report.append("")
report.append("## Next step")
report.append("")
report.append("Phase 9B should train a simple temporal-delta baseline model and compare it with Phase 8C. After that, a temporal graph neural network can be designed.")

(REPORTS / "Phase9A_Longitudinal_FTV_Graph_Dataset_Report.md").write_text("\n".join(report))

# Dashboard update
dash = REPORTS / "Dashboard.md"
old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
dash_add = (
    "\n\n## Phase 9A longitudinal FTV graph dataset\n\n"
    "- [Longitudinal graph dataset report](Phase9A_Longitudinal_FTV_Graph_Dataset_Report.md)\n"
    "- [Visit nodes](table_views/phase9a_longitudinal_visit_nodes.md)\n"
    "- [Temporal edges](table_views/phase9a_temporal_edges.md)\n"
    "- [Patient graph summary](table_views/phase9a_patient_graph_summary.md)\n"
)
if "Phase 9A longitudinal FTV graph dataset" not in old_dash:
    dash.write_text(old_dash.rstrip() + dash_add)

# README update
readme = REPO / "README.md"
old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
readme_add = (
    "\n\n### Phase 9A: longitudinal FTV graph-ready dataset\n\n"
    "Main outputs:\n\n"
    "- reports/Phase9A_Longitudinal_FTV_Graph_Dataset_Report.md\n"
    "- reports/tables/phase9a_longitudinal_visit_nodes.csv\n"
    "- reports/tables/phase9a_temporal_edges.csv\n"
)
if "Phase 9A: longitudinal FTV graph-ready dataset" not in old_readme:
    readme.write_text(old_readme.rstrip() + readme_add)

print("Phase 9A complete.")
print("Patient graphs:", n_patients)
print("Visit nodes:", n_nodes)
print("Temporal edges:", n_edges)
print("pCR positive graphs:", n_pos)
print("non-pCR graphs:", n_neg)
