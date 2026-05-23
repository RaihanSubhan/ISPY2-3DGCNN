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

STAGE10A = TABLES / "stage10a_dataset_structure_summary.csv"
STAGE10B = TABLES / "stage10b_preprocessed_visual_case_manifest.csv"
STAGE10C = TABLES / "stage10c_breast_pair_case_manifest.csv"
PHASE8D = TABLES / "phase8d_baseline_vs_ftv_model_comparison.csv"
PHASE9C = TABLES / "phase9c_model_decision_summary.csv"

required = [STAGE10B, STAGE10C]
for p in required:
    if not p.exists():
        raise SystemExit(f"Missing required file: {p}")

b = pd.read_csv(STAGE10B, dtype=str).fillna("")
c = pd.read_csv(STAGE10C, dtype=str).fillna("")

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def resolve_path(p):
    """
    Some paths in CSV are absolute Cradle paths.
    This keeps those if they exist, otherwise tries repo-relative figure paths.
    """
    p = str(p)
    if not p:
        return Path("")
    path = Path(p)
    if path.exists():
        return path
    alt = REPO / p
    if alt.exists():
        return alt
    alt2 = FIGS / Path(p).name
    if alt2.exists():
        return alt2
    return path

# ---------------------------------------------------------
# 1. Select the best display cases
# ---------------------------------------------------------
b_success = b[b.get("status", "").eq("success")].copy()
c_success = c[c.get("status", "").eq("success")].copy()

if "case_no" not in b_success.columns or "case_no" not in c_success.columns:
    raise SystemExit("Stage10B and Stage10C manifest tables must contain case_no.")

merged = b_success.merge(
    c_success,
    on=["case_no", "patient_folder", "study_folder", "seg_series_uid"],
    how="inner",
    suffixes=("_10b", "_10c")
)

if merged.empty:
    raise SystemExit("No matching successful Stage10B and Stage10C cases found.")

# Prefer visible but not huge masks.
if "mask_fraction_10c" in merged.columns:
    merged["mask_fraction_num"] = to_num(merged["mask_fraction_10c"])
elif "mask_fraction" in merged.columns:
    merged["mask_fraction_num"] = to_num(merged["mask_fraction"])
else:
    merged["mask_fraction_num"] = np.nan

merged["mask_fraction_num"] = merged["mask_fraction_num"].fillna(0)

# Good display score: visible masks, not massive.
merged["display_score"] = merged["mask_fraction_num"].apply(
    lambda x: x if 0.0005 <= x <= 0.05 else 0
)

if merged["display_score"].max() == 0:
    selected = merged.head(8).copy()
else:
    selected = merged.sort_values("display_score", ascending=False).head(8).copy()

selected.to_csv(TABLES / "stage10d_selected_thesis_cases.csv", index=False)

main_case = selected.iloc[0].to_dict()

def get_col(row, candidates):
    for col in candidates:
        if col in row and str(row[col]):
            return row[col]
    return ""

main_four = resolve_path(get_col(main_case, ["four_panel_figure", "four_panel_figure_10b"]))
main_pair = resolve_path(get_col(main_case, ["breast_pair_figure", "breast_pair_figure_10c"]))
main_obstacle = resolve_path(get_col(main_case, ["obstacle_proxy_figure", "obstacle_proxy_figure_10c"]))
main_transport = resolve_path(get_col(main_case, ["transport_field_figure", "transport_field_figure_10c"]))

# ---------------------------------------------------------
# 2. Load results for final story
# ---------------------------------------------------------
baseline_model = ""
baseline_auroc = np.nan
baseline_auprc = np.nan
baseline_brier = np.nan

ftv_model = ""
ftv_auroc = np.nan
ftv_auprc = np.nan
ftv_brier = np.nan

if PHASE8D.exists():
    comp = pd.read_csv(PHASE8D, dtype=str).fillna("")
    if len(comp) >= 2:
        base = comp.iloc[0]
        ftv = comp.iloc[1]
        baseline_model = base.get("best_model", "")
        ftv_model = ftv.get("best_model", "")
        baseline_auroc = float(base.get("AUROC", "nan"))
        baseline_auprc = float(base.get("AUPRC", "nan"))
        baseline_brier = float(base.get("Brier_score", "nan"))
        ftv_auroc = float(ftv.get("AUROC", "nan"))
        ftv_auprc = float(ftv.get("AUPRC", "nan"))
        ftv_brier = float(ftv.get("Brier_score", "nan"))

temporal_model = ""
temporal_auroc = np.nan
temporal_auprc = np.nan
temporal_brier = np.nan

if PHASE9C.exists():
    p9 = pd.read_csv(PHASE9C, dtype=str).fillna("")
    if len(p9) >= 2:
        t = p9.iloc[1]
        temporal_model = t.get("best_model", "")
        temporal_auroc = float(t.get("AUROC", "nan"))
        temporal_auprc = float(t.get("AUPRC", "nan"))
        temporal_brier = float(t.get("Brier_score", "nan"))

dataset_size = ""
patients = ""
series = ""
dicom_count = ""

if STAGE10A.exists():
    ds = pd.read_csv(STAGE10A, dtype=str).fillna("")
    dmap = dict(zip(ds["metric"], ds["value"]))
    dataset_size = dmap.get("dataset_size_du", "")
    patients = dmap.get("patients", "")
    series = dmap.get("series", "")
    dicom_count = dmap.get("dicom_files_counted", "")

# ---------------------------------------------------------
# 3. Helper functions for figures
# ---------------------------------------------------------
def show_image_or_text(ax, path, title):
    path = resolve_path(path)
    if path.exists():
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    else:
        ax.text(0.5, 0.5, f"Missing:\n{path.name}", ha="center", va="center")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

def make_md(csv_path, md_path, title, max_rows=80, max_cols=10):
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

# ---------------------------------------------------------
# 4. Master methods panel
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.reshape(-1)

show_image_or_text(axes[0], main_four, "A. Raw and preprocessed MRI with mask")
show_image_or_text(axes[1], main_pair, "B. Left/right breast-pair image")
show_image_or_text(axes[2], main_obstacle, "C. Tumor contrast-transport proxy")
show_image_or_text(axes[3], main_transport, "D. Toy transport field near tumor")

axes[4].axis("off")
text_result = (
    "E. Main pilot model\n\n"
    f"Dataset size: {dataset_size}\n"
    f"Patients in inventory: {patients}\n"
    f"Series: {series}\n"
    f"DICOM files counted: {dicom_count}\n\n"
    "Main modeling path:\n"
    "Official MRI / FTV support-data features\n\n"
    f"Best FTV model: {ftv_model}\n"
    f"AUROC: {ftv_auroc:.4f}\n"
    f"AUPRC: {ftv_auprc:.4f}\n"
    f"Brier: {ftv_brier:.4f}"
)
axes[4].text(0.02, 0.98, text_result, va="top", ha="left", fontsize=11)

axes[5].axis("off")
text_future = (
    "F. Future graph-learning direction\n\n"
    "Current decision:\n"
    "Do not train temporal GNN as the main model yet.\n\n"
    "Reason:\n"
    "Only 13 patient graphs are available and\n"
    "the temporal-delta model is weaker than FTV.\n\n"
    f"Temporal model: {temporal_model}\n"
    f"AUROC: {temporal_auroc:.4f}\n"
    f"AUPRC: {temporal_auprc:.4f}\n\n"
    "Future work:\n"
    "larger longitudinal FTV / PE / SER graph"
)
axes[5].text(0.02, 0.98, text_future, va="top", ha="left", fontsize=11)

fig.suptitle(
    "Stage 10D: Thesis-ready methods panel for ISPY2 FTV-based pCR prediction",
    fontsize=14
)
plt.tight_layout()
plt.savefig(FIGS / "340_stage10d_master_methods_panel.png", dpi=220)
plt.close()

# ---------------------------------------------------------
# 5. Contact sheet of selected cases
# ---------------------------------------------------------
fig, axes = plt.subplots(4, 2, figsize=(14, 20))
axes = axes.reshape(-1)

for ax, (_, row) in zip(axes, selected.iterrows()):
    path = resolve_path(get_col(row.to_dict(), ["four_panel_figure", "four_panel_figure_10b"]))
    title = f"Case {row.get('case_no', '')}: {row.get('patient_folder', '')}"
    show_image_or_text(ax, path, title)

for ax in axes[len(selected):]:
    ax.axis("off")

fig.suptitle("Stage 10D: Selected preprocessing and mask examples", fontsize=14)
plt.tight_layout()
plt.savefig(FIGS / "341_stage10d_selected_case_contact_sheet.png", dpi=180)
plt.close()

# ---------------------------------------------------------
# 6. Final pipeline and model result figure
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].axis("off")
txt0 = (
    "Pipeline\n\n"
    "ISPY2 DICOM on Cradle\n"
    "↓\n"
    "Dataset inventory\n"
    "↓\n"
    "SEG diagnosis\n"
    "↓\n"
    "Recovered visual masks\n"
    "↓\n"
    "Official FTV support features\n"
    "↓\n"
    "pCR prediction"
)
axes[0].text(0.5, 0.5, txt0, ha="center", va="center", fontsize=12)
axes[0].set_title("A. Data-to-model pipeline")

axes[1].bar(
    ["Phase 5A\nbaseline", "Phase 8C\nFTV"],
    [baseline_auroc, ftv_auroc]
)
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("AUROC")
axes[1].set_title("B. Main model improvement")

axes[2].bar(
    ["FTV\nsupport model", "Temporal-delta\nbaseline"],
    [ftv_auroc, temporal_auroc]
)
axes[2].set_ylim(0, 1)
axes[2].set_ylabel("AUROC")
axes[2].set_title("C. GNN decision evidence")

plt.tight_layout()
plt.savefig(FIGS / "342_stage10d_final_pipeline_and_model_result.png", dpi=220)
plt.close()

# ---------------------------------------------------------
# 7. Figure legend / paper-use map
# ---------------------------------------------------------
caption_rows = [
    {
        "figure_file": "340_stage10d_master_methods_panel.png",
        "paper_section": "Methods overview",
        "caption": "Composite panel showing preprocessing, recovered tumor mask overlay, breast-pair formation, contrast-transport proxy visualization, and the final modeling decision."
    },
    {
        "figure_file": "341_stage10d_selected_case_contact_sheet.png",
        "paper_section": "Preprocessing visual QC",
        "caption": "Representative case examples showing raw and preprocessed MRI with recovered tumor-mask candidates."
    },
    {
        "figure_file": "342_stage10d_final_pipeline_and_model_result.png",
        "paper_section": "Main results",
        "caption": "Summary of the analysis path and model comparison showing that the official MRI/FTV support-data model is the strongest pilot result."
    },
    {
        "figure_file": "270_phase8e_project_pipeline.png",
        "paper_section": "Study workflow",
        "caption": "Overall project pipeline from ISPY2 data organization to FTV modeling and future temporal graph learning."
    },
    {
        "figure_file": "283_phase9a_graph_schema.png",
        "paper_section": "Future work",
        "caption": "Graph-ready longitudinal representation where patient visits become nodes and temporal edges encode treatment progression."
    }
]

captions = pd.DataFrame(caption_rows)
captions.to_csv(TABLES / "stage10d_thesis_figure_caption_table.csv", index=False)

story_rows = [
    {
        "item": "Main current model",
        "value": "Phase 8C FTV SVM_RBF",
        "meaning": "Use this as the main pilot pCR prediction result."
    },
    {
        "item": "Best AUROC",
        "value": f"{ftv_auroc:.4f}",
        "meaning": "FTV support-data model performs better than baseline and temporal-delta models."
    },
    {
        "item": "Why not GNN yet",
        "value": "small_n",
        "meaning": "Only 13 patient graphs are available, so temporal GNN should be future work."
    },
    {
        "item": "How to explain obstacle map",
        "value": "conceptual proxy",
        "meaning": "It visualizes contrast-transport disturbance, not measured blood/milk/fluid flow."
    },
    {
        "item": "Safe paper claim",
        "value": "pilot feasibility",
        "meaning": "This is not clinical validation."
    }
]

story = pd.DataFrame(story_rows)
story.to_csv(TABLES / "stage10d_final_thesis_story_summary.csv", index=False)

make_md(
    TABLES / "stage10d_selected_thesis_cases.csv",
    VIEWS / "stage10d_selected_thesis_cases.md",
    "Stage 10D Selected Thesis Cases"
)
make_md(
    TABLES / "stage10d_thesis_figure_caption_table.csv",
    VIEWS / "stage10d_thesis_figure_caption_table.md",
    "Stage 10D Thesis Figure Caption Table"
)
make_md(
    TABLES / "stage10d_final_thesis_story_summary.csv",
    VIEWS / "stage10d_final_thesis_story_summary.md",
    "Stage 10D Final Thesis Story Summary"
)

# Simple caption map figure
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")
legend_text = (
    "Stage 10D figure package\n\n"
    "Figure 340: Master methods panel\n"
    "Figure 341: Selected preprocessing examples\n"
    "Figure 342: Final pipeline and model result\n\n"
    "Use in thesis:\n"
    "Methods: Fig. 340 and Fig. 341\n"
    "Results: Fig. 342\n"
    "Future work: Phase 9A/9C graph figures\n\n"
    "Main message:\n"
    "FTV support features are the main current result.\n"
    "Temporal graph learning is future work."
)
ax.text(0.5, 0.5, legend_text, ha="center", va="center", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "343_stage10d_paper_figure_legend_map.png", dpi=220)
plt.close()

# ---------------------------------------------------------
# 8. Report
# ---------------------------------------------------------
report = []
report.append("# Stage 10D Thesis Methods Figure Package Report")
report.append("")
report.append("This stage packages the Stage 10A, Stage 10B, and Stage 10C visual outputs into thesis-ready figures.")
report.append("")
report.append("## Main outputs")
report.append("")
report.append("- reports/figures/340_stage10d_master_methods_panel.png")
report.append("- reports/figures/341_stage10d_selected_case_contact_sheet.png")
report.append("- reports/figures/342_stage10d_final_pipeline_and_model_result.png")
report.append("- reports/figures/343_stage10d_paper_figure_legend_map.png")
report.append("- reports/tables/stage10d_selected_thesis_cases.csv")
report.append("- reports/tables/stage10d_thesis_figure_caption_table.csv")
report.append("- reports/tables/stage10d_final_thesis_story_summary.csv")
report.append("")
report.append("## Selected main case")
report.append("")
report.append("- Patient: " + str(main_case.get("patient_folder", "")))
report.append("- Case number: " + str(main_case.get("case_no", "")))
report.append("- Mask fraction: " + str(main_case.get("mask_fraction_num", "")))
report.append("")
report.append("## Main thesis story")
report.append("")
report.append("The visual preprocessing and breast-pair figures support the imaging-method section. The primary research result remains the official MRI/FTV support-data pCR prediction model.")
report.append("")
report.append("## Safe interpretation")
report.append("")
report.append("The obstacle proxy is a conceptual contrast-transport disturbance visualization. It should not be described as measured blood flow, milk flow, or fluid flow.")
report.append("")
report.append("## Next step")
report.append("")
report.append("Use these figures to write the methods and results sections. After this, create the final Word report and PowerPoint.")

(REPORTS / "Stage10D_Thesis_Methods_Figure_Package_Report.md").write_text("\n".join(report))

# Dashboard / README
dash = REPORTS / "Dashboard.md"
old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
dash_add = (
    "\n\n## Stage 10D thesis methods figure package\n\n"
    "- [Thesis methods figure package report](Stage10D_Thesis_Methods_Figure_Package_Report.md)\n"
    "- [Selected thesis cases](table_views/stage10d_selected_thesis_cases.md)\n"
    "- [Figure caption table](table_views/stage10d_thesis_figure_caption_table.md)\n"
)
if "Stage 10D thesis methods figure package" not in old_dash:
    dash.write_text(old_dash.rstrip() + dash_add)

readme = REPO / "README.md"
old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
readme_add = (
    "\n\n### Stage 10D: thesis methods figure package\n\n"
    "Main outputs:\n\n"
    "- reports/Stage10D_Thesis_Methods_Figure_Package_Report.md\n"
    "- reports/figures/340_stage10d_master_methods_panel.png\n"
    "- reports/figures/341_stage10d_selected_case_contact_sheet.png\n"
    "- reports/figures/342_stage10d_final_pipeline_and_model_result.png\n"
)
if "Stage 10D: thesis methods figure package" not in old_readme:
    readme.write_text(old_readme.rstrip() + readme_add)

print("Stage 10D complete.")
print("Selected cases:", len(selected))
print("Main case patient:", main_case.get("patient_folder", ""))
print("Main FTV model:", ftv_model)
print("FTV AUROC:", round(ftv_auroc, 4))
print("Temporal AUROC:", round(temporal_auroc, 4))
