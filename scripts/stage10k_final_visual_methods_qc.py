from pathlib import Path
import pandas as pd
import numpy as np

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

def exists(path):
    return (REPO / path).exists()

def size_mb(path):
    p = REPO / path
    if p.exists():
        return round(p.stat().st_size / (1024 * 1024), 4)
    return ""

def read_csv(path):
    p = REPO / path
    if p.exists():
        return pd.read_csv(p, dtype=str).fillna("")
    return pd.DataFrame()

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

def add_image_or_text(ax, path, title):
    p = REPO / path
    if p.exists():
        img = plt.imread(p)
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    else:
        ax.text(0.5, 0.5, "Missing\n" + path, ha="center", va="center")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

# ---------------------------------------------------------
# 1. Check the four requested tasks
# ---------------------------------------------------------
task_rows = [
    {
        "task_no": 1,
        "requested_task": "Dataset structure with size and output",
        "status": "done" if exists("reports/tables/stage10a_dataset_structure_summary.csv") else "missing",
        "main_report": "reports/Stage10A_Dataset_Visual_Preprocessing_Report.md",
        "main_table": "reports/tables/stage10a_dataset_structure_summary.csv",
        "main_figure": "reports/figures/310_stage10a_dataset_modality_counts.png",
        "notes": "Reports size, patient count, studies, series, and DICOM files."
    },
    {
        "task_no": 2,
        "requested_task": "Preprocessed dataset without mask and with mask visual output",
        "status": "done" if exists("reports/figures/320_stage10b_without_mask_grid.png") and exists("reports/figures/321_stage10b_with_mask_grid.png") else "missing",
        "main_report": "reports/Stage10B_Expanded_Preprocessed_Visual_Output_Report.md",
        "main_table": "reports/tables/stage10b_preprocessed_visual_case_manifest.csv",
        "main_figure": "reports/figures/322_stage10b_four_panel_grid.png",
        "notes": "Raw MRI, preprocessed MRI, and mask overlay panels."
    },
    {
        "task_no": 3,
        "requested_task": "Segmentation top raw MRI + raw mask, bottom preprocessed MRI + preprocessed mask",
        "status": "done" if exists("reports/figures/312_stage10a_segmentation_top_bottom_grid.png") or exists("reports/figures/322_stage10b_four_panel_grid.png") else "missing",
        "main_report": "reports/Stage10B_Expanded_Preprocessed_Visual_Output_Report.md",
        "main_table": "reports/tables/stage10b_preprocessed_visual_case_manifest.csv",
        "main_figure": "reports/figures/322_stage10b_four_panel_grid.png",
        "notes": "Use Stage 10B four-panel grid as the main thesis-ready segmentation visual."
    },
    {
        "task_no": 4,
        "requested_task": "Merge left and right breasts into a single breast pair with obstacle-style visualization",
        "status": "done" if exists("reports/figures/330_stage10c_breast_pair_grid.png") and exists("reports/figures/331_stage10c_obstacle_proxy_grid.png") else "missing",
        "main_report": "reports/Stage10C_Breast_Pair_Obstacle_Visualization_Report.md",
        "main_table": "reports/tables/stage10c_breast_pair_case_manifest.csv",
        "main_figure": "reports/figures/330_stage10c_breast_pair_grid.png",
        "notes": "Obstacle map must be described as contrast-transport proxy only, not direct fluid flow."
    },
]

task_df = pd.DataFrame(task_rows)
task_csv = TABLES / "stage10k_four_task_completion_checklist.csv"
task_df.to_csv(task_csv, index=False)

make_md(
    task_csv,
    VIEWS / "stage10k_four_task_completion_checklist.md",
    "Stage 10K Four Task Completion Checklist"
)

# ---------------------------------------------------------
# 2. Select best visual cases from Stage 10D, 10B, 10C
# ---------------------------------------------------------
stage10d = read_csv("reports/tables/stage10d_selected_thesis_cases.csv")
stage10b = read_csv("reports/tables/stage10b_preprocessed_visual_case_manifest.csv")
stage10c = read_csv("reports/tables/stage10c_breast_pair_case_manifest.csv")

best_rows = []

if not stage10d.empty:
    for _, row in stage10d.head(8).iterrows():
        best_rows.append({
            "case_no": row.get("case_no", ""),
            "patient_folder": row.get("patient_folder", ""),
            "reason_selected": "selected by Stage 10D display score",
            "four_panel_figure": row.get("four_panel_figure", row.get("four_panel_figure_10b", "")),
            "breast_pair_figure": row.get("breast_pair_figure", row.get("breast_pair_figure_10c", "")),
            "obstacle_proxy_figure": row.get("obstacle_proxy_figure", row.get("obstacle_proxy_figure_10c", "")),
            "transport_field_figure": row.get("transport_field_figure", row.get("transport_field_figure_10c", "")),
        })

elif not stage10b.empty:
    good = stage10b[stage10b.get("status", "").eq("success")].head(8)
    for _, row in good.iterrows():
        case_no = row.get("case_no", "")
        c_row = stage10c[stage10c.get("case_no", "").astype(str).eq(str(case_no))]
        c_first = c_row.iloc[0] if len(c_row) else {}
        best_rows.append({
            "case_no": case_no,
            "patient_folder": row.get("patient_folder", ""),
            "reason_selected": "fallback from Stage 10B success list",
            "four_panel_figure": row.get("four_panel_figure", ""),
            "breast_pair_figure": c_first.get("breast_pair_figure", "") if len(c_row) else "",
            "obstacle_proxy_figure": c_first.get("obstacle_proxy_figure", "") if len(c_row) else "",
            "transport_field_figure": c_first.get("transport_field_figure", "") if len(c_row) else "",
        })

best_df = pd.DataFrame(best_rows)
best_csv = TABLES / "stage10k_best_visual_cases.csv"
best_df.to_csv(best_csv, index=False)

make_md(
    best_csv,
    VIEWS / "stage10k_best_visual_cases.md",
    "Stage 10K Best Visual Cases"
)

# ---------------------------------------------------------
# 3. Thesis visual figure plan
# ---------------------------------------------------------
figure_plan = pd.DataFrame([
    {
        "figure_order": 1,
        "figure_title": "Dataset structure and pipeline",
        "file_to_show": "reports/figures/270_phase8e_project_pipeline.png",
        "paper_section": "Methods overview",
        "why_show_it": "Explains the full data and modeling workflow."
    },
    {
        "figure_order": 2,
        "figure_title": "Dataset size and modality structure",
        "file_to_show": "reports/figures/310_stage10a_dataset_modality_counts.png",
        "paper_section": "Dataset",
        "why_show_it": "Supports dataset structure and DICOM inventory."
    },
    {
        "figure_order": 3,
        "figure_title": "Raw and preprocessed MRI with mask",
        "file_to_show": "reports/figures/322_stage10b_four_panel_grid.png",
        "paper_section": "Preprocessing",
        "why_show_it": "Shows preprocessing without mask and with mask."
    },
    {
        "figure_order": 4,
        "figure_title": "Breast-pair and contrast-transport proxy",
        "file_to_show": "reports/figures/331_stage10c_obstacle_proxy_grid.png",
        "paper_section": "Visual methods",
        "why_show_it": "Shows the single breast-pair and conceptual tumor obstacle map."
    },
    {
        "figure_order": 5,
        "figure_title": "Final methods panel",
        "file_to_show": "reports/figures/340_stage10d_master_methods_panel.png",
        "paper_section": "Methods summary",
        "why_show_it": "Best one-figure summary of the visual methods."
    },
    {
        "figure_order": 6,
        "figure_title": "Main model comparison",
        "file_to_show": "reports/figures/260_phase8d_baseline_vs_ftv_metrics.png",
        "paper_section": "Results",
        "why_show_it": "Shows FTV model improvement over baseline."
    },
    {
        "figure_order": 7,
        "figure_title": "Temporal graph future-work decision",
        "file_to_show": "reports/figures/300_phase9c_model_decision.png",
        "paper_section": "Discussion and future work",
        "why_show_it": "Shows why temporal GNN is future work, not the current main model."
    },
])

plan_csv = TABLES / "stage10k_thesis_visual_figure_plan.csv"
figure_plan.to_csv(plan_csv, index=False)

make_md(
    plan_csv,
    VIEWS / "stage10k_thesis_visual_figure_plan.md",
    "Stage 10K Thesis Visual Figure Plan"
)

# ---------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------
plt.figure(figsize=(9, 5))
task_df["status"].value_counts().sort_values().plot(kind="barh")
plt.title("Stage 10K Four Requested Tasks Status")
plt.xlabel("Tasks")
plt.tight_layout()
plt.savefig(FIGS / "400_stage10k_four_task_done_checklist.png", dpi=220)
plt.close()

# Best case panel.
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.reshape(-1)

if not best_df.empty:
    main = best_df.iloc[0]
    add_image_or_text(axes[0], main.get("four_panel_figure", ""), "A. Raw/preprocessed MRI with mask")
    add_image_or_text(axes[1], main.get("breast_pair_figure", ""), "B. Single breast-pair image")
    add_image_or_text(axes[2], main.get("obstacle_proxy_figure", ""), "C. Contrast-transport obstacle proxy")
    add_image_or_text(axes[3], main.get("transport_field_figure", ""), "D. Toy transport field")
else:
    for ax in axes:
        ax.text(0.5, 0.5, "No selected cases", ha="center", va="center")
        ax.axis("off")

fig.suptitle("Stage 10K Best Visual Case Panel", fontsize=14)
plt.tight_layout()
plt.savefig(FIGS / "401_stage10k_best_visual_case_panel.png", dpi=220)
plt.close()

# Storyboard.
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.reshape(-1)

add_image_or_text(axes[0], "reports/figures/270_phase8e_project_pipeline.png", "A. Project pipeline")
add_image_or_text(axes[1], "reports/figures/322_stage10b_four_panel_grid.png", "B. Preprocessing and mask")
add_image_or_text(axes[2], "reports/figures/331_stage10c_obstacle_proxy_grid.png", "C. Breast-pair obstacle proxy")
add_image_or_text(axes[3], "reports/figures/260_phase8d_baseline_vs_ftv_metrics.png", "D. Main model result")

fig.suptitle("Stage 10K Thesis Visual Storyboard", fontsize=14)
plt.tight_layout()
plt.savefig(FIGS / "402_stage10k_thesis_visual_storyboard.png", dpi=220)
plt.close()

# ---------------------------------------------------------
# 5. Report
# ---------------------------------------------------------
n_done = int((task_df["status"] == "done").sum())
n_missing = int((task_df["status"] != "done").sum())

report = []
report.append("# Stage 10K Final Visual Methods QC Report")
report.append("")
report.append("This stage checks the four requested visual-methods tasks and prepares a thesis-ready visual figure plan.")
report.append("")
report.append("## Four requested tasks")
report.append("")
for _, row in task_df.iterrows():
    report.append(f"- Task {row['task_no']}: {row['requested_task']} — **{row['status']}**")
report.append("")
report.append("## Main outputs")
report.append("")
report.append("- reports/tables/stage10k_four_task_completion_checklist.csv")
report.append("- reports/tables/stage10k_best_visual_cases.csv")
report.append("- reports/tables/stage10k_thesis_visual_figure_plan.csv")
report.append("- reports/figures/400_stage10k_four_task_done_checklist.png")
report.append("- reports/figures/401_stage10k_best_visual_case_panel.png")
report.append("- reports/figures/402_stage10k_thesis_visual_storyboard.png")
report.append("")
report.append("## Main counts")
report.append("")
report.append("- Tasks done: " + str(n_done))
report.append("- Tasks missing: " + str(n_missing))
report.append("")
report.append("## Important wording")
report.append("")
report.append("The breast-pair obstacle map must be described as a conceptual contrast-transport disturbance proxy. It is not a direct measure of blood flow, milk flow, or fluid flow.")
report.append("")
report.append("## Next step")
report.append("")
if n_missing == 0:
    report.append("All four visual-method tasks are complete. Use the Stage 10K figure plan to write the thesis methods section.")
else:
    report.append("Some visual-method tasks are missing. Check the checklist and rerun the corresponding stage.")

(REPORTS / "Stage10K_Final_Visual_Methods_QC_Report.md").write_text("\n".join(report))

# Dashboard and README.
dash = REPORTS / "Dashboard.md"
old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
dash_add = (
    "\n\n## Stage 10K final visual methods QC\n\n"
    "- [Final visual methods QC report](Stage10K_Final_Visual_Methods_QC_Report.md)\n"
    "- [Four task checklist](table_views/stage10k_four_task_completion_checklist.md)\n"
    "- [Thesis visual figure plan](table_views/stage10k_thesis_visual_figure_plan.md)\n"
)
if "Stage 10K final visual methods QC" not in old_dash:
    dash.write_text(old_dash.rstrip() + dash_add)

readme = REPO / "README.md"
old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
readme_add = (
    "\n\n### Stage 10K: final visual methods QC\n\n"
    "Main outputs:\n\n"
    "- reports/Stage10K_Final_Visual_Methods_QC_Report.md\n"
    "- reports/tables/stage10k_four_task_completion_checklist.csv\n"
    "- reports/figures/401_stage10k_best_visual_case_panel.png\n"
)
if "Stage 10K: final visual methods QC" not in old_readme:
    readme.write_text(old_readme.rstrip() + readme_add)

print("Stage 10K complete.")
print("Tasks done:", n_done)
print("Tasks missing:", n_missing)
if not best_df.empty:
    print("Best visual case:", best_df.iloc[0].get("patient_folder", ""))
