from pathlib import Path
import subprocess
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path.home() / "ISPY2-3DGCNN"
REPORTS = REPO / "reports"
FINAL = REPORTS / "final_package"
TABLES = REPORTS / "tables"
FIGS = REPORTS / "figures"
VIEWS = REPORTS / "table_views"

for p in [REPORTS, FINAL, TABLES, FIGS, VIEWS]:
    p.mkdir(parents=True, exist_ok=True)

def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, cwd=REPO, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as e:
        return e.output.strip()

def exists_rel(path):
    p = REPO / path
    return "yes" if p.exists() else "no"

def size_mb(path):
    p = REPO / path
    if p.exists():
        return round(p.stat().st_size / (1024 * 1024), 4)
    return ""

final_files = [
    "README.md",
    "reports/Dashboard.md",
    "reports/Final_GitHub_Output_Index.md",
    "reports/Stage10G_Final_Handoff_Fix_Report.md",
    "reports/final_package/ISPY2_FTV_pCR_Final_Report.docx",
    "reports/final_package/ISPY2_FTV_pCR_Final_Presentation.pptx",
    "reports/final_package/ISPY2_FTV_pCR_One_Page_Summary.md",
    "reports/final_package/ISPY2_FTV_pCR_Professor_Talk_Track.md",
    "reports/final_package/Professor_Email_Draft.md",
    "reports/final_package/ISPY2_FTV_pCR_Professor_Review_Bundle.zip",
    "reports/figures/340_stage10d_master_methods_panel.png",
    "reports/figures/342_stage10d_final_pipeline_and_model_result.png",
    "reports/figures/260_phase8d_baseline_vs_ftv_metrics.png",
    "reports/figures/300_phase9c_model_decision.png",
    "reports/figures/283_phase9a_graph_schema.png",
    "reports/tables/phase8c_ftv_model_performance.csv",
    "reports/tables/phase8d_baseline_vs_ftv_model_comparison.csv",
    "reports/tables/phase9c_model_decision_summary.csv",
    "reports/tables/stage10g_final_send_checklist.csv",
]

tracked_files = run("git ls-files").splitlines()
tracked_set = set(tracked_files)

rows = []
for f in final_files:
    rows.append({
        "file": f,
        "exists": exists_rel(f),
        "tracked_by_git": "yes" if f in tracked_set else "no",
        "size_mb": size_mb(f),
    })

qc = pd.DataFrame(rows)
qc.to_csv(TABLES / "stage10h_final_repository_freeze_qc.csv", index=False)

missing = int((qc["exists"] == "no").sum())
untracked = int((qc["tracked_by_git"] == "no").sum())

# Update Stage 10G report wording so it is not outdated.
stage10g = REPORTS / "Stage10G_Final_Handoff_Fix_Report.md"
if stage10g.exists():
    text = stage10g.read_text()
    old = "Force-add the ZIP bundle because it is ignored by .gitignore, then push to GitHub."
    new = "The ZIP bundle has been force-added and pushed. Send the final Word report, PowerPoint, or review bundle to the professor."
    if old in text:
        stage10g.write_text(text.replace(old, new))

# Final send message.
send_msg = []
send_msg.append("# Final Send Message to Professor")
send_msg.append("")
send_msg.append("Subject: ISPY2 breast MRI pilot project package")
send_msg.append("")
send_msg.append("Dear Professor,")
send_msg.append("")
send_msg.append("I have completed the first full research cycle for my ISPY2 breast MRI project. The final package includes a Word report, PowerPoint presentation, one-page summary, and review bundle.")
send_msg.append("")
send_msg.append("The main current result is a pilot pCR prediction model using official MRI/FTV support-data features. The best model is SVM_RBF with AUROC 0.6667, AUPRC 0.7528, and Brier score 0.2405.")
send_msg.append("")
send_msg.append("A key technical finding is that the raw DICOM SEG mask path was not reliable enough for the main tumor-source pathway, so the project moved toward official MRI/FTV support-data features.")
send_msg.append("")
send_msg.append("I also built a graph-ready longitudinal FTV dataset, but the temporal-delta model was weaker than the FTV model. Therefore, temporal GNN is included as future work rather than the current main model.")
send_msg.append("")
send_msg.append("I would appreciate your feedback on the title, main claim, limitation wording, and future graph-learning direction.")
send_msg.append("")
send_msg.append("Best regards,")
send_msg.append("Raihan")
(FINAL / "Final_Send_Message_to_Professor.md").write_text("\n".join(send_msg))

# Release notes.
release = []
release.append("# Release Notes: v1.0-pilot-ftv-pcr")
release.append("")
release.append("## Project title")
release.append("")
release.append("Pilot FTV-Based Prediction of Pathologic Complete Response in I-SPY2 Breast MRI with Tumor-Source Validation and a Future Temporal Graph Learning Framework")
release.append("")
release.append("## Final current result")
release.append("")
release.append("- Main pilot model: Phase 8C FTV support-data SVM_RBF")
release.append("- AUROC: 0.6667")
release.append("- AUPRC: 0.7528")
release.append("- Brier score: 0.2405")
release.append("")
release.append("## Final decision")
release.append("")
release.append("The FTV support-data model is the main current result. Temporal GNN is future work because the graph-ready dataset is too small and the temporal-delta model did not beat the FTV model.")
release.append("")
release.append("## Final package")
release.append("")
release.append("- Word report")
release.append("- PowerPoint presentation")
release.append("- One-page summary")
release.append("- Professor talk track")
release.append("- Professor review ZIP bundle")
release.append("")
release.append("## Important limitation")
release.append("")
release.append("This is a pilot feasibility study, not clinical validation.")
(FINAL / "RELEASE_NOTES_v1.0-pilot-ftv-pcr.md").write_text("\n".join(release))

# Root FINAL_STATUS file.
final_status = []
final_status.append("# FINAL STATUS")
final_status.append("")
final_status.append("The first research cycle is complete.")
final_status.append("")
final_status.append("## Main result")
final_status.append("")
final_status.append("The current main pilot result is the official MRI/FTV support-data SVM_RBF model:")
final_status.append("")
final_status.append("- AUROC = 0.6667")
final_status.append("- AUPRC = 0.7528")
final_status.append("- Brier score = 0.2405")
final_status.append("")
final_status.append("## Final decision")
final_status.append("")
final_status.append("Use FTV support-data modeling as the main current result. Keep temporal graph learning as future work.")
final_status.append("")
final_status.append("## Handoff")
final_status.append("")
final_status.append("Use files in `reports/final_package/` for professor review.")
(REPO / "FINAL_STATUS.md").write_text("\n".join(final_status))

# Markdown views.
def make_md(csv_path, md_path, title):
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    small = df.head(100)
    def clean(x):
        x = str(x).replace("|", "\\|").replace("\n", " ")
        return x[:87] + "..." if len(x) > 90 else x
    lines = ["# " + title, "", f"Rows: {len(df)}", f"Columns: {len(df.columns)}", ""]
    lines.append("| " + " | ".join(clean(c) for c in small.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(small.columns)) + " |")
    for _, r in small.iterrows():
        lines.append("| " + " | ".join(clean(v) for v in r.tolist()) + " |")
    md_path.write_text("\n".join(lines))

make_md(TABLES / "stage10h_final_repository_freeze_qc.csv",
        VIEWS / "stage10h_final_repository_freeze_qc.md",
        "Stage 10H Final Repository Freeze QC")

# Figures.
plt.figure(figsize=(9, 5))
qc["exists"].value_counts().sort_values().plot(kind="barh")
plt.title("Stage 10H File Existence Check")
plt.xlabel("Files")
plt.tight_layout()
plt.savefig(FIGS / "370_stage10h_file_existence_check.png", dpi=220)
plt.close()

plt.figure(figsize=(9, 5))
qc["tracked_by_git"].value_counts().sort_values().plot(kind="barh")
plt.title("Stage 10H Git Tracking Check")
plt.xlabel("Files")
plt.tight_layout()
plt.savefig(FIGS / "371_stage10h_git_tracking_check.png", dpi=220)
plt.close()

plt.figure(figsize=(11, 6))
plt.axis("off")
txt = (
    "Final repository freeze\n\n"
    "Main model: Phase 8C FTV SVM_RBF\n"
    "AUROC 0.6667 | AUPRC 0.7528 | Brier 0.2405\n\n"
    "Final decision:\n"
    "FTV support-data model is the main pilot result.\n"
    "Temporal GNN is future work.\n\n"
    "Package:\n"
    "Word report + PowerPoint + professor review ZIP\n\n"
    f"Missing files: {missing}\n"
    f"Untracked final files: {untracked}"
)
plt.text(0.5, 0.5, txt, ha="center", va="center", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "372_stage10h_final_freeze_summary.png", dpi=220)
plt.close()

# Stage 10H report.
report = []
report.append("# Stage 10H Final Repository Freeze Report")
report.append("")
report.append("This stage freezes the repository after the final professor handoff package.")
report.append("")
report.append("## Main outputs")
report.append("")
report.append("- FINAL_STATUS.md")
report.append("- reports/final_package/Final_Send_Message_to_Professor.md")
report.append("- reports/final_package/RELEASE_NOTES_v1.0-pilot-ftv-pcr.md")
report.append("- reports/tables/stage10h_final_repository_freeze_qc.csv")
report.append("- reports/table_views/stage10h_final_repository_freeze_qc.md")
report.append("- reports/figures/370_stage10h_file_existence_check.png")
report.append("- reports/figures/371_stage10h_git_tracking_check.png")
report.append("- reports/figures/372_stage10h_final_freeze_summary.png")
report.append("")
report.append("## Main counts")
report.append("")
report.append("- Final files checked: " + str(len(qc)))
report.append("- Missing files: " + str(missing))
report.append("- Final files not tracked by Git: " + str(untracked))
report.append("")
report.append("## Final status")
report.append("")
if missing == 0 and untracked == 0:
    report.append("The repository is ready to freeze and tag.")
else:
    report.append("Some files are missing or untracked. Add/restore them before freezing.")
report.append("")
report.append("## Next step")
report.append("")
report.append("Commit Stage 10H, push it, and create the tag `v1.0-pilot-ftv-pcr`.")

(REPORTS / "Stage10H_Final_Repository_Freeze_Report.md").write_text("\n".join(report))

# Update Dashboard, README.
dash = REPORTS / "Dashboard.md"
old = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
add = (
    "\n\n## Stage 10H final repository freeze\n\n"
    "- [Final repository freeze report](Stage10H_Final_Repository_Freeze_Report.md)\n"
    "- [Final repository freeze QC](table_views/stage10h_final_repository_freeze_qc.md)\n"
    "- [Release notes](final_package/RELEASE_NOTES_v1.0-pilot-ftv-pcr.md)\n"
)
if "Stage 10H final repository freeze" not in old:
    dash.write_text(old.rstrip() + add)

readme = REPO / "README.md"
old = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
add = (
    "\n\n### Stage 10H: final repository freeze\n\n"
    "Main outputs:\n\n"
    "- FINAL_STATUS.md\n"
    "- reports/Stage10H_Final_Repository_Freeze_Report.md\n"
    "- reports/final_package/RELEASE_NOTES_v1.0-pilot-ftv-pcr.md\n"
)
if "Stage 10H: final repository freeze" not in old:
    readme.write_text(old.rstrip() + add)

print("Stage 10H complete.")
print("Final files checked:", len(qc))
print("Missing files:", missing)
print("Untracked final files:", untracked)
