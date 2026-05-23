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
        return subprocess.check_output(
            cmd, shell=True, cwd=REPO, text=True, stderr=subprocess.STDOUT
        ).strip()
    except subprocess.CalledProcessError as e:
        return e.output.strip()

def replace_text(path, old, new):
    path = Path(path)
    if not path.exists():
        return "missing"
    text = path.read_text()
    if old in text:
        path.write_text(text.replace(old, new))
        return "updated"
    if new in text:
        return "already_updated"
    return "old_text_not_found"

# Fix outdated Stage 10G wording.
stage10g = REPORTS / "Stage10G_Final_Handoff_Fix_Report.md"
g_status = replace_text(
    stage10g,
    "Force-add the ZIP bundle because it is ignored by .gitignore, then push to GitHub.",
    "The ZIP bundle has been force-added, tracked by Git, and pushed. The package is ready for professor review."
)

# Fix outdated Stage 10H wording.
stage10h = REPORTS / "Stage10H_Final_Repository_Freeze_Report.md"
h_status = replace_text(
    stage10h,
    "Commit Stage 10H, push it, and create the tag `v1.0-pilot-ftv-pcr`.",
    "Stage 10H has been committed, pushed, and tagged as `v1.0-pilot-ftv-pcr`. No further code is needed for this research cycle."
)

# Final ready-to-send note.
ready = []
ready.append("# FINAL HANDOFF READY")
ready.append("")
ready.append("The project is ready to send to the professor.")
ready.append("")
ready.append("## Main current result")
ready.append("")
ready.append("- Main model: Phase 8C FTV support-data SVM_RBF")
ready.append("- AUROC: 0.6667")
ready.append("- AUPRC: 0.7528")
ready.append("- Brier score: 0.2405")
ready.append("")
ready.append("## Final decision")
ready.append("")
ready.append("Use the FTV support-data model as the main pilot result. Keep temporal GNN as future work.")
ready.append("")
ready.append("## Send these files")
ready.append("")
ready.append("1. `reports/final_package/ISPY2_FTV_pCR_Final_Report.docx`")
ready.append("2. `reports/final_package/ISPY2_FTV_pCR_Final_Presentation.pptx`")
ready.append("3. `reports/final_package/ISPY2_FTV_pCR_Professor_Review_Bundle.zip`")
ready.append("4. `reports/final_package/Final_Send_Message_to_Professor.md`")
ready.append("")
ready.append("## Safe wording")
ready.append("")
ready.append("This is a pilot feasibility study, not clinical validation.")
ready.append("")
ready.append("## What to ask professor")
ready.append("")
ready.append("- Is the title acceptable?")
ready.append("- Is the main claim safe?")
ready.append("- Should temporal GNN stay as future work?")
ready.append("- Which parts should be expanded for thesis or publication?")
(FINAL / "FINAL_HANDOFF_READY.md").write_text("\n".join(ready))

# Check final files.
final_files = [
    "FINAL_STATUS.md",
    "reports/Stage10G_Final_Handoff_Fix_Report.md",
    "reports/Stage10H_Final_Repository_Freeze_Report.md",
    "reports/final_package/FINAL_HANDOFF_READY.md",
    "reports/final_package/ISPY2_FTV_pCR_Final_Report.docx",
    "reports/final_package/ISPY2_FTV_pCR_Final_Presentation.pptx",
    "reports/final_package/ISPY2_FTV_pCR_Professor_Review_Bundle.zip",
    "reports/final_package/Final_Send_Message_to_Professor.md",
    "reports/final_package/RELEASE_NOTES_v1.0-pilot-ftv-pcr.md",
]

tracked = set(run("git ls-files").splitlines())

rows = []
for f in final_files:
    path = REPO / f
    rows.append({
        "file": f,
        "exists": "yes" if path.exists() else "no",
        "tracked_by_git": "yes" if f in tracked else "no",
        "size_mb": round(path.stat().st_size / (1024 * 1024), 4) if path.exists() else "",
    })

qc = pd.DataFrame(rows)
qc.to_csv(TABLES / "stage10i_post_release_polish_qc.csv", index=False)

missing = int((qc["exists"] == "no").sum())
untracked = int((qc["tracked_by_git"] == "no").sum())

# Markdown view.
def make_md(csv_path, md_path, title):
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    def clean(x):
        x = str(x).replace("|", "\\|").replace("\n", " ")
        return x[:87] + "..." if len(x) > 90 else x
    lines = ["# " + title, "", f"Rows: {len(df)}", f"Columns: {len(df.columns)}", ""]
    lines.append("| " + " | ".join(clean(c) for c in df.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(clean(v) for v in r.tolist()) + " |")
    md_path.write_text("\n".join(lines))

make_md(
    TABLES / "stage10i_post_release_polish_qc.csv",
    VIEWS / "stage10i_post_release_polish_qc.md",
    "Stage 10I Post-Release Polish QC"
)

# Figures.
plt.figure(figsize=(9, 5))
qc["exists"].value_counts().sort_values().plot(kind="barh")
plt.title("Stage 10I Final File Existence")
plt.xlabel("Files")
plt.tight_layout()
plt.savefig(FIGS / "380_stage10i_file_existence.png", dpi=220)
plt.close()

plt.figure(figsize=(9, 5))
qc["tracked_by_git"].value_counts().sort_values().plot(kind="barh")
plt.title("Stage 10I Git Tracking")
plt.xlabel("Files")
plt.tight_layout()
plt.savefig(FIGS / "381_stage10i_git_tracking.png", dpi=220)
plt.close()

plt.figure(figsize=(11, 6))
plt.axis("off")
txt = (
    "Final project status\n\n"
    "Ready to send to professor\n\n"
    "Main result:\n"
    "FTV support-data SVM_RBF\n"
    "AUROC 0.6667 | AUPRC 0.7528 | Brier 0.2405\n\n"
    "Decision:\n"
    "FTV model is main pilot result.\n"
    "Temporal GNN is future work.\n\n"
    f"Missing files: {missing}\n"
    f"Untracked files: {untracked}"
)
plt.text(0.5, 0.5, txt, ha="center", va="center", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "382_stage10i_ready_to_send_summary.png", dpi=220)
plt.close()

# Stage 10I report.
report = []
report.append("# Stage 10I Post-Release Polish Report")
report.append("")
report.append("This stage fixes outdated post-release wording and creates a final ready-to-send note.")
report.append("")
report.append("## Main outputs")
report.append("")
report.append("- reports/final_package/FINAL_HANDOFF_READY.md")
report.append("- reports/tables/stage10i_post_release_polish_qc.csv")
report.append("- reports/table_views/stage10i_post_release_polish_qc.md")
report.append("- reports/figures/380_stage10i_file_existence.png")
report.append("- reports/figures/381_stage10i_git_tracking.png")
report.append("- reports/figures/382_stage10i_ready_to_send_summary.png")
report.append("")
report.append("## Updates")
report.append("")
report.append("- Stage 10G wording status: " + g_status)
report.append("- Stage 10H wording status: " + h_status)
report.append("")
report.append("## Final counts")
report.append("")
report.append("- Missing files: " + str(missing))
report.append("- Untracked files before this commit: " + str(untracked))
report.append("")
report.append("## Final status")
report.append("")
if missing == 0:
    report.append("The handoff package is ready. Send the Word report, PowerPoint, or ZIP bundle to the professor.")
else:
    report.append("Some files are missing. Check the QC table.")
report.append("")
report.append("## Next step")
report.append("")
report.append("Commit this polish step and push it. No new modeling code is needed before professor feedback.")

(REPORTS / "Stage10I_Post_Release_Polish_Report.md").write_text("\n".join(report))

# Update Dashboard and README.
dash = REPORTS / "Dashboard.md"
old = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
add = (
    "\n\n## Stage 10I post-release polish\n\n"
    "- [Post-release polish report](Stage10I_Post_Release_Polish_Report.md)\n"
    "- [Final handoff ready note](final_package/FINAL_HANDOFF_READY.md)\n"
    "- [Post-release polish QC](table_views/stage10i_post_release_polish_qc.md)\n"
)
if "Stage 10I post-release polish" not in old:
    dash.write_text(old.rstrip() + add)

readme = REPO / "README.md"
old = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
add = (
    "\n\n### Stage 10I: post-release polish\n\n"
    "Main outputs:\n\n"
    "- reports/Stage10I_Post_Release_Polish_Report.md\n"
    "- reports/final_package/FINAL_HANDOFF_READY.md\n"
    "- reports/tables/stage10i_post_release_polish_qc.csv\n"
)
if "Stage 10I: post-release polish" not in old:
    readme.write_text(old.rstrip() + add)

print("Stage 10I complete.")
print("Stage10G wording:", g_status)
print("Stage10H wording:", h_status)
print("Missing files:", missing)
print("Untracked files before commit:", untracked)
