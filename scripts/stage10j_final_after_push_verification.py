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
            cmd,
            shell=True,
            cwd=REPO,
            text=True,
            stderr=subprocess.STDOUT
        ).strip()
    except subprocess.CalledProcessError as e:
        return e.output.strip()

final_files = [
    "FINAL_STATUS.md",
    "README.md",
    "reports/Dashboard.md",
    "reports/Final_GitHub_Output_Index.md",
    "reports/Stage10I_Post_Release_Polish_Report.md",
    "reports/final_package/FINAL_HANDOFF_READY.md",
    "reports/final_package/ISPY2_FTV_pCR_Final_Report.docx",
    "reports/final_package/ISPY2_FTV_pCR_Final_Presentation.pptx",
    "reports/final_package/ISPY2_FTV_pCR_Professor_Review_Bundle.zip",
    "reports/final_package/Final_Send_Message_to_Professor.md",
    "reports/final_package/RELEASE_NOTES_v1.0-pilot-ftv-pcr.md",
    "reports/figures/340_stage10d_master_methods_panel.png",
    "reports/figures/342_stage10d_final_pipeline_and_model_result.png",
    "reports/figures/260_phase8d_baseline_vs_ftv_metrics.png",
    "reports/figures/300_phase9c_model_decision.png",
    "reports/tables/phase8c_ftv_model_performance.csv",
    "reports/tables/phase8d_baseline_vs_ftv_model_comparison.csv",
    "reports/tables/phase9c_model_decision_summary.csv",
    "reports/tables/stage10i_post_release_polish_qc.csv",
]

tracked = set(run("git ls-files").splitlines())
git_status = run("git status --porcelain")
current_commit = run("git rev-parse HEAD")
current_branch = run("git branch --show-current")
remote_commit = run("git rev-parse origin/main")
old_tag_commit = run("git rev-list -n 1 v1.0-pilot-ftv-pcr 2>/dev/null || true")

rows = []
for f in final_files:
    path = REPO / f
    rows.append({
        "file": f,
        "exists": "yes" if path.exists() else "no",
        "tracked_by_git_after_push": "yes" if f in tracked else "no",
        "size_mb": round(path.stat().st_size / (1024 * 1024), 4) if path.exists() else "",
    })

qc = pd.DataFrame(rows)
qc.to_csv(TABLES / "stage10j_final_after_push_qc.csv", index=False)

missing = int((qc["exists"] == "no").sum())
untracked = int((qc["tracked_by_git_after_push"] == "no").sum())
branch_synced = "yes" if current_commit == remote_commit else "no"
working_tree_clean = "yes" if git_status == "" else "no"
old_tag_on_head = "yes" if old_tag_commit == current_commit else "no"

# Update the old Stage 10I QC table if it exists, so GitHub no longer shows stale tracked state.
stage10i_qc = TABLES / "stage10i_post_release_polish_qc.csv"
if stage10i_qc.exists():
    old = pd.read_csv(stage10i_qc, dtype=str).fillna("")
    if "file" in old.columns:
        old["tracked_by_git"] = old["file"].map(lambda x: "yes" if x in tracked else "no")
        old["exists"] = old["file"].map(lambda x: "yes" if (REPO / x).exists() else "no")
        old["size_mb"] = old["file"].map(
            lambda x: round((REPO / x).stat().st_size / (1024 * 1024), 4) if (REPO / x).exists() else ""
        )
        old.to_csv(stage10i_qc, index=False)

# Final release status table.
release_status = pd.DataFrame([{
    "current_branch": current_branch,
    "current_commit": current_commit,
    "origin_main_commit": remote_commit,
    "branch_synced_with_origin_main": branch_synced,
    "working_tree_clean_before_stage10j_commit": working_tree_clean,
    "old_tag_v1_0_pilot_ftv_pcr_commit": old_tag_commit,
    "old_tag_points_to_current_head": old_tag_on_head,
    "recommended_new_tag": "v1.0.1-final-handoff"
}])
release_status.to_csv(TABLES / "stage10j_release_status.csv", index=False)

# Markdown view helper.
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

make_md(TABLES / "stage10j_final_after_push_qc.csv", VIEWS / "stage10j_final_after_push_qc.md", "Stage 10J Final After-Push QC")
make_md(TABLES / "stage10j_release_status.csv", VIEWS / "stage10j_release_status.md", "Stage 10J Release Status")

# Figures.
plt.figure(figsize=(9, 5))
qc["tracked_by_git_after_push"].value_counts().sort_values().plot(kind="barh")
plt.title("Stage 10J Final Git Tracking After Push")
plt.xlabel("Files")
plt.tight_layout()
plt.savefig(FIGS / "390_stage10j_git_tracking_after_push.png", dpi=220)
plt.close()

plt.figure(figsize=(9, 5))
qc["exists"].value_counts().sort_values().plot(kind="barh")
plt.title("Stage 10J Final File Existence")
plt.xlabel("Files")
plt.tight_layout()
plt.savefig(FIGS / "391_stage10j_file_existence.png", dpi=220)
plt.close()

plt.figure(figsize=(11, 6))
plt.axis("off")
txt = (
    "Stage 10J final verification\n\n"
    f"Branch: {current_branch}\n"
    f"Working tree clean before commit: {working_tree_clean}\n"
    f"Synced with origin/main: {branch_synced}\n"
    f"Missing final files: {missing}\n"
    f"Untracked final files: {untracked}\n\n"
    "Main result:\n"
    "Phase 8C FTV support-data SVM_RBF\n"
    "AUROC 0.6667 | AUPRC 0.7528 | Brier 0.2405\n\n"
    "Final action:\n"
    "Create tag v1.0.1-final-handoff after this commit."
)
plt.text(0.5, 0.5, txt, ha="center", va="center", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "392_stage10j_final_verification_summary.png", dpi=220)
plt.close()

# Final report.
report = []
report.append("# Stage 10J Final After-Push Verification Report")
report.append("")
report.append("This stage verifies the final package after Stage 10I was pushed.")
report.append("")
report.append("## Main outputs")
report.append("")
report.append("- reports/tables/stage10j_final_after_push_qc.csv")
report.append("- reports/tables/stage10j_release_status.csv")
report.append("- reports/table_views/stage10j_final_after_push_qc.md")
report.append("- reports/table_views/stage10j_release_status.md")
report.append("- reports/figures/390_stage10j_git_tracking_after_push.png")
report.append("- reports/figures/391_stage10j_file_existence.png")
report.append("- reports/figures/392_stage10j_final_verification_summary.png")
report.append("")
report.append("## Final counts")
report.append("")
report.append("- Missing final files: " + str(missing))
report.append("- Untracked final files after push: " + str(untracked))
report.append("- Working tree clean before Stage 10J commit: " + working_tree_clean)
report.append("- Synced with origin/main before Stage 10J commit: " + branch_synced)
report.append("")
report.append("## Tag status")
report.append("")
if old_tag_on_head == "yes":
    report.append("The old tag `v1.0-pilot-ftv-pcr` points to the current HEAD.")
else:
    report.append("The old tag `v1.0-pilot-ftv-pcr` does not point to the latest HEAD. This is okay because it froze Stage 10H. Create a new tag `v1.0.1-final-handoff` after this commit.")
report.append("")
report.append("## Final status")
report.append("")
if missing == 0 and untracked == 0:
    report.append("The final package is ready. No more modeling code is needed before professor feedback.")
else:
    report.append("Some files are missing or untracked. Check the QC table.")
report.append("")
report.append("## Next step")
report.append("")
report.append("Commit Stage 10J, push it, and create the tag `v1.0.1-final-handoff`.")

(REPORTS / "Stage10J_Final_After_Push_Verification_Report.md").write_text("\n".join(report))

# Update final handoff ready note.
ready = FINAL / "FINAL_HANDOFF_READY.md"
if ready.exists():
    text = ready.read_text()
    add = (
        "\n\n## Final verification\n\n"
        "Stage 10J verifies that the final package is present and Git-tracked after push. "
        "Use tag `v1.0.1-final-handoff` as the final professor-handoff version.\n"
    )
    if "Stage 10J verifies" not in text:
        ready.write_text(text.rstrip() + add)

# Dashboard/README.
dash = REPORTS / "Dashboard.md"
old = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
add = (
    "\n\n## Stage 10J final after-push verification\n\n"
    "- [Final after-push verification report](Stage10J_Final_After_Push_Verification_Report.md)\n"
    "- [Final after-push QC](table_views/stage10j_final_after_push_qc.md)\n"
    "- [Release status](table_views/stage10j_release_status.md)\n"
)
if "Stage 10J final after-push verification" not in old:
    dash.write_text(old.rstrip() + add)

readme = REPO / "README.md"
old = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
add = (
    "\n\n### Stage 10J: final after-push verification\n\n"
    "Main outputs:\n\n"
    "- reports/Stage10J_Final_After_Push_Verification_Report.md\n"
    "- reports/tables/stage10j_final_after_push_qc.csv\n"
    "- reports/tables/stage10j_release_status.csv\n"
)
if "Stage 10J: final after-push verification" not in old:
    readme.write_text(old.rstrip() + add)

print("Stage 10J complete.")
print("Missing final files:", missing)
print("Untracked final files:", untracked)
print("Working tree clean before commit:", working_tree_clean)
print("Synced with origin/main before commit:", branch_synced)
print("Old tag points to current HEAD:", old_tag_on_head)
print("Recommended new tag: v1.0.1-final-handoff")
