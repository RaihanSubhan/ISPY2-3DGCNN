from pathlib import Path
import zipfile
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

for p in [FINAL, TABLES, FIGS, VIEWS]:
    p.mkdir(parents=True, exist_ok=True)

DOCX = FINAL / "ISPY2_FTV_pCR_Final_Report.docx"
PPTX = FINAL / "ISPY2_FTV_pCR_Final_Presentation.pptx"
SUMMARY = FINAL / "ISPY2_FTV_pCR_One_Page_Summary.md"
TALK = FINAL / "ISPY2_FTV_pCR_Professor_Talk_Track.md"
BUNDLE = FINAL / "ISPY2_FTV_pCR_Professor_Review_Bundle.zip"

IMPORTANT_FILES = [
    DOCX,
    PPTX,
    SUMMARY,
    TALK,
    REPORTS / "Final_GitHub_Output_Index.md",
    REPORTS / "Stage10F_Final_Package_QC_Professor_Review_Report.md",
    REPORTS / "Phase8D_FTV_vs_Baseline_Manuscript_Result.md",
    REPORTS / "Phase9C_Temporal_Graph_Model_Decision_Report.md",
    FIGS / "340_stage10d_master_methods_panel.png",
    FIGS / "342_stage10d_final_pipeline_and_model_result.png",
    FIGS / "260_phase8d_baseline_vs_ftv_metrics.png",
    FIGS / "300_phase9c_model_decision.png",
    FIGS / "283_phase9a_graph_schema.png",
    FIGS / "350_stage10f_package_qc_status.png",
    FIGS / "351_stage10f_final_story_flow.png",
    FIGS / "352_stage10f_professor_meeting_agenda.png",
    TABLES / "phase8d_baseline_vs_ftv_model_comparison.csv",
    TABLES / "phase8c_ftv_model_performance.csv",
    TABLES / "phase9c_model_decision_summary.csv",
    TABLES / "stage10f_professor_review_checklist.csv",
]

# Rebuild ZIP bundle.
with zipfile.ZipFile(BUNDLE, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for f in IMPORTANT_FILES:
        f = Path(f)
        if f.exists():
            z.write(f, arcname=str(f.relative_to(REPO)))

# Final QC.
rows = []
for f in IMPORTANT_FILES + [BUNDLE]:
    f = Path(f)
    rows.append({
        "file": str(f.relative_to(REPO)) if f.exists() else str(f.relative_to(REPO)),
        "exists": "yes" if f.exists() else "no",
        "size_mb": round(f.stat().st_size / (1024 * 1024), 4) if f.exists() else "",
        "purpose": "professor review package" if f == BUNDLE else "supporting final output"
    })

qc = pd.DataFrame(rows)
qc.to_csv(TABLES / "stage10g_final_handoff_qc.csv", index=False)

# Final package manifest.
manifest = pd.DataFrame([
    {
        "file": "reports/final_package/ISPY2_FTV_pCR_Final_Report.docx",
        "type": "Word report",
        "send_to_professor": "yes",
        "note": "Main report draft"
    },
    {
        "file": "reports/final_package/ISPY2_FTV_pCR_Final_Presentation.pptx",
        "type": "PowerPoint",
        "send_to_professor": "yes",
        "note": "Presentation slide deck"
    },
    {
        "file": "reports/final_package/ISPY2_FTV_pCR_One_Page_Summary.md",
        "type": "short summary",
        "send_to_professor": "optional",
        "note": "Short project summary"
    },
    {
        "file": "reports/final_package/ISPY2_FTV_pCR_Professor_Talk_Track.md",
        "type": "meeting script",
        "send_to_professor": "no",
        "note": "Use for your own meeting preparation"
    },
    {
        "file": "reports/final_package/ISPY2_FTV_pCR_Professor_Review_Bundle.zip",
        "type": "ZIP bundle",
        "send_to_professor": "optional",
        "note": "Contains report, slides, figures, summary, and checklist"
    },
])
manifest.to_csv(TABLES / "stage10g_professor_handoff_manifest.csv", index=False)

# Professor email draft.
email = []
email.append("# Professor Email Draft")
email.append("")
email.append("Subject: ISPY2 breast MRI pilot results and proposal draft")
email.append("")
email.append("Dear Professor,")
email.append("")
email.append("I have completed the first full research cycle for my ISPY2 breast MRI project. The main current result is a pilot pCR prediction model using official MRI/FTV support-data features. The best model is SVM_RBF with AUROC 0.6667, AUPRC 0.7528, and Brier score 0.2405.")
email.append("")
email.append("During the work, I also tested the raw DICOM SEG mask path and found that it was not reliable enough as the main tumor-source path. Many masks behaved like full-slice analysis masks. Because of that, I moved to the official FTV/MRI support-data features as the main modeling source.")
email.append("")
email.append("I also built a graph-ready longitudinal FTV dataset and tested a temporal-delta baseline. That model was weaker than the FTV support-data model, so I am treating temporal GNN as future work rather than the current main model.")
email.append("")
email.append("I am attaching the Word report and PowerPoint. I would like your feedback on the title, the main claim, the limitation wording, and whether the future temporal graph direction is appropriate.")
email.append("")
email.append("Best regards,")
email.append("Raihan")
email.append("")
(FINAL / "Professor_Email_Draft.md").write_text("\n".join(email))

# Final checklist table.
checklist = pd.DataFrame([
    {"item": "Word report exists", "status": "yes" if DOCX.exists() else "no"},
    {"item": "PowerPoint exists", "status": "yes" if PPTX.exists() else "no"},
    {"item": "One-page summary exists", "status": "yes" if SUMMARY.exists() else "no"},
    {"item": "Talk track exists", "status": "yes" if TALK.exists() else "no"},
    {"item": "Professor review ZIP exists", "status": "yes" if BUNDLE.exists() else "no"},
    {"item": "Safe main claim stated", "status": "yes"},
    {"item": "Clinical validation overclaim avoided", "status": "yes"},
    {"item": "Temporal GNN kept as future work", "status": "yes"},
    {"item": "Obstacle map described as conceptual proxy only", "status": "yes"},
])
checklist.to_csv(TABLES / "stage10g_final_send_checklist.csv", index=False)

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

make_md(TABLES / "stage10g_final_handoff_qc.csv", VIEWS / "stage10g_final_handoff_qc.md", "Stage 10G Final Handoff QC")
make_md(TABLES / "stage10g_professor_handoff_manifest.csv", VIEWS / "stage10g_professor_handoff_manifest.md", "Stage 10G Professor Handoff Manifest")
make_md(TABLES / "stage10g_final_send_checklist.csv", VIEWS / "stage10g_final_send_checklist.md", "Stage 10G Final Send Checklist")

# Figure.
plt.figure(figsize=(9, 5))
qc["exists"].value_counts().sort_values().plot(kind="barh")
plt.title("Stage 10G Final Handoff QC")
plt.xlabel("Files")
plt.tight_layout()
plt.savefig(FIGS / "360_stage10g_final_handoff_qc.png", dpi=220)
plt.close()

plt.figure(figsize=(11, 6))
plt.axis("off")
txt = (
    "Final handoff package\n\n"
    "Send to professor:\n"
    "1. Word report\n"
    "2. PowerPoint slides\n"
    "3. Optional ZIP review bundle\n\n"
    "Main claim:\n"
    "Official MRI/FTV support-data features gave the strongest pilot pCR signal.\n\n"
    "Important limitation:\n"
    "Pilot only, not clinical validation.\n\n"
    "Future work:\n"
    "Temporal GNN after larger longitudinal FTV/PE/SER cohort."
)
plt.text(0.5, 0.5, txt, ha="center", va="center", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "361_stage10g_handoff_summary.png", dpi=220)
plt.close()

# Report.
missing = int((qc["exists"] == "no").sum())
present = int((qc["exists"] == "yes").sum())

report = []
report.append("# Stage 10G Final Handoff Fix Report")
report.append("")
report.append("This stage fixes the final handoff package by rebuilding the professor review ZIP bundle and preparing the final email/checklist.")
report.append("")
report.append("## Main outputs")
report.append("")
report.append("- reports/final_package/ISPY2_FTV_pCR_Professor_Review_Bundle.zip")
report.append("- reports/final_package/Professor_Email_Draft.md")
report.append("- reports/tables/stage10g_final_handoff_qc.csv")
report.append("- reports/tables/stage10g_professor_handoff_manifest.csv")
report.append("- reports/tables/stage10g_final_send_checklist.csv")
report.append("- reports/figures/360_stage10g_final_handoff_qc.png")
report.append("- reports/figures/361_stage10g_handoff_summary.png")
report.append("")
report.append("## Main counts")
report.append("")
report.append("- Files present: " + str(present))
report.append("- Files missing: " + str(missing))
report.append("")
report.append("## Final status")
report.append("")
if missing == 0:
    report.append("The professor handoff package is ready.")
else:
    report.append("Some files are missing. Check the QC table before sending.")
report.append("")
report.append("## Next step")
report.append("")
report.append("Force-add the ZIP bundle because it is ignored by .gitignore, then push to GitHub.")

(REPORTS / "Stage10G_Final_Handoff_Fix_Report.md").write_text("\n".join(report))

# Update final index.
index = REPORTS / "Final_GitHub_Output_Index.md"
old = index.read_text() if index.exists() else "# Final GitHub Output Index\n"
add = (
    "\n\n## Final professor handoff files\n\n"
    "- `reports/final_package/ISPY2_FTV_pCR_Professor_Review_Bundle.zip`\n"
    "- `reports/final_package/Professor_Email_Draft.md`\n"
    "- `reports/Stage10G_Final_Handoff_Fix_Report.md`\n"
    "- `reports/tables/stage10g_professor_handoff_manifest.csv`\n"
)
if "Final professor handoff files" not in old:
    index.write_text(old.rstrip() + add)

# README and dashboard.
dash = REPORTS / "Dashboard.md"
old = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
add = (
    "\n\n## Stage 10G final professor handoff\n\n"
    "- [Final handoff fix report](Stage10G_Final_Handoff_Fix_Report.md)\n"
    "- [Professor handoff manifest](table_views/stage10g_professor_handoff_manifest.md)\n"
    "- `reports/final_package/ISPY2_FTV_pCR_Professor_Review_Bundle.zip`\n"
)
if "Stage 10G final professor handoff" not in old:
    dash.write_text(old.rstrip() + add)

readme = REPO / "README.md"
old = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
add = (
    "\n\n### Stage 10G: final professor handoff fix\n\n"
    "Main outputs:\n\n"
    "- reports/Stage10G_Final_Handoff_Fix_Report.md\n"
    "- reports/final_package/ISPY2_FTV_pCR_Professor_Review_Bundle.zip\n"
    "- reports/final_package/Professor_Email_Draft.md\n"
)
if "Stage 10G: final professor handoff fix" not in old:
    readme.write_text(old.rstrip() + add)

print("Stage 10G complete.")
print("Files present:", present)
print("Files missing:", missing)
print("ZIP bundle:", BUNDLE)
