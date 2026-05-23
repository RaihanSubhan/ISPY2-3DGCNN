from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
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
FINAL = REPORTS / "final_package"

for p in [REPORTS, TABLES, FIGS, VIEWS, FINAL]:
    p.mkdir(parents=True, exist_ok=True)

DOCX = FINAL / "ISPY2_FTV_pCR_Final_Report.docx"
PPTX = FINAL / "ISPY2_FTV_pCR_Final_Presentation.pptx"
SUMMARY = FINAL / "ISPY2_FTV_pCR_One_Page_Summary.md"

KEY_FILES = [
    DOCX,
    PPTX,
    SUMMARY,
    REPORTS / "Stage10E_Final_Word_PowerPoint_Package_Report.md",
    REPORTS / "Stage10D_Thesis_Methods_Figure_Package_Report.md",
    REPORTS / "Phase8D_FTV_vs_Baseline_Manuscript_Result.md",
    REPORTS / "Phase9C_Temporal_Graph_Model_Decision_Report.md",
    FIGS / "340_stage10d_master_methods_panel.png",
    FIGS / "342_stage10d_final_pipeline_and_model_result.png",
    FIGS / "260_phase8d_baseline_vs_ftv_metrics.png",
    FIGS / "300_phase9c_model_decision.png",
    FIGS / "283_phase9a_graph_schema.png",
    TABLES / "phase8d_baseline_vs_ftv_model_comparison.csv",
    TABLES / "phase9c_model_decision_summary.csv",
    TABLES / "stage10e_final_document_package_manifest.csv",
]

def file_info(path):
    path = Path(path)
    if not path.exists():
        return {
            "file": str(path.relative_to(REPO)) if str(path).startswith(str(REPO)) else str(path),
            "exists": "no",
            "size_mb": "",
            "status": "missing",
            "extra_check": ""
        }

    size_mb = path.stat().st_size / (1024 * 1024)
    return {
        "file": str(path.relative_to(REPO)) if str(path).startswith(str(REPO)) else str(path),
        "exists": "yes",
        "size_mb": round(size_mb, 4),
        "status": "ok",
        "extra_check": ""
    }

def docx_check(path):
    path = Path(path)
    if not path.exists():
        return "missing"

    try:
        with zipfile.ZipFile(path, "r") as z:
            names = z.namelist()
            if "word/document.xml" not in names:
                return "invalid_docx_missing_document_xml"

            xml = z.read("word/document.xml")
            root = ET.fromstring(xml)

            texts = []
            for node in root.iter():
                if node.tag.endswith("}t") and node.text:
                    texts.append(node.text)

            text = " ".join(texts)
            words = len(text.split())
            return f"valid_docx_words_{words}"
    except Exception as e:
        return "docx_check_failed_" + str(e)[:80]

def pptx_check(path):
    path = Path(path)
    if not path.exists():
        return "missing"

    try:
        with zipfile.ZipFile(path, "r") as z:
            names = z.namelist()
            slides = [n for n in names if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
            if len(slides) == 0:
                return "invalid_pptx_no_slides"
            return f"valid_pptx_slides_{len(slides)}"
    except Exception as e:
        return "pptx_check_failed_" + str(e)[:80]

# ---------------------------------------------------------------------
# 1. Final package QC table
# ---------------------------------------------------------------------
qc_rows = []

for f in KEY_FILES:
    row = file_info(f)
    if f == DOCX:
        row["extra_check"] = docx_check(f)
    elif f == PPTX:
        row["extra_check"] = pptx_check(f)
    qc_rows.append(row)

qc = pd.DataFrame(qc_rows)
qc.to_csv(TABLES / "stage10f_final_package_qc.csv", index=False)

# ---------------------------------------------------------------------
# 2. Professor review checklist
# ---------------------------------------------------------------------
checklist = pd.DataFrame([
    {
        "order": 1,
        "review_item": "Confirm title",
        "what_to_ask_professor": "Is this title acceptable: Pilot FTV-Based Prediction of pCR in I-SPY2 Breast MRI?",
        "why_it_matters": "The title must not overclaim deep learning or clinical validation.",
        "status": "to_review"
    },
    {
        "order": 2,
        "review_item": "Confirm main model",
        "what_to_ask_professor": "Should Phase 8C FTV SVM_RBF be the main pilot model?",
        "why_it_matters": "It is currently the strongest model result.",
        "status": "to_review"
    },
    {
        "order": 3,
        "review_item": "Confirm segmentation wording",
        "what_to_ask_professor": "Can I present DICOM SEG as a diagnostic negative result?",
        "why_it_matters": "The SEG masks were not reliable tumor-only masks.",
        "status": "to_review"
    },
    {
        "order": 4,
        "review_item": "Confirm obstacle-map language",
        "what_to_ask_professor": "Should I describe the obstacle map as conceptual contrast-transport disturbance only?",
        "why_it_matters": "It should not be described as real blood, milk, or fluid flow.",
        "status": "to_review"
    },
    {
        "order": 5,
        "review_item": "Confirm limitation",
        "what_to_ask_professor": "Is the limitation statement strong enough: only 13 labeled FTV patients, pilot only?",
        "why_it_matters": "This protects the study from overclaiming.",
        "status": "to_review"
    },
    {
        "order": 6,
        "review_item": "Confirm future work",
        "what_to_ask_professor": "Should temporal GNN be placed as future work instead of current main model?",
        "why_it_matters": "Phase 9C says GNN is not justified as main model yet.",
        "status": "to_review"
    },
    {
        "order": 7,
        "review_item": "Confirm figures",
        "what_to_ask_professor": "Are Figures 340, 342, 260, 300, and 283 enough for presentation?",
        "why_it_matters": "These figures tell the final story cleanly.",
        "status": "to_review"
    }
])

checklist.to_csv(TABLES / "stage10f_professor_review_checklist.csv", index=False)

# ---------------------------------------------------------------------
# 3. Talk track for meeting
# ---------------------------------------------------------------------
talk = []
talk.append("# Stage 10F Professor Meeting Talk Track")
talk.append("")
talk.append("## 60-second summary")
talk.append("")
talk.append("I built a full ISPY2 breast MRI pipeline on Cradle and saved it in GitHub. I started with raw DICOM and DICOM SEG files, but I found that the SEG masks were not reliable tumor-only masks. Because of that, I moved to official MRI/FTV support-data features. That path gave the strongest pilot pCR prediction result. The best current model is SVM_RBF using FTV support features, with AUROC 0.6667, AUPRC 0.7528, and Brier score 0.2405. I also built a graph-ready longitudinal FTV dataset, but the temporal-delta model was weaker than the FTV model. So I am keeping temporal GNN as future work, not the current main model.")
talk.append("")
talk.append("## Questions for professor")
talk.append("")
for _, r in checklist.iterrows():
    talk.append(f"{r['order']}. {r['what_to_ask_professor']}")
talk.append("")
talk.append("## Safe final claim")
talk.append("")
talk.append("Official MRI/FTV support-data features produced a stronger pilot pCR prediction signal than the earlier proxy and temporal-delta approaches.")
talk.append("")
talk.append("## Claim to avoid")
talk.append("")
talk.append("This model is clinically validated or ready for clinical use.")

(FINAL / "ISPY2_FTV_pCR_Professor_Talk_Track.md").write_text("\n".join(talk))

# ---------------------------------------------------------------------
# 4. GitHub final index
# ---------------------------------------------------------------------
index = []
index.append("# Final GitHub Output Index")
index.append("")
index.append("## Final report and slides")
index.append("")
index.append("- `reports/final_package/ISPY2_FTV_pCR_Final_Report.docx`")
index.append("- `reports/final_package/ISPY2_FTV_pCR_Final_Presentation.pptx`")
index.append("- `reports/final_package/ISPY2_FTV_pCR_One_Page_Summary.md`")
index.append("- `reports/final_package/ISPY2_FTV_pCR_Professor_Talk_Track.md`")
index.append("")
index.append("## Most important figures")
index.append("")
index.append("- `reports/figures/340_stage10d_master_methods_panel.png`")
index.append("- `reports/figures/342_stage10d_final_pipeline_and_model_result.png`")
index.append("- `reports/figures/260_phase8d_baseline_vs_ftv_metrics.png`")
index.append("- `reports/figures/300_phase9c_model_decision.png`")
index.append("- `reports/figures/283_phase9a_graph_schema.png`")
index.append("")
index.append("## Most important tables")
index.append("")
index.append("- `reports/tables/phase8d_baseline_vs_ftv_model_comparison.csv`")
index.append("- `reports/tables/phase8c_ftv_model_performance.csv`")
index.append("- `reports/tables/phase9c_model_decision_summary.csv`")
index.append("- `reports/tables/stage10f_professor_review_checklist.csv`")
index.append("")
index.append("## Final decision")
index.append("")
index.append("The main current result is the Phase 8C FTV support-data SVM_RBF model. Temporal GNN is future work.")

(REPORTS / "Final_GitHub_Output_Index.md").write_text("\n".join(index))

# ---------------------------------------------------------------------
# 5. Figures for Stage 10F
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))
qc["status"].value_counts().sort_values().plot(kind="barh")
plt.title("Stage 10F Final Package QC Status")
plt.xlabel("Files")
plt.tight_layout()
plt.savefig(FIGS / "350_stage10f_package_qc_status.png", dpi=220)
plt.close()

plt.figure(figsize=(11, 6))
plt.axis("off")
story = (
    "Final project story\n\n"
    "Raw ISPY2 DICOM on Cradle\n"
    "↓\n"
    "DICOM inventory and SEG diagnosis\n"
    "↓\n"
    "SEG path found weak for tumor-only modeling\n"
    "↓\n"
    "Official MRI / FTV support-data features\n"
    "↓\n"
    "Best pilot model: SVM_RBF\n"
    "AUROC 0.6667, AUPRC 0.7528, Brier 0.2405\n"
    "↓\n"
    "Temporal GNN remains future work"
)
plt.text(0.5, 0.5, story, ha="center", va="center", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "351_stage10f_final_story_flow.png", dpi=220)
plt.close()

plt.figure(figsize=(11, 6))
plt.axis("off")
agenda = (
    "Professor meeting agenda\n\n"
    "1. Confirm title and research direction\n"
    "2. Confirm FTV SVM as main pilot result\n"
    "3. Confirm DICOM SEG discussion as diagnostic negative result\n"
    "4. Confirm obstacle map wording\n"
    "5. Confirm limitation: pilot only, 13 labeled FTV patients\n"
    "6. Confirm temporal GNN as future work\n"
    "7. Revise Word report and slides"
)
plt.text(0.5, 0.5, agenda, ha="center", va="center", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "352_stage10f_professor_meeting_agenda.png", dpi=220)
plt.close()

# ---------------------------------------------------------------------
# 6. Markdown table helper
# ---------------------------------------------------------------------
def make_md(csv_path, md_path, title, max_rows=80, max_cols=10):
    data = pd.read_csv(csv_path, dtype=str).fillna("")
    small = data.head(max_rows).iloc[:, :max_cols]

    def clean(x):
        x = str(x).replace("|", "\\|").replace("\n", " ")
        return x[:87] + "..." if len(x) > 90 else x

    lines = []
    lines.append("# " + title)
    lines.append("")
    lines.append("Rows: " + str(len(data)))
    lines.append("Columns: " + str(len(data.columns)))
    lines.append("")
    lines.append("| " + " | ".join(clean(c) for c in small.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(small.columns)) + " |")

    for _, row in small.iterrows():
        lines.append("| " + " | ".join(clean(v) for v in row.tolist()) + " |")

    md_path.write_text("\n".join(lines))

make_md(
    TABLES / "stage10f_final_package_qc.csv",
    VIEWS / "stage10f_final_package_qc.md",
    "Stage 10F Final Package QC"
)

make_md(
    TABLES / "stage10f_professor_review_checklist.csv",
    VIEWS / "stage10f_professor_review_checklist.md",
    "Stage 10F Professor Review Checklist"
)

# ---------------------------------------------------------------------
# 7. Create professor review bundle zip
# ---------------------------------------------------------------------
bundle_path = FINAL / "ISPY2_FTV_pCR_Professor_Review_Bundle.zip"

bundle_files = [
    DOCX,
    PPTX,
    SUMMARY,
    FINAL / "ISPY2_FTV_pCR_Professor_Talk_Track.md",
    REPORTS / "Final_GitHub_Output_Index.md",
    REPORTS / "Stage10E_Final_Word_PowerPoint_Package_Report.md",
    FIGS / "340_stage10d_master_methods_panel.png",
    FIGS / "342_stage10d_final_pipeline_and_model_result.png",
    FIGS / "260_phase8d_baseline_vs_ftv_metrics.png",
    FIGS / "300_phase9c_model_decision.png",
    FIGS / "283_phase9a_graph_schema.png",
    FIGS / "350_stage10f_package_qc_status.png",
    FIGS / "351_stage10f_final_story_flow.png",
    FIGS / "352_stage10f_professor_meeting_agenda.png",
    TABLES / "stage10f_final_package_qc.csv",
    TABLES / "stage10f_professor_review_checklist.csv",
]

with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for f in bundle_files:
        f = Path(f)
        if f.exists():
            z.write(f, arcname=str(f.relative_to(REPO)))

# ---------------------------------------------------------------------
# 8. Final Stage 10F report
# ---------------------------------------------------------------------
missing_count = int((qc["exists"] == "no").sum())
ok_count = int((qc["exists"] == "yes").sum())

report = []
report.append("# Stage 10F Final Package Quality Check and Professor Review Report")
report.append("")
report.append("This stage validates the final Word and PowerPoint package and creates a professor-review bundle.")
report.append("")
report.append("## Main outputs")
report.append("")
report.append("- reports/tables/stage10f_final_package_qc.csv")
report.append("- reports/tables/stage10f_professor_review_checklist.csv")
report.append("- reports/table_views/stage10f_final_package_qc.md")
report.append("- reports/table_views/stage10f_professor_review_checklist.md")
report.append("- reports/final_package/ISPY2_FTV_pCR_Professor_Talk_Track.md")
report.append("- reports/final_package/ISPY2_FTV_pCR_Professor_Review_Bundle.zip")
report.append("- reports/Final_GitHub_Output_Index.md")
report.append("- reports/figures/350_stage10f_package_qc_status.png")
report.append("- reports/figures/351_stage10f_final_story_flow.png")
report.append("- reports/figures/352_stage10f_professor_meeting_agenda.png")
report.append("")
report.append("## Main counts")
report.append("")
report.append("- Files checked: " + str(len(qc)))
report.append("- Files present: " + str(ok_count))
report.append("- Files missing: " + str(missing_count))
report.append("")
report.append("## Final interpretation")
report.append("")
if missing_count == 0:
    report.append("The final package is ready for professor review.")
else:
    report.append("Some files are missing. Check the QC table before sending.")
report.append("")
report.append("## Next step")
report.append("")
report.append("Send the Word report and PowerPoint to your professor. Use the professor checklist to guide the meeting.")

(REPORTS / "Stage10F_Final_Package_QC_Professor_Review_Report.md").write_text("\n".join(report))

# Dashboard/README
dash = REPORTS / "Dashboard.md"
old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
dash_add = (
    "\n\n## Stage 10F final package QC and professor review\n\n"
    "- [Final package QC report](Stage10F_Final_Package_QC_Professor_Review_Report.md)\n"
    "- [Final GitHub output index](Final_GitHub_Output_Index.md)\n"
    "- [Professor review checklist](table_views/stage10f_professor_review_checklist.md)\n"
)
if "Stage 10F final package QC and professor review" not in old_dash:
    dash.write_text(old_dash.rstrip() + dash_add)

readme = REPO / "README.md"
old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
readme_add = (
    "\n\n### Stage 10F: final package QC and professor review\n\n"
    "Main outputs:\n\n"
    "- reports/Stage10F_Final_Package_QC_Professor_Review_Report.md\n"
    "- reports/Final_GitHub_Output_Index.md\n"
    "- reports/final_package/ISPY2_FTV_pCR_Professor_Review_Bundle.zip\n"
)
if "Stage 10F: final package QC and professor review" not in old_readme:
    readme.write_text(old_readme.rstrip() + readme_add)

print("Stage 10F complete.")
print("Files checked:", len(qc))
print("Files present:", ok_count)
print("Files missing:", missing_count)
print("Review bundle:", bundle_path)
