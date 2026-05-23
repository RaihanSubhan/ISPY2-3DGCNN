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
FINAL = REPORTS / "final_package"

for p in [REPORTS, TABLES, FIGS, VIEWS, FINAL]:
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Robust helpers
# ----------------------------
def read_csv_safe(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()


def best_model_from_file(path, default):
    df = read_csv_safe(path)

    if not df.empty and "AUROC" in df.columns:
        d = df.copy()
        for c in ["AUROC", "AUPRC", "Brier_score"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["AUROC"])
        if not d.empty:
            d = d.sort_values(["AUROC", "AUPRC", "Brier_score"], ascending=[False, False, True])
            r = d.iloc[0].to_dict()
            return {
                "analysis": default["analysis"],
                "best_model": str(r.get("model", default["best_model"])),
                "n_patients": int(float(r.get("n_patients", default["n_patients"]))),
                "AUROC": float(r.get("AUROC", default["AUROC"])),
                "AUPRC": float(r.get("AUPRC", default["AUPRC"])),
                "Brier_score": float(r.get("Brier_score", default["Brier_score"])),
            }

    return default


phase8c = best_model_from_file(
    TABLES / "phase8c_ftv_model_performance.csv",
    {
        "analysis": "Phase 8C FTV support-data model",
        "best_model": "SVM_RBF",
        "n_patients": 13,
        "AUROC": 0.6666666667,
        "AUPRC": 0.7527734171,
        "Brier_score": 0.2405088067,
    }
)

phase9b = best_model_from_file(
    TABLES / "phase9b_temporal_delta_model_performance.csv",
    {
        "analysis": "Phase 9B temporal-delta baseline",
        "best_model": "RandomForest",
        "n_patients": 13,
        "AUROC": 0.5714285714,
        "AUPRC": 0.6059523810,
        "Brier_score": 0.2724052885,
    }
)

stage12b = best_model_from_file(
    TABLES / "stage12b_3dgcnn_model_performance.csv",
    {
        "analysis": "Stage 12B exploratory temporal 3DGCNN",
        "best_model": "Exploratory_Temporal_3DGCNN",
        "n_patients": 13,
        "AUROC": 0.5714285714,
        "AUPRC": 0.5897186147,
        "Brier_score": 0.3909146815,
    }
)

ranking = pd.DataFrame([phase8c, phase9b, stage12b])
ranking = ranking.sort_values(["AUROC", "AUPRC", "Brier_score"], ascending=[False, False, True]).reset_index(drop=True)
ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
ranking.to_csv(TABLES / "stage12d_final_model_ranking.csv", index=False, lineterminator="\n")


# ----------------------------
# Paper idea tables
# ----------------------------
paper_alignment = pd.DataFrame([
    {
        "paper_inspiration": "3D MRI / 2D-to-3D reconstruction paper",
        "core_lesson": "A 3D representation should be justified by anatomy, spatial consistency, or longitudinal structure.",
        "how_it_shapes_this_project": "Use 3D/graph learning only after reliable tumor-source and longitudinal visit features are built.",
        "what_to_claim": "3DGCNN is a future-ready framework, not the current strongest model."
    },
    {
        "paper_inspiration": "MRI-to-3D anatomical modeling / printing paper",
        "core_lesson": "3D models must be spatially meaningful and validated before clinical interpretation.",
        "how_it_shapes_this_project": "Do not treat weak DICOM SEG masks as valid tumor geometry without checking them.",
        "what_to_claim": "Tumor-source validation is a core contribution."
    },
    {
        "paper_inspiration": "3D MRI segmentation / model validation paper",
        "core_lesson": "Serious 3D medical imaging work should report segmentation accuracy, surface distance, and validation when claiming true anatomy.",
        "how_it_shapes_this_project": "Since validated 3D tumor masks are limited, keep FTV features as main result and 3DGCNN as exploratory.",
        "what_to_claim": "The current study is pilot feasibility, not clinical validation."
    }
])
paper_alignment.to_csv(TABLES / "stage12d_three_paper_alignment.csv", index=False, lineterminator="\n")


novelty = pd.DataFrame([
    {
        "novel_component": "Tumor-source validation before modeling",
        "description": "The study first diagnoses DICOM SEG masks and shows that weak tumor sources can mislead downstream ML.",
        "why_novel": "Many studies start modeling directly. This work makes tumor-source reliability a formal part of the pipeline.",
        "paper_section": "Methods and Results"
    },
    {
        "novel_component": "Official MRI/FTV support-data pathway",
        "description": "The project moves from unreliable recovered masks to official MRI/FTV support features for pCR prediction.",
        "why_novel": "It demonstrates a safer tumor-feature source for pilot pCR modeling in ISPY2.",
        "paper_section": "Main Results"
    },
    {
        "novel_component": "Exploratory temporal 3DGCNN framework",
        "description": "The project builds patient graphs where visits are nodes and temporal edges connect treatment time points.",
        "why_novel": "It creates a bridge between conventional FTV modeling and future temporal graph learning.",
        "paper_section": "Future Work / Prototype"
    },
    {
        "novel_component": "3D tumor and transport visualization package",
        "description": "The project creates breast-pair, mask-derived 3D tumor shape, vessel-schematic, and multivisit tumor growth visuals.",
        "why_novel": "It helps explain the biological and modeling intuition while staying careful about what is measured.",
        "paper_section": "Visual Methods"
    }
])
novelty.to_csv(TABLES / "stage12d_novel_contribution_map.csv", index=False, lineterminator="\n")


claims = pd.DataFrame([
    {
        "claim_type": "main_safe_claim",
        "claim": "Official MRI/FTV support-data features produced the strongest current pilot pCR prediction signal.",
        "evidence": "Phase 8C FTV SVM_RBF ranked first with AUROC 0.6667, AUPRC 0.7528, and Brier 0.2405."
    },
    {
        "claim_type": "secondary_safe_claim",
        "claim": "A temporal 3DGCNN-style model was implemented as an exploratory prototype.",
        "evidence": "Stage 12B ran a graph convolutional model on 13 patient-visit graphs."
    },
    {
        "claim_type": "secondary_safe_claim",
        "claim": "The 3DGCNN is not yet the main result.",
        "evidence": "Stage 12B did not outperform Phase 8C FTV SVM."
    },
    {
        "claim_type": "avoid_claim",
        "claim": "The 3DGCNN is clinically validated.",
        "evidence": "Only 13 patient graphs are available."
    },
    {
        "claim_type": "avoid_claim",
        "claim": "The flow or vessel animation measures true blood, milk, or fluid flow.",
        "evidence": "The current animations are conceptual contrast-transport proxies."
    },
    {
        "claim_type": "future_claim",
        "claim": "A larger longitudinal FTV/PE/SER graph model may improve pCR prediction.",
        "evidence": "This requires a larger cohort and richer tumor-region node features."
    }
])
claims.to_csv(TABLES / "stage12d_final_publication_claims.csv", index=False, lineterminator="\n")


outline = pd.DataFrame([
    {
        "section": "Title",
        "content": "Tumor-Source Validation and FTV-Based pCR Prediction in I-SPY2 Breast MRI with an Exploratory Temporal 3DGCNN Framework",
        "purpose": "Honest title. It names FTV as the main result and 3DGCNN as exploratory."
    },
    {
        "section": "Abstract",
        "content": "Summarize tumor-source validation, FTV model result, and exploratory 3DGCNN framework.",
        "purpose": "Do not overclaim clinical validation."
    },
    {
        "section": "Introduction",
        "content": "Explain pCR prediction, ISPY2 MRI, tumor-source reliability, and why temporal graph learning is attractive.",
        "purpose": "Motivate the study."
    },
    {
        "section": "Methods: Dataset",
        "content": "Describe ISPY2 on Cradle, DICOM inventory, and support-data feature source.",
        "purpose": "Show reproducibility."
    },
    {
        "section": "Methods: Tumor-source validation",
        "content": "Describe DICOM SEG diagnosis and why raw masks were not used as the main source.",
        "purpose": "This is a key novelty."
    },
    {
        "section": "Methods: FTV model",
        "content": "Describe pCR label merge, FTV features, and classical ML models.",
        "purpose": "Main modeling path."
    },
    {
        "section": "Methods: Exploratory 3DGCNN",
        "content": "Describe patient-visit graph, node features, temporal edges, GCN layers, and graph pooling.",
        "purpose": "Prototype framework."
    },
    {
        "section": "Results",
        "content": "Report model ranking. FTV SVM is best; 3DGCNN is exploratory.",
        "purpose": "Main results."
    },
    {
        "section": "Discussion",
        "content": "Explain why tumor-source validation matters and why GNN should remain future work.",
        "purpose": "Interpretation."
    },
    {
        "section": "Limitations",
        "content": "Small sample, no clinical validation, conceptual flow visuals, and limited validated 3D masks.",
        "purpose": "Safe scientific framing."
    },
    {
        "section": "Future work",
        "content": "Expand longitudinal FTV/PE/SER graph cohort and true tumor-region node features.",
        "purpose": "Path toward a real 3DGCNN paper."
    }
])
outline.to_csv(TABLES / "stage12d_ideal_paper_outline.csv", index=False, lineterminator="\n")


# ----------------------------
# Figures
# ----------------------------
plt.figure(figsize=(10, 5))
plot = ranking.copy()
labels = plot["analysis"] + "\n" + plot["best_model"]
plt.barh(labels, plot["AUROC"])
plt.xlabel("AUROC")
plt.title("Stage 12D Final Model Ranking")
plt.tight_layout()
plt.savefig(FIGS / "530_stage12d_final_model_ranking.png", dpi=220)
plt.close()

plt.figure(figsize=(11, 6))
plt.axis("off")
txt = (
    "Novel paper idea\n\n"
    "Tumor-source validation\n"
    "↓\n"
    "Official MRI/FTV support-data modeling\n"
    "↓\n"
    "Best current model: FTV SVM_RBF\n"
    "AUROC 0.6667 | AUPRC 0.7528 | Brier 0.2405\n"
    "↓\n"
    "Patient-visit graph construction\n"
    "↓\n"
    "Exploratory temporal 3DGCNN prototype\n"
    "↓\n"
    "Future: larger FTV/PE/SER tumor-region graph model"
)
plt.text(0.5, 0.5, txt, ha="center", va="center", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "531_stage12d_novel_paper_framework.png", dpi=220)
plt.close()

plt.figure(figsize=(11, 6))
plt.axis("off")
txt = (
    "Final claim map\n\n"
    "Safe main claim:\n"
    "FTV support-data features give the strongest current pilot pCR signal.\n\n"
    "Safe secondary claim:\n"
    "3DGCNN-style temporal graph model was implemented as a prototype.\n\n"
    "Avoid:\n"
    "Do not claim clinical validation.\n"
    "Do not claim measured blood/milk/fluid flow.\n"
    "Do not claim 3DGCNN is best model yet."
)
plt.text(0.5, 0.5, txt, ha="center", va="center", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "532_stage12d_claim_map.png", dpi=220)
plt.close()

plt.figure(figsize=(12, 7))
plt.axis("off")
txt = (
    "Ideal paper structure\n\n"
    "1. Clinical problem: pCR prediction in breast MRI\n"
    "2. Technical problem: tumor-source reliability\n"
    "3. Diagnosis: weak DICOM SEG path\n"
    "4. Main solution: official MRI/FTV support-data features\n"
    "5. Main result: FTV SVM_RBF is strongest pilot model\n"
    "6. Prototype: temporal 3DGCNN patient-visit graph\n"
    "7. Future: larger longitudinal FTV/PE/SER graph"
)
plt.text(0.5, 0.5, txt, ha="center", va="center", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "533_stage12d_ideal_paper_structure.png", dpi=220)
plt.close()


# ----------------------------
# Markdown helper
# ----------------------------
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

    for _, r in small.iterrows():
        lines.append("| " + " | ".join(clean(v) for v in r.tolist()) + " |")

    md_path.write_text("\n".join(lines))

make_md(TABLES / "stage12d_final_model_ranking.csv", VIEWS / "stage12d_final_model_ranking.md", "Stage 12D Final Model Ranking")
make_md(TABLES / "stage12d_three_paper_alignment.csv", VIEWS / "stage12d_three_paper_alignment.md", "Stage 12D Three Paper Alignment")
make_md(TABLES / "stage12d_novel_contribution_map.csv", VIEWS / "stage12d_novel_contribution_map.md", "Stage 12D Novel Contribution Map")
make_md(TABLES / "stage12d_final_publication_claims.csv", VIEWS / "stage12d_final_publication_claims.md", "Stage 12D Final Publication Claims")
make_md(TABLES / "stage12d_ideal_paper_outline.csv", VIEWS / "stage12d_ideal_paper_outline.md", "Stage 12D Ideal Paper Outline")


# ----------------------------
# Paper draft markdown
# ----------------------------
paper = []
paper.append("# Tumor-Source Validation and FTV-Based pCR Prediction in I-SPY2 Breast MRI with an Exploratory Temporal 3DGCNN Framework")
paper.append("")
paper.append("## Abstract draft")
paper.append("")
paper.append("This pilot study develops a reproducible Cradle-to-GitHub pipeline for I-SPY2 breast MRI analysis. The study first evaluates tumor-source reliability, showing that raw DICOM SEG masks are not strong enough as the main tumor-source pathway. It then uses official MRI/FTV support-data features for pCR prediction and implements an exploratory temporal 3DGCNN framework over patient-visit graphs. The strongest current model is an FTV support-data SVM_RBF model, with AUROC 0.6667, AUPRC 0.7528, and Brier score 0.2405. The temporal 3DGCNN prototype ran successfully but did not outperform the FTV SVM model. Therefore, FTV-based prediction is the current main pilot result, while temporal 3DGCNN is positioned as future work.")
paper.append("")
paper.append("## Main hypothesis")
paper.append("")
paper.append("Official MRI/FTV support features provide a more reliable pilot pCR prediction signal than weak DICOM SEG-derived features, and these features can later support temporal 3DGCNN modeling when a larger longitudinal graph cohort is available.")
paper.append("")
paper.append("## Main contribution")
paper.append("")
paper.append("1. Tumor-source validation before modeling.")
paper.append("2. Official MRI/FTV support-data pCR modeling.")
paper.append("3. Graph-ready longitudinal patient-visit representation.")
paper.append("4. Exploratory temporal 3DGCNN prototype.")
paper.append("5. Visual 3D tumor and transport-proxy explanation.")
paper.append("")
paper.append("## Main result")
paper.append("")
for _, row in ranking.iterrows():
    paper.append(f"- Rank {int(row['rank'])}: {row['analysis']} using {row['best_model']} gave AUROC {row['AUROC']:.4f}, AUPRC {row['AUPRC']:.4f}, and Brier {row['Brier_score']:.4f}.")
paper.append("")
paper.append("## Safe conclusion")
paper.append("")
paper.append("The official MRI/FTV support-data SVM model is the strongest current pilot model. The temporal 3DGCNN is a working exploratory framework, but it is not yet the main model.")
paper.append("")
paper.append("## Limitation")
paper.append("")
paper.append("This is pilot feasibility work based on a small labeled cohort. It is not clinical validation.")
paper.append("")
paper.append("## Future work")
paper.append("")
paper.append("Future work should expand the longitudinal graph cohort and add richer PE/SER, tumor-region, and validated segmentation features before training a full 3DGCNN as the main model.")

(FINAL / "Stage12D_Ideal_Novel_Paper_Draft.md").write_text("\n".join(paper))


# ----------------------------
# Main Stage 12D report
# ----------------------------
report = []
report.append("# Stage 12D Ideal Novel Paper Framework Report")
report.append("")
report.append("This stage builds the final paper framework based on the three attached paper directions, your ISPY2 pipeline, and the exploratory temporal 3DGCNN result.")
report.append("")
report.append("## Main decision")
report.append("")
report.append("The paper should not claim that the 3DGCNN is the best model. The paper should claim that tumor-source validation and FTV-based pCR prediction are the current main contributions, with temporal 3DGCNN as an exploratory future-ready framework.")
report.append("")
report.append("## Final model ranking")
report.append("")
for _, row in ranking.iterrows():
    report.append(f"- Rank {int(row['rank'])}: {row['analysis']} / {row['best_model']} with AUROC {float(row['AUROC']):.4f}, AUPRC {float(row['AUPRC']):.4f}, and Brier {float(row['Brier_score']):.4f}.")
report.append("")
report.append("## Ideal title")
report.append("")
report.append("Tumor-Source Validation and FTV-Based pCR Prediction in I-SPY2 Breast MRI with an Exploratory Temporal 3DGCNN Framework")
report.append("")
report.append("## Novelty")
report.append("")
report.append("The novelty is not only the model. The novelty is the complete path: validate tumor source first, reject weak DICOM SEG as the main path, use official MRI/FTV features for the main pilot model, and build a temporal graph framework for future 3DGCNN work.")
report.append("")
report.append("## Main outputs")
report.append("")
report.append("- reports/tables/stage12d_final_model_ranking.csv")
report.append("- reports/tables/stage12d_three_paper_alignment.csv")
report.append("- reports/tables/stage12d_novel_contribution_map.csv")
report.append("- reports/tables/stage12d_final_publication_claims.csv")
report.append("- reports/tables/stage12d_ideal_paper_outline.csv")
report.append("- reports/figures/530_stage12d_final_model_ranking.png")
report.append("- reports/figures/531_stage12d_novel_paper_framework.png")
report.append("- reports/figures/532_stage12d_claim_map.png")
report.append("- reports/figures/533_stage12d_ideal_paper_structure.png")
report.append("- reports/final_package/Stage12D_Ideal_Novel_Paper_Draft.md")
report.append("")
report.append("## Next step")
report.append("")
report.append("Use this framework to write the manuscript. Do not run more models unless the goal is to expand the cohort or add richer PE/SER tumor-region features.")

(REPORTS / "Stage12D_Ideal_Novel_Paper_Framework_Report.md").write_text("\n".join(report))


# ----------------------------
# Dashboard and README
# ----------------------------
dash = REPORTS / "Dashboard.md"
old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
dash_add = (
    "\n\n## Stage 12D ideal novel paper framework\n\n"
    "- [Ideal novel paper framework report](Stage12D_Ideal_Novel_Paper_Framework_Report.md)\n"
    "- [Final model ranking](table_views/stage12d_final_model_ranking.md)\n"
    "- [Three paper alignment](table_views/stage12d_three_paper_alignment.md)\n"
    "- [Ideal paper draft](final_package/Stage12D_Ideal_Novel_Paper_Draft.md)\n"
)
if "Stage 12D ideal novel paper framework" not in old_dash:
    dash.write_text(old_dash.rstrip() + dash_add)

readme = REPO / "README.md"
old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
readme_add = (
    "\n\n### Stage 12D: ideal novel paper framework\n\n"
    "Main outputs:\n\n"
    "- reports/Stage12D_Ideal_Novel_Paper_Framework_Report.md\n"
    "- reports/tables/stage12d_final_model_ranking.csv\n"
    "- reports/final_package/Stage12D_Ideal_Novel_Paper_Draft.md\n"
)
if "Stage 12D: ideal novel paper framework" not in old_readme:
    readme.write_text(old_readme.rstrip() + readme_add)

print("Stage 12D complete.")
print("Best current model:", ranking.iloc[0]["analysis"], "/", ranking.iloc[0]["best_model"])
print("Best AUROC:", round(float(ranking.iloc[0]["AUROC"]), 4))
print("3DGCNN rank:", int(ranking[ranking["analysis"].str.contains("3DGCNN", case=False)]["rank"].iloc[0]))
print("Ideal title: Tumor-Source Validation and FTV-Based pCR Prediction in I-SPY2 Breast MRI with an Exploratory Temporal 3DGCNN Framework")
