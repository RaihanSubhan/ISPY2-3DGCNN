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

TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
VIEWS.mkdir(parents=True, exist_ok=True)

bio_path = TABLES / "phase3d_verified_biomarker_table.csv"
candidate_path = TABLES / "phase4b_label_merge_candidates.csv"
old_template_path = TABLES / "phase4a_required_label_template.csv"

if not bio_path.exists():
    raise SystemExit("Missing reports/tables/phase3d_verified_biomarker_table.csv. Run Phase 3D first.")

bio = pd.read_csv(bio_path, dtype=str).fillna("")

# Patient-level biomarker summary
if "patient_folder" not in bio.columns:
    raise SystemExit("phase3d table does not contain patient_folder column.")

patient_rows = []

for pid, sub in bio.groupby("patient_folder"):
    tier_counts = sub["biomarker_tier"].value_counts().to_dict() if "biomarker_tier" in sub.columns else {}

    volume_vals = pd.to_numeric(sub.get("tumor_volume_proxy_cm3", pd.Series([])), errors="coerce")
    median_volume = volume_vals.median() if volume_vals.notna().sum() > 0 else ""

    patient_rows.append({
        "patient_folder": pid,
        "pCR_label": "",
        "response_label": "",
        "tumor_subtype": "",
        "treatment_arm": "",
        "label_source": "",
        "notes": "",
        "n_biomarker_records": len(sub),
        "n_strict_verified_records": tier_counts.get("strict_verified", 0),
        "n_warning_records": tier_counts.get("geometry_warning", 0) + tier_counts.get("z_verified_shape_warning", 0),
        "median_tumor_volume_proxy_cm3": median_volume,
    })

template = pd.DataFrame(patient_rows).sort_values("patient_folder")

# Preserve old labels if they exist
if old_template_path.exists():
    old = pd.read_csv(old_template_path, dtype=str).fillna("")
    if "patient_folder" in old.columns:
        keep_cols = ["patient_folder", "pCR_label", "response_label", "tumor_subtype", "treatment_arm", "label_source", "notes"]
        old_keep = old[[c for c in keep_cols if c in old.columns]].copy()

        template = template.drop(columns=[c for c in old_keep.columns if c != "patient_folder" and c in template.columns])
        template = template.merge(old_keep, on="patient_folder", how="left")

        for c in keep_cols:
            if c not in template.columns:
                template[c] = ""
            template[c] = template[c].fillna("")

        ordered = ["patient_folder", "pCR_label", "response_label", "tumor_subtype", "treatment_arm", "label_source", "notes"]
        rest = [c for c in template.columns if c not in ordered]
        template = template[ordered + rest]

# Write both names
template.to_csv(TABLES / "phase4a_required_label_template.csv", index=False)
template.to_csv(TABLES / "phase4c_pcr_label_template.csv", index=False)

# Candidate label review table
if candidate_path.exists():
    cand = pd.read_csv(candidate_path, dtype=str).fillna("")
else:
    cand = pd.DataFrame()

if not cand.empty:
    for col in ["usable_labeled_patients", "patient_overlap", "n_positive", "n_negative"]:
        if col in cand.columns:
            cand[col] = pd.to_numeric(cand[col], errors="coerce").fillna(0)

    sort_cols = [c for c in ["usable_labeled_patients", "patient_overlap", "n_positive", "n_negative"] if c in cand.columns]
    cand_review = cand.sort_values(sort_cols, ascending=False) if sort_cols else cand.copy()
else:
    cand_review = pd.DataFrame(columns=[
        "file_path", "patient_column", "patient_overlap", "label_column",
        "usable_labeled_patients", "n_positive", "n_negative", "status"
    ])

cand_review.to_csv(TABLES / "phase4c_label_candidate_review.csv", index=False)

# Markdown helper
def make_md(csv_path, md_path, title, max_rows=80, max_cols=12):
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    small = df.head(max_rows).iloc[:, :max_cols]

    def clean(x):
        x = str(x).replace("|", "\\|").replace("\n", " ")
        if len(x) > 90:
            x = x[:87] + "..."
        return x

    lines = []
    lines.append("# " + title)
    lines.append("")
    lines.append("Rows: " + str(len(df)))
    lines.append("Columns: " + str(len(df.columns)))
    lines.append("")
    lines.append("This is a GitHub-readable table preview.")
    lines.append("")
    lines.append("| " + " | ".join(clean(c) for c in small.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(small.columns)) + " |")

    for _, row in small.iterrows():
        lines.append("| " + " | ".join(clean(v) for v in row.tolist()) + " |")

    md_path.write_text("\n".join(lines))

make_md(
    TABLES / "phase4c_pcr_label_template.csv",
    VIEWS / "phase4c_pcr_label_template.md",
    "Phase 4C pCR Label Template"
)

make_md(
    TABLES / "phase4c_label_candidate_review.csv",
    VIEWS / "phase4c_label_candidate_review.md",
    "Phase 4C Label Candidate Review"
)

# Figures
plt.figure(figsize=(8, 5))
label_status = template["pCR_label"].replace("", "blank").value_counts()
label_status.plot(kind="bar")
plt.title("Phase 4C pCR Label Template Status")
plt.ylabel("Patient count")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(FIGS / "80_phase4c_label_template_status.png", dpi=220)
plt.close()

plt.figure(figsize=(9, 6))
if not cand_review.empty and "usable_labeled_patients" in cand_review.columns:
    top = cand_review.head(15).copy()
    label = top["label_column"].astype(str).replace("", "none")
    plt.barh(label, pd.to_numeric(top["usable_labeled_patients"], errors="coerce").fillna(0))
    plt.xlabel("Usable labeled patients")
    plt.title("Phase 4C Candidate Label Columns")
else:
    plt.text(0.5, 0.5, "No candidate label table found", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "81_phase4c_label_candidate_review.png", dpi=220)
plt.close()

# Summary
n_patients = len(template)
n_filled = int(template["pCR_label"].astype(str).str.strip().isin(["0", "1", "0.0", "1.0"]).sum())
n_blank = n_patients - n_filled
n_candidates = len(cand_review)

summary = [
    "# Phase 4C pCR Label Resolution Summary",
    "",
    "Phase 4B did not find a usable two-class pCR or response label automatically. Phase 4C creates a clean label template for the biomarker-ready patients.",
    "",
    "## Main outputs",
    "",
    "- reports/tables/phase4a_required_label_template.csv",
    "- reports/tables/phase4c_pcr_label_template.csv",
    "- reports/tables/phase4c_label_candidate_review.csv",
    "- reports/table_views/phase4c_pcr_label_template.md",
    "- reports/table_views/phase4c_label_candidate_review.md",
    "- reports/figures/80_phase4c_label_template_status.png",
    "- reports/figures/81_phase4c_label_candidate_review.png",
    "",
    "## Main counts",
    "",
    "- Patients needing labels: " + str(n_patients),
    "- Patients with pCR_label already filled: " + str(n_filled),
    "- Patients still blank: " + str(n_blank),
    "- Candidate label rows reviewed: " + str(n_candidates),
    "",
    "## How to fill the template",
    "",
    "Fill `pCR_label` using this rule:",
    "",
    "- 1 = pCR / responder",
    "- 0 = non-pCR / non-responder",
    "- blank = unknown",
    "",
    "## Next step",
    "",
    "After pCR labels are filled, rerun Phase 4B. Then baseline ML models can train.",
]

(REPORTS / "Phase4C_PCR_Label_Resolution_Summary.md").write_text("\n".join(summary))

# Dashboard update
dash_path = REPORTS / "Dashboard.md"
old_dash = dash_path.read_text() if dash_path.exists() else "# ISPY2 4D Atlas Dashboard\n"
dash_add = "\n\n## Phase 4C table views\n\n- [pCR label template](table_views/phase4c_pcr_label_template.md)\n- [Label candidate review](table_views/phase4c_label_candidate_review.md)\n"
if "Phase 4C table views" not in old_dash:
    dash_path.write_text(old_dash.rstrip() + dash_add)

# README update
readme_path = REPO / "README.md"
old_readme = readme_path.read_text() if readme_path.exists() else "# ISPY2-3DGCNN\n"
readme_add = "\n\n### Phase 4C: pCR label resolution\n\nMain outputs:\n\n- reports/tables/phase4a_required_label_template.csv\n- reports/tables/phase4c_pcr_label_template.csv\n- reports/Phase4C_PCR_Label_Resolution_Summary.md\n"
if "Phase 4C: pCR label resolution" not in old_readme:
    readme_path.write_text(old_readme.rstrip() + readme_add)

print("Phase 4C complete.")
print("Patients needing labels:", n_patients)
print("Filled pCR labels:", n_filled)
print("Blank pCR labels:", n_blank)
print("Candidate label rows reviewed:", n_candidates)
