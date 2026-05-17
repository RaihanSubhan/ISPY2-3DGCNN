from pathlib import Path
import re
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

for p in [REPORTS, TABLES, FIGS, VIEWS]:
    p.mkdir(parents=True, exist_ok=True)

SRC = TABLES / "phase8a_support_source_candidates.csv"

if not SRC.exists():
    raise SystemExit("Missing reports/tables/phase8a_support_source_candidates.csv. Run Phase 8A first.")

df = pd.read_csv(SRC, dtype=str).fillna("")

KEY_PATIENT = ["patient", "subject", "case", "id", "patientid", "ispy"]
KEY_VISIT = ["visit", "time", "timepoint", "t0", "t1", "t2", "t3", "studydate", "acquisition"]
KEY_TUMOR = ["tumor", "tumour", "lesion", "mass", "roi", "mask", "segmentation"]
KEY_FTV = ["ftv", "functional", "functional tumor volume", "functional_tumor_volume"]
KEY_PESER = ["pe", "ser", "percent", "enhancement", "washout"]
KEY_GRAPH = ["supervoxel", "centroid", "node", "edge", "parquet"]
KEY_KINETIC = ["ktrans", "ve", "vp", "perfusion", "adc", "dwi"]
BAD_INTERNAL = [
    "/reports/",
    "/scripts/",
    "/configs/",
    "/docs/",
    "/notebooks/",
    "/slurm/",
    "README.md",
]

def clean_text(x):
    return str(x).replace("\n", " ").replace("\r", " ").strip()

def is_internal_generated(path):
    p = str(path)
    if str(REPO / "reports") in p:
        return True
    if str(REPO / "scripts") in p:
        return True
    if str(REPO / "configs") in p:
        return True
    if str(REPO / "docs") in p:
        return True
    if str(REPO / "notebooks") in p:
        return True
    if str(REPO / "slurm") in p:
        return True
    return False

def has_any(text, keys):
    low = str(text).lower()
    return any(k in low for k in keys)

def get_ext(path):
    p = str(path).lower()
    if p.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(path).suffix.lower()

def read_table(path):
    path = Path(path)
    try:
        ext = get_ext(path)
        if ext == ".csv":
            return pd.read_csv(path, dtype=str, nrows=100).fillna("")
        if ext == ".tsv":
            return pd.read_csv(path, sep="\t", dtype=str, nrows=100).fillna("")
        if ext == ".txt":
            return pd.read_csv(path, sep=None, engine="python", dtype=str, nrows=100).fillna("")
        if ext in [".xlsx", ".xls"]:
            return pd.read_excel(path, dtype=str, nrows=100).fillna("")
    except Exception:
        return None
    return None

def inspect_npz(path):
    try:
        arr = np.load(path, allow_pickle=False)
        rows = []
        for k in arr.files:
            x = arr[k]
            rows.append(f"{k}:{x.shape}:{x.dtype}")
        return "; ".join(rows), "npz_read_ok"
    except Exception as e:
        return "", "npz_read_failed: " + str(e)[:120]

def inspect_docx(path):
    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml")
        root = ET.fromstring(xml)
        texts = []
        for node in root.iter():
            if node.tag.endswith("}t") and node.text:
                texts.append(node.text)
        txt = " ".join(texts)
        txt = re.sub(r"\s+", " ", txt)
        return txt[:3000], "docx_read_ok"
    except Exception as e:
        return "", "docx_read_failed: " + str(e)[:120]

def classify_candidate(path, source, extension, columns, preview, read_status):
    text = " ".join([str(path), str(source), str(extension), str(columns), str(preview)])
    low = text.lower()

    if is_internal_generated(path):
        return "ignore_generated_project_output"

    if "external/ispy2_support_sources" in str(path) or "downloaded_or_extracted_support" in str(source):
        external_flag = True
    else:
        external_flag = False

    if extension in [".docx", ".pdf", ".html", ".dat"] and "description" in low:
        return "documentation_only"

    if has_any(low, KEY_GRAPH):
        return "graph_or_supervoxel_source"

    if has_any(low, KEY_FTV):
        return "ftv_source_candidate"

    if has_any(low, KEY_PESER):
        return "pe_ser_source_candidate"

    if has_any(low, KEY_TUMOR):
        return "tumor_mask_or_roi_source_candidate"

    if has_any(low, KEY_KINETIC):
        return "kinetic_or_diffusion_source_candidate"

    if external_flag:
        return "external_support_unknown"

    return "low_priority_unknown"

def score_candidate(row):
    source_class = row["source_class"]
    score = 0

    if source_class == "ignore_generated_project_output":
        return -1000

    if source_class == "documentation_only":
        return 10

    if source_class in [
        "ftv_source_candidate",
        "pe_ser_source_candidate",
        "tumor_mask_or_roi_source_candidate",
        "graph_or_supervoxel_source",
        "kinetic_or_diffusion_source_candidate",
    ]:
        score += 100

    if row["has_patient_column"] == "yes":
        score += 25

    if row["has_visit_column"] == "yes":
        score += 20

    if row["has_tumor_feature_column"] == "yes":
        score += 25

    if row["has_pe_ser_or_ftv_column"] == "yes":
        score += 30

    if row["read_status"].endswith("ok") or "read_ok" in row["read_status"]:
        score += 10

    try:
        size = float(row.get("size_mb", 0))
        if size > 0:
            score += min(10, size)
    except Exception:
        pass

    return score

triage_rows = []
manifest_rows = []

for _, r in df.iterrows():
    path = r.get("file_path", "")
    source = r.get("source", "")
    ext = r.get("extension", get_ext(path))
    size_mb = r.get("size_mb", "")
    original_type = r.get("candidate_type", "")
    keyword_hits = r.get("keyword_hits", "")
    read_status = r.get("read_status", "")

    p = Path(path)

    columns = r.get("columns_or_keys_preview", "")
    preview = r.get("first_row_preview", "")

    if p.exists():
        ext2 = get_ext(p)

        if ext2 in [".csv", ".tsv", ".txt", ".xlsx", ".xls"]:
            tab = read_table(p)
            if tab is not None:
                columns = "; ".join([str(c) for c in tab.columns])
                preview = "; ".join([str(v)[:80] for v in tab.head(1).values.flatten()[:30]])
                read_status = "read_table_ok"
            else:
                read_status = "read_table_failed"

        elif ext2 == ".npz":
            columns, read_status = inspect_npz(p)

        elif ext2 == ".docx":
            preview, read_status = inspect_docx(p)
            columns = "docx_text"

    text_for_cols = " ".join([columns, preview, path])

    has_patient = "yes" if has_any(text_for_cols, KEY_PATIENT) else "no"
    has_visit = "yes" if has_any(text_for_cols, KEY_VISIT) else "no"
    has_tumor = "yes" if has_any(text_for_cols, KEY_TUMOR + KEY_KINETIC) else "no"
    has_peser_ftv = "yes" if has_any(text_for_cols, KEY_FTV + KEY_PESER) else "no"

    source_class = classify_candidate(path, source, ext, columns, preview, read_status)

    out = {
        "file_path": path,
        "source": source,
        "extension": ext,
        "size_mb": size_mb,
        "original_candidate_type": original_type,
        "source_class": source_class,
        "keyword_hits": keyword_hits,
        "read_status": read_status,
        "has_patient_column": has_patient,
        "has_visit_column": has_visit,
        "has_tumor_feature_column": has_tumor,
        "has_pe_ser_or_ftv_column": has_peser_ftv,
        "columns_or_keys_preview": columns,
        "first_row_or_doc_preview": preview[:500],
    }

    out["priority_score_phase8b"] = score_candidate(out)

    if source_class == "ignore_generated_project_output":
        out["manifest_decision"] = "exclude_generated_output"
    elif source_class == "documentation_only":
        out["manifest_decision"] = "keep_as_documentation"
    elif out["priority_score_phase8b"] >= 120:
        out["manifest_decision"] = "primary_candidate"
    elif out["priority_score_phase8b"] >= 80:
        out["manifest_decision"] = "secondary_candidate"
    else:
        out["manifest_decision"] = "low_priority"

    triage_rows.append(out)

triage = pd.DataFrame(triage_rows)
triage = triage.sort_values("priority_score_phase8b", ascending=False)
triage.to_csv(TABLES / "phase8b_support_source_triage.csv", index=False)

manifest = triage[triage["manifest_decision"].isin(["primary_candidate", "secondary_candidate", "keep_as_documentation"])].copy()
manifest.to_csv(TABLES / "phase8b_tumor_source_manifest.csv", index=False)

def make_md(csv_path, md_path, title, max_rows=80, max_cols=12):
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    small = df.head(max_rows).iloc[:, :max_cols]

    def c(x):
        x = str(x).replace("|", "\\|").replace("\n", " ")
        return x[:87] + "..." if len(x) > 90 else x

    lines = []
    lines.append("# " + title)
    lines.append("")
    lines.append("Rows: " + str(len(df)))
    lines.append("Columns: " + str(len(df.columns)))
    lines.append("")
    lines.append("| " + " | ".join(c(col) for col in small.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(small.columns)) + " |")

    for _, row in small.iterrows():
        lines.append("| " + " | ".join(c(v) for v in row.tolist()) + " |")

    md_path.write_text("\n".join(lines))

make_md(TABLES / "phase8b_support_source_triage.csv", VIEWS / "phase8b_support_source_triage.md", "Phase 8B Support Source Triage")
make_md(TABLES / "phase8b_tumor_source_manifest.csv", VIEWS / "phase8b_tumor_source_manifest.md", "Phase 8B Tumor Source Manifest")

# Figures
plt.figure(figsize=(10, 5))
triage["source_class"].value_counts().sort_values().plot(kind="barh")
plt.title("Phase 8B Source Classes")
plt.xlabel("Files")
plt.tight_layout()
plt.savefig(FIGS / "240_phase8b_source_classes.png", dpi=220)
plt.close()

plt.figure(figsize=(10, 6))
top = triage.head(20).copy()
labels = top["source_class"].astype(str) + " | " + top["extension"].astype(str)
plt.barh(labels, pd.to_numeric(top["priority_score_phase8b"], errors="coerce").fillna(0))
plt.title("Phase 8B Top Tumor-Source Candidates")
plt.xlabel("Priority score")
plt.tight_layout()
plt.savefig(FIGS / "241_phase8b_top_source_scores.png", dpi=220)
plt.close()

plt.figure(figsize=(8, 5))
triage["manifest_decision"].value_counts().sort_values().plot(kind="barh")
plt.title("Phase 8B Manifest Decisions")
plt.xlabel("Files")
plt.tight_layout()
plt.savefig(FIGS / "242_phase8b_manifest_decisions.png", dpi=220)
plt.close()

n_total = len(triage)
n_internal = int((triage["source_class"] == "ignore_generated_project_output").sum())
n_doc = int((triage["source_class"] == "documentation_only").sum())
n_primary = int((triage["manifest_decision"] == "primary_candidate").sum())
n_secondary = int((triage["manifest_decision"] == "secondary_candidate").sum())
n_manifest = len(manifest)

report = []
report.append("# Phase 8B Tumor Source Manifest Report")
report.append("")
report.append("This phase cleans the Phase 8A support-candidate list and separates real external/support sources from generated project outputs.")
report.append("")
report.append("## Main outputs")
report.append("")
report.append("- reports/tables/phase8b_support_source_triage.csv")
report.append("- reports/tables/phase8b_tumor_source_manifest.csv")
report.append("- reports/table_views/phase8b_support_source_triage.md")
report.append("- reports/table_views/phase8b_tumor_source_manifest.md")
report.append("- reports/figures/240_phase8b_source_classes.png")
report.append("- reports/figures/241_phase8b_top_source_scores.png")
report.append("- reports/figures/242_phase8b_manifest_decisions.png")
report.append("")
report.append("## Main counts")
report.append("")
report.append("- Total Phase 8A candidates triaged: " + str(n_total))
report.append("- Generated project outputs excluded: " + str(n_internal))
report.append("- Documentation-only sources: " + str(n_doc))
report.append("- Primary candidates: " + str(n_primary))
report.append("- Secondary candidates: " + str(n_secondary))
report.append("- Manifest rows kept: " + str(n_manifest))
report.append("")
report.append("## Interpretation")
report.append("")

if n_primary > 0:
    report.append("Phase 8B found one or more primary tumor-source candidates. Inspect the manifest and select the strongest source for Phase 8C.")
elif n_secondary > 0:
    report.append("Phase 8B found secondary candidates but no strong primary source. These may be useful, but manual review is needed before using them.")
else:
    report.append("Phase 8B did not find a strong real tumor-source file. Most Phase 8A candidates appear to be generated project reports, documentation, or low-priority files.")

report.append("")
report.append("## Next step")
report.append("")

if n_primary > 0:
    report.append("Run Phase 8C to build a patient/visit tumor-source manifest from the highest-priority primary candidate.")
else:
    report.append("If no primary candidate exists, manually download or place the official ISPY2 image-analysis support package under `external/ispy2_support_sources/`, then rerun Phase 8A and Phase 8B.")

(REPORTS / "Phase8B_Tumor_Source_Manifest_Report.md").write_text("\n".join(report))

# Dashboard update
dash = REPORTS / "Dashboard.md"
old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
dash_add = "\n\n## Phase 8B tumor source manifest\n\n- [Tumor source manifest report](Phase8B_Tumor_Source_Manifest_Report.md)\n- [Support source triage](table_views/phase8b_support_source_triage.md)\n- [Tumor source manifest](table_views/phase8b_tumor_source_manifest.md)\n"
if "Phase 8B tumor source manifest" not in old_dash:
    dash.write_text(old_dash.rstrip() + dash_add)

# README update
readme = REPO / "README.md"
old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
readme_add = "\n\n### Phase 8B: tumor source manifest\n\nMain outputs:\n\n- reports/Phase8B_Tumor_Source_Manifest_Report.md\n- reports/tables/phase8b_tumor_source_manifest.csv\n- reports/tables/phase8b_support_source_triage.csv\n"
if "Phase 8B: tumor source manifest" not in old_readme:
    readme.write_text(old_readme.rstrip() + readme_add)

print("Phase 8B complete.")
print("Total candidates triaged:", n_total)
print("Generated project outputs excluded:", n_internal)
print("Documentation-only sources:", n_doc)
print("Primary candidates:", n_primary)
print("Secondary candidates:", n_secondary)
print("Manifest rows kept:", n_manifest)
