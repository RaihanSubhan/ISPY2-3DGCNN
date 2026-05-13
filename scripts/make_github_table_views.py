from pathlib import Path
import pandas as pd

REPO = Path.home() / "ISPY2-3DGCNN"
TABLES = REPO / "reports" / "tables"
VIEWS = REPO / "reports" / "table_views"
REPORTS = REPO / "reports"

VIEWS.mkdir(parents=True, exist_ok=True)

FILES = [
    "ispy2_patient_summary.csv",
    "cohort_visit_tracker_table.csv",
    "candidate_mr_series_best_per_patient.csv",
    "seg_series_metadata.csv",
    "seg_mr_link_table.csv",
    "seg_mask_qc_features.csv",
    "unified_model_heads_for_atlas.csv",
]

def md_escape(x):
    text = "" if pd.isna(x) else str(x)
    text = text.replace("|", "\\|").replace("\n", " ")
    if len(text) > 90:
        text = text[:87] + "..."
    return text

def dataframe_to_markdown(df, max_rows=50, max_cols=10):
    small = df.head(max_rows).iloc[:, :max_cols].copy()
    cols = [md_escape(c) for c in small.columns]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in small.iterrows():
        vals = [md_escape(v) for v in row.tolist()]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

summary_rows = []

for fname in FILES:
    path = TABLES / fname
    if not path.exists():
        summary_rows.append({
            "table": fname,
            "exists": "no",
            "rows": 0,
            "columns": 0,
            "view_file": "",
        })
        continue

    df = pd.read_csv(path, dtype=str)
    view_name = fname.replace(".csv", ".md")
    view_path = VIEWS / view_name

    content = []
    content.append(f"# {fname}")
    content.append("")
    content.append(f"Original CSV: `reports/tables/{fname}`")
    content.append("")
    content.append(f"Rows: {len(df)}")
    content.append(f"Columns: {len(df.columns)}")
    content.append("")
    content.append("This page shows the first 50 rows and first 10 columns only.")
    content.append("Open the CSV file for the full GitHub interactive table.")
    content.append("")
    content.append(dataframe_to_markdown(df, max_rows=50, max_cols=10))
    content.append("")

    view_path.write_text("\n".join(content))

    summary_rows.append({
        "table": fname,
        "exists": "yes",
        "rows": len(df),
        "columns": len(df.columns),
        "view_file": f"reports/table_views/{view_name}",
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(TABLES / "github_table_view_index.csv", index=False)

dash = []
dash.append("# ISPY2 4D Atlas Dashboard")
dash.append("")
dash.append("This dashboard gives GitHub-readable table previews for the ISPY2 4D vascular-perfusion atlas project.")
dash.append("")
dash.append("## Main table views")
dash.append("")

for row in summary_rows:
    if row["exists"] == "yes":
        dash.append(f"- [{row['table']}](table_views/{row['table'].replace('.csv', '.md')})")
    else:
        dash.append(f"- {row['table']}: not found")

dash.append("")
dash.append("## Notes")
dash.append("")
dash.append("- GitHub also renders the original `.csv` files as interactive tables.")
dash.append("- Markdown table views are smaller previews for README-style reading.")
dash.append("- Raw ISPY2 DICOM data is not stored in this repository.")
dash.append("")

(REPORTS / "Dashboard.md").write_text("\n".join(dash))

print("Created GitHub table views.")
print(f"Dashboard: {REPORTS / 'Dashboard.md'}")
print(f"Table views folder: {VIEWS}")
