from pathlib import Path
import os
import re
import hashlib
import urllib.request
import urllib.parse
import zipfile
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
CLIN = REPO / "external" / "ispy2_clinical_support"

for p in [REPORTS, TABLES, FIGS, VIEWS, CLIN]:
    p.mkdir(parents=True, exist_ok=True)

TEMPLATE = TABLES / "phase4c_pcr_label_template.csv"

if not TEMPLATE.exists():
    raise SystemExit("Missing reports/tables/phase4c_pcr_label_template.csv. Run Phase 4C first.")

template = pd.read_csv(TEMPLATE, dtype=str).fillna("")
patients = sorted(set(template["patient_folder"].astype(str)))

URLS_TO_SCAN = [
    "https://www.cancerimagingarchive.net/collection/ispy2/",
    "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230072",
]

TABLE_EXTS = [".csv", ".tsv", ".txt", ".xlsx", ".xls", ".zip"]
LABEL_KEYS = [
    "pcr", "pathologic", "pathological", "complete", "response",
    "responder", "rcb", "outcome", "clinical", "treatment",
    "arm", "subtype", "her2", "hr", "er", "pr", "tnbc"
]

def norm_patient(x):
    s = str(x).upper().strip()
    m = re.search(r"ISPY2[-_ ]?(\d+)", s)
    if m:
        return "ISPY2-" + m.group(1)
    m = re.search(r"(\d{6})", s)
    if m:
        return "ISPY2-" + m.group(1)
    return s

def short_file_name(url):
    parsed = urllib.parse.urlparse(url)
    base = Path(urllib.parse.unquote(parsed.path)).name
    if not base:
        base = "downloaded_file"
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    ext = Path(base).suffix.lower()
    if ext not in TABLE_EXTS:
        ext = ".dat"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    stem = Path(base).stem[:40]
    return f"{stem}_{digest}{ext}"

def fetch_html(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=45) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return ""

def find_candidate_links(html, base_url):
    links = set()

    for m in re.finditer(r'href=["\']([^"\']+)["\']', html, flags=re.I):
        href = m.group(1)
        full = urllib.parse.urljoin(base_url, href)
        low = full.lower()

        # Avoid huge non-data Atlassian/application-link URLs.
        has_table_ext = any(low.split("?")[0].endswith(ext) for ext in TABLE_EXTS)
        has_label_word = any(k in low for k in LABEL_KEYS)
        is_download_like = "download" in low or "attachment" in low

        if has_table_ext or (has_label_word and is_download_like):
            links.add(full)

    return sorted(links)

def download_small(url, max_mb=40):
    out = CLIN / short_file_name(url)

    if out.exists() and out.stat().st_size > 0:
        return out, "already_exists"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=90) as r:
            data = r.read(max_mb * 1024 * 1024 + 1)

        if len(data) > max_mb * 1024 * 1024:
            return None, "too_large_skipped"

        out.write_bytes(data)
        return out, "downloaded"
    except Exception as e:
        return None, "download_failed: " + str(e)[:160]

def read_table(path):
    path = Path(path)
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, dtype=str).fillna("")
        if path.suffix.lower() == ".tsv":
            return pd.read_csv(path, sep="\t", dtype=str).fillna("")
        if path.suffix.lower() == ".txt":
            return pd.read_csv(path, sep=None, engine="python", dtype=str).fillna("")
        if path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(path, dtype=str).fillna("")
    except Exception:
        return None
    return None

def extract_zip_tables(zip_path):
    extracted = []
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            for info in z.infolist():
                name_low = info.filename.lower()
                if not any(name_low.endswith(ext) for ext in [".csv", ".tsv", ".txt", ".xlsx", ".xls"]):
                    continue
                if info.file_size > 40 * 1024 * 1024:
                    continue

                safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(info.filename).name)
                out = CLIN / f"zip_{hashlib.sha1((str(zip_path)+info.filename).encode()).hexdigest()[:10]}_{safe}"

                with z.open(info) as src:
                    out.write_bytes(src.read())

                extracted.append(str(out))
    except Exception:
        pass
    return extracted

def find_patient_col(df):
    best_col = ""
    best_overlap = -1

    patient_set = set(patients)

    for col in df.columns:
        vals = set(df[col].astype(str).map(norm_patient))
        overlap = len(vals.intersection(patient_set))

        low = col.lower()
        if "patient" in low or "subject" in low or low in ["id", "patientid"]:
            overlap += 2

        if overlap > best_overlap:
            best_overlap = overlap
            best_col = col

    return best_col, max(best_overlap - 2, 0)

def find_label_cols(df):
    cols = []
    bad = ["path", "file", "folder", "series", "study", "uid", "description"]

    for col in df.columns:
        low = col.lower()
        if any(k in low for k in LABEL_KEYS) and not any(b in low for b in bad):
            cols.append(col)

    return cols

def map_label(v, col):
    s = str(v).strip().lower()
    c = str(col).lower()

    if s in ["", "nan", "none", "null", "na", "n/a", "unknown"]:
        return np.nan

    if "rcb" in c or "residual" in c:
        if s in ["0", "0.0", "rcb0", "rcb 0", "rcb-0", "class 0"]:
            return 1
        if s in ["1", "1.0", "2", "2.0", "3", "3.0", "rcb1", "rcb2", "rcb3", "rcb-i", "rcb-ii", "rcb-iii"]:
            return 0

    negatives = ["non-pcr", "non pcr", "no pcr", "not pcr", "nonresponder", "non-responder", "no", "false", "not complete"]
    positives = ["pcr", "yes", "true", "complete response", "pathologic complete", "pathological complete", "responder"]

    for w in negatives:
        if w in s:
            return 0

    for w in positives:
        if w in s:
            return 1

    if s in ["1", "1.0"]:
        return 1
    if s in ["0", "0.0"]:
        return 0

    return np.nan

# 1. Scan pages
scan_rows = []
download_rows = []
candidate_links = []

for url in URLS_TO_SCAN:
    html = fetch_html(url)
    links = find_candidate_links(html, url) if html else []

    scan_rows.append({
        "source_url": url,
        "html_read": "yes" if html else "no",
        "candidate_links_found": len(links),
    })

    candidate_links.extend(links)

# 2. Download candidate table files
local_files = []

for link in sorted(set(candidate_links)):
    out, status = download_small(link)
    download_rows.append({
        "url": link,
        "local_file": str(out) if out else "",
        "status": status,
    })

    if out:
        if out.suffix.lower() == ".zip":
            local_files.extend(extract_zip_tables(out))
        elif out.suffix.lower() in [".csv", ".tsv", ".txt", ".xlsx", ".xls"]:
            local_files.append(str(out))

# 3. Also inspect any manually placed clinical files
for root, _, files in os.walk(CLIN):
    for f in files:
        if f.lower().endswith((".csv", ".tsv", ".txt", ".xlsx", ".xls")):
            local_files.append(str(Path(root) / f))

local_files = sorted(set(local_files))

# 4. Inspect label columns
label_rows = []
label_maps = []

for f in local_files:
    df = read_table(f)

    if df is None or df.empty:
        label_rows.append({
            "file_path": f,
            "read_status": "failed_or_empty",
            "patient_col": "",
            "patient_overlap": 0,
            "label_col": "",
            "usable_labels": 0,
            "positive": 0,
            "negative": 0,
        })
        continue

    pc, overlap = find_patient_col(df)
    label_cols = find_label_cols(df)

    if not pc or not label_cols:
        label_rows.append({
            "file_path": f,
            "read_status": "read_ok_no_label_col",
            "patient_col": pc,
            "patient_overlap": overlap,
            "label_col": "",
            "usable_labels": 0,
            "positive": 0,
            "negative": 0,
        })
        continue

    for lc in label_cols:
        temp = pd.DataFrame()
        temp["patient_folder"] = df[pc].astype(str).map(norm_patient)
        temp["pCR_label"] = df[lc].map(lambda x: map_label(x, lc))
        temp = temp.dropna(subset=["pCR_label"])
        temp["pCR_label"] = temp["pCR_label"].astype(int)
        temp = temp[temp["patient_folder"].isin(patients)]
        temp = temp.drop_duplicates("patient_folder")

        n_pos = int((temp["pCR_label"] == 1).sum())
        n_neg = int((temp["pCR_label"] == 0).sum())

        label_rows.append({
            "file_path": f,
            "read_status": "read_ok",
            "patient_col": pc,
            "patient_overlap": overlap,
            "label_col": lc,
            "usable_labels": len(temp),
            "positive": n_pos,
            "negative": n_neg,
        })

        if len(temp) >= 6 and n_pos >= 2 and n_neg >= 2:
            temp["label_source_file"] = f
            temp["label_source_column"] = lc
            label_maps.append(temp)

scan_df = pd.DataFrame(scan_rows)
down_df = pd.DataFrame(download_rows)
label_df = pd.DataFrame(label_rows)

scan_df.to_csv(TABLES / "phase4e_clinical_page_scan.csv", index=False)
down_df.to_csv(TABLES / "phase4e_clinical_download_log.csv", index=False)
label_df.to_csv(TABLES / "phase4e_downloaded_label_candidates.csv", index=False)

# 5. Fill template if a usable label file exists
filled = False
best_n = 0

if label_maps:
    best = sorted(label_maps, key=lambda x: len(x), reverse=True)[0]
    best_n = len(best)

    updated = template.copy()

    if "pCR_label" in updated.columns:
        updated = updated.drop(columns=["pCR_label"])
    if "label_source" in updated.columns:
        updated = updated.drop(columns=["label_source"])

    updated = updated.merge(
        best[["patient_folder", "pCR_label", "label_source_file", "label_source_column"]],
        on="patient_folder",
        how="left",
    )

    updated["pCR_label"] = updated["pCR_label"].fillna("")
    updated["label_source"] = (
        updated["label_source_file"].fillna("") + "::" + updated["label_source_column"].fillna("")
    ).str.strip(":")

    updated = updated.drop(columns=["label_source_file", "label_source_column"], errors="ignore")

    ordered = ["patient_folder", "pCR_label", "response_label", "tumor_subtype", "treatment_arm", "label_source", "notes"]
    for c in ordered:
        if c not in updated.columns:
            updated[c] = ""

    rest = [c for c in updated.columns if c not in ordered]
    updated = updated[ordered + rest]

    updated.to_csv(TABLES / "phase4c_pcr_label_template.csv", index=False)
    updated.to_csv(TABLES / "phase4a_required_label_template.csv", index=False)
    best.to_csv(TABLES / "phase4e_auto_pcr_label_mapping.csv", index=False)

    filled = True

# 6. Markdown previews
def make_md(csv_path, md_path, title, max_rows=50, max_cols=10):
    if not csv_path.exists():
        return

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
    lines.append("| " + " | ".join(clean(c) for c in small.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(small.columns)) + " |")

    for _, row in small.iterrows():
        lines.append("| " + " | ".join(clean(v) for v in row.tolist()) + " |")

    md_path.write_text("\n".join(lines))

make_md(TABLES / "phase4e_clinical_download_log.csv", VIEWS / "phase4e_clinical_download_log.md", "Phase 4E Clinical Download Log")
make_md(TABLES / "phase4e_downloaded_label_candidates.csv", VIEWS / "phase4e_downloaded_label_candidates.md", "Phase 4E Downloaded Label Candidates")

# 7. Figure
plt.figure(figsize=(9, 5))
if not label_df.empty and "usable_labels" in label_df.columns:
    plot = label_df.sort_values("usable_labels", ascending=False).head(15)
    y = plot["label_col"].astype(str).replace("", "none").str[:40]
    x = pd.to_numeric(plot["usable_labels"], errors="coerce").fillna(0)
    plt.barh(y, x)
    plt.xlabel("Usable labels matched to biomarker patients")
    plt.title("Phase 4E Downloaded Clinical Label Candidates")
else:
    plt.text(0.5, 0.5, "No downloaded label candidates found", ha="center", va="center")
    plt.axis("off")

plt.tight_layout()
plt.savefig(FIGS / "100_phase4e_downloaded_label_candidates.png", dpi=220)
plt.close()

n_files = len(local_files)
n_candidates = len(label_df)
n_usable_sources = int((pd.to_numeric(label_df.get("usable_labels", pd.Series(dtype=float)), errors="coerce").fillna(0) >= 6).sum()) if len(label_df) else 0

summary = []
summary.append("# Phase 4E Official pCR Label Acquisition Summary")
summary.append("")
summary.append("This phase tries to locate or download official clinical support tables and map pCR labels into the project template.")
summary.append("")
summary.append("## Main outputs")
summary.append("")
summary.append("- reports/tables/phase4e_clinical_page_scan.csv")
summary.append("- reports/tables/phase4e_clinical_download_log.csv")
summary.append("- reports/tables/phase4e_downloaded_label_candidates.csv")
summary.append("- reports/table_views/phase4e_downloaded_label_candidates.md")
summary.append("- reports/table_views/phase4e_clinical_download_log.md")
summary.append("- reports/figures/100_phase4e_downloaded_label_candidates.png")
summary.append("")
summary.append("## Main counts")
summary.append("")
summary.append("- Clinical table files inspected: " + str(n_files))
summary.append("- Label candidate rows created: " + str(n_candidates))
summary.append("- Candidate sources with at least 6 usable labels: " + str(n_usable_sources))
summary.append("- Automatic label template filled: " + ("yes" if filled else "no"))
summary.append("- Automatically filled patients: " + str(best_n))
summary.append("")
summary.append("## Next step")
summary.append("")

if filled:
    summary.append("The pCR label template was filled automatically. Rerun Phase 4B to train baseline ML models.")
else:
    summary.append("No usable pCR label source was found automatically. Manually place the TCIA clinical CSV in `external/ispy2_clinical_support/` or fill `reports/tables/phase4c_pcr_label_template.csv`, then rerun Phase 4B.")

(REPORTS / "Phase4E_Official_PCR_Label_Acquisition_Summary.md").write_text("\n".join(summary))

# Dashboard update
dash_path = REPORTS / "Dashboard.md"
old = dash_path.read_text() if dash_path.exists() else "# ISPY2 4D Atlas Dashboard\n"
add = "\n\n## Phase 4E table views\n\n- [Downloaded label candidates](table_views/phase4e_downloaded_label_candidates.md)\n- [Clinical download log](table_views/phase4e_clinical_download_log.md)\n"
if "Phase 4E table views" not in old:
    dash_path.write_text(old.rstrip() + add)

print("Phase 4E complete.")
print("Clinical table files inspected:", n_files)
print("Label candidate rows:", n_candidates)
print("Usable label sources:", n_usable_sources)
print("Automatic label template filled:", "yes" if filled else "no")
print("Automatically filled patients:", best_n)
