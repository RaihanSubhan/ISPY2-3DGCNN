from pathlib import Path
import re
import os
import json
import urllib.request
import urllib.parse
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

for p in [TABLES, FIGS, VIEWS, CLIN]:
    p.mkdir(parents=True, exist_ok=True)

BIO_PATH = TABLES / "phase3d_verified_biomarker_table.csv"
TEMPLATE_PATH = TABLES / "phase4c_pcr_label_template.csv"

if not BIO_PATH.exists():
    raise SystemExit("Missing reports/tables/phase3d_verified_biomarker_table.csv")

if not TEMPLATE_PATH.exists():
    raise SystemExit("Missing reports/tables/phase4c_pcr_label_template.csv")

bio = pd.read_csv(BIO_PATH, dtype=str).fillna("")
template = pd.read_csv(TEMPLATE_PATH, dtype=str).fillna("")

patients = sorted(set(template["patient_folder"].astype(str)))

URLS_TO_SCAN = [
    "https://www.cancerimagingarchive.net/collection/ispy2/",
    "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230072",
]

KEYWORDS = [
    "clinical", "pcr", "response", "outcome", "pathologic",
    "pathological", "rcb", "treatment", "arm", "subtype"
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

def fetch_text(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return ""

def find_links(html, base_url):
    links = set()
    # href="..."
    for m in re.finditer(r'href=["\']([^"\']+)["\']', html, flags=re.I):
        href = m.group(1)
        full = urllib.parse.urljoin(base_url, href)
        low = full.lower()
        if any(k in low for k in KEYWORDS) or low.endswith((".csv", ".tsv", ".xlsx", ".xls", ".zip")):
            links.add(full)

    # also catch raw URLs
    for m in re.finditer(r'https?://[^\s"\']+', html, flags=re.I):
        full = m.group(0).rstrip(")>.,")
        low = full.lower()
        if any(k in low for k in KEYWORDS) or low.endswith((".csv", ".tsv", ".xlsx", ".xls", ".zip")):
            links.add(full)

    return sorted(links)

def safe_name_from_url(url):
    parsed = urllib.parse.urlparse(url)
    name = Path(parsed.path).name
    if not name:
        name = re.sub(r"[^A-Za-z0-9]+", "_", url)[:80] + ".html"
    name = urllib.parse.unquote(name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name

def download_small(url, out_dir, max_mb=25):
    name = safe_name_from_url(url)
    out = out_dir / name

    if out.exists() and out.stat().st_size > 0:
        return out, "already_exists"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read(max_mb * 1024 * 1024 + 1)
        if len(data) > max_mb * 1024 * 1024:
            return None, "too_large_skipped"
        out.write_bytes(data)
        return out, "downloaded"
    except Exception as e:
        return None, "download_failed: " + str(e)[:120]

def read_table(path):
    path = Path(path)
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, dtype=str).fillna("")
        if path.suffix.lower() == ".tsv":
            return pd.read_csv(path, sep="\t", dtype=str).fillna("")
        if path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(path, dtype=str).fillna("")
        if path.suffix.lower() == ".txt":
            return pd.read_csv(path, sep=None, engine="python", dtype=str).fillna("")
    except Exception:
        return None
    return None

def patient_column(df):
    best = None
    best_overlap = 0
    for c in df.columns:
        vals = set(df[c].astype(str).map(norm_patient))
        overlap = len(vals.intersection(set(patients)))
        c_low = c.lower()
        bonus = 2 if ("patient" in c_low or "subject" in c_low or "case" in c_low or c_low in ["id", "patientid"]) else 0
        score = overlap + bonus
        if score > best_overlap:
            best_overlap = score
            best = c
    return best, max(best_overlap - 2, 0)

def label_columns(df):
    good = []
    keys = ["pcr", "pathologic", "pathological", "complete", "response", "responder", "rcb", "outcome"]
    bad = ["path", "file", "folder", "series", "study", "uid"]
    for c in df.columns:
        low = c.lower()
        if any(k in low for k in keys) and not any(b in low for b in bad):
            good.append(c)
    return good

def map_label(v, col):
    s = str(v).strip().lower()
    c = str(col).lower()
    if s in ["", "nan", "none", "null", "na", "n/a", "unknown"]:
        return np.nan

    # RCB 0 is usually pCR; RCB 1/2/3 are non-pCR.
    if "rcb" in c or "residual" in c:
        if s in ["0", "0.0", "rcb0", "rcb 0", "rcb-0", "class 0"]:
            return 1
        if s in ["1", "1.0", "2", "2.0", "3", "3.0", "rcb1", "rcb2", "rcb3", "rcb-i", "rcb-ii", "rcb-iii"]:
            return 0

    neg = ["non-pcr", "non pcr", "no pcr", "not pcr", "nonresponder", "non-responder", "no", "false", "not complete"]
    pos = ["pcr", "yes", "true", "complete response", "pathologic complete", "pathological complete", "responder"]

    for w in neg:
        if w in s:
            return 0
    for w in pos:
        if w in s:
            return 1

    if s in ["1", "1.0"]:
        return 1
    if s in ["0", "0.0"]:
        return 0

    return np.nan

# 1) Scan pages for links.
scan_rows = []
download_rows = []
all_links = []

for url in URLS_TO_SCAN:
    html = fetch_text(url)
    links = find_links(html, url) if html else []
    scan_rows.append({
        "source_url": url,
        "html_read": bool(html),
        "links_found": len(links),
    })
    all_links.extend(links)

# Add already existing local files in clinical support folder.
local_files = []
for root, _, files in os.walk(CLIN):
    for f in files:
        if f.lower().endswith((".csv", ".tsv", ".txt", ".xlsx", ".xls")):
            local_files.append(str(Path(root) / f))

# Download candidate links.
downloaded_files = []
for link in sorted(set(all_links)):
    low = link.lower()
    if not any(x in low for x in [".csv", ".tsv", ".xlsx", ".xls", ".txt"] + KEYWORDS):
        continue
    out, status = download_small(link, CLIN, max_mb=25)
    download_rows.append({"url": link, "local_file": str(out) if out else "", "status": status})
    if out and out.suffix.lower() in [".csv", ".tsv", ".txt", ".xlsx", ".xls"]:
        downloaded_files.append(str(out))

all_files = list(dict.fromkeys(local_files + downloaded_files))

# 2) Inspect downloaded/local files.
label_rows = []
label_maps = []

for f in all_files:
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

    pc, overlap = patient_column(df)
    labs = label_columns(df)

    if pc is None:
        labs = []

    for lc in labs:
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

    if not labs:
        label_rows.append({
            "file_path": f,
            "read_status": "read_ok_no_label_col",
            "patient_col": pc if pc else "",
            "patient_overlap": overlap,
            "label_col": "",
            "usable_labels": 0,
            "positive": 0,
            "negative": 0,
        })

scan_df = pd.DataFrame(scan_rows)
down_df = pd.DataFrame(download_rows)
label_df = pd.DataFrame(label_rows)

scan_df.to_csv(TABLES / "phase4e_clinical_page_scan.csv", index=False)
down_df.to_csv(TABLES / "phase4e_clinical_download_log.csv", index=False)
label_df.to_csv(TABLES / "phase4e_downloaded_label_candidates.csv", index=False)

filled = False
best_n = 0

if label_maps:
    best = sorted(label_maps, key=lambda x: len(x), reverse=True)[0]
    best_n = len(best)

    updated = template.copy()
    updated = updated.drop(columns=[c for c in ["pCR_label", "label_source"] if c in updated.columns])
    updated = updated.merge(best[["patient_folder", "pCR_label", "label_source_file", "label_source_column"]], on="patient_folder", how="left")
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

# Markdown preview
def make_md(csv_path, md_path, title, max_rows=50, max_cols=10):
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    small = df.head(max_rows).iloc[:, :max_cols]
    def clean(x):
        x = str(x).replace("|", "\\|").replace("\n", " ")
        return x[:90] + "..." if len(x) > 93 else x
    lines = ["# " + title, "", "Rows: " + str(len(df)), "Columns: " + str(len(df.columns)), ""]
    lines.append("| " + " | ".join(clean(c) for c in small.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(small.columns)) + " |")
    for _, r in small.iterrows():
        lines.append("| " + " | ".join(clean(v) for v in r.tolist()) + " |")
    md_path.write_text("\n".join(lines))

make_md(TABLES / "phase4e_downloaded_label_candidates.csv", VIEWS / "phase4e_downloaded_label_candidates.md", "Phase 4E Downloaded Label Candidates")
make_md(TABLES / "phase4e_clinical_download_log.csv", VIEWS / "phase4e_clinical_download_log.md", "Phase 4E Clinical Download Log")

# Figure
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

n_files = len(all_files)
n_candidates = len(label_df)
n_usable_sources = int((label_df.get("usable_labels", pd.Series(dtype=float)).astype(float) >= 6).sum()) if len(label_df) else 0

summary = [
    "# Phase 4E Official pCR Label Acquisition Summary",
    "",
    "This phase tries to locate/download official clinical support tables and map pCR labels into the project template.",
    "",
    "## Main outputs",
    "",
    "- reports/tables/phase4e_clinical_page_scan.csv",
    "- reports/tables/phase4e_clinical_download_log.csv",
    "- reports/tables/phase4e_downloaded_label_candidates.csv",
    "- reports/table_views/phase4e_downloaded_label_candidates.md",
    "- reports/figures/100_phase4e_downloaded_label_candidates.png",
    "",
    "## Main counts",
    "",
    "- Local/downloaded clinical table files inspected: " + str(n_files),
    "- Label candidate rows created: " + str(n_candidates),
    "- Candidate sources with at least 6 usable labels: " + str(n_usable_sources),
    "- Automatic label template filled: " + ("yes" if filled else "no"),
    "- Automatically filled patients: " + str(best_n),
    "",
    "## Next step",
    "",
]

if filled:
    summary.append("The pCR label template was filled automatically. Now rerun Phase 4B to train baseline ML models.")
else:
    summary.append("No usable pCR label source was found automatically. Manually download the TCIA clinical CSV or fill reports/tables/phase4c_pcr_label_template.csv, then rerun Phase 4B.")

(REPORTS / "Phase4E_Official_PCR_Label_Acquisition_Summary.md").write_text("\n".join(summary))

# Dashboard
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
