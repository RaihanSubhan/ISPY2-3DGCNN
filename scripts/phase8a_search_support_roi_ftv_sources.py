from pathlib import Path
import os
import re
import json
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
DATA = Path.home() / "ispy2"
REPORTS = REPO / "reports"
TABLES = REPORTS / "tables"
FIGS = REPORTS / "figures"
VIEWS = REPORTS / "table_views"
EXT = REPO / "external" / "ispy2_support_sources"

for p in [REPORTS, TABLES, FIGS, VIEWS, EXT]:
    p.mkdir(parents=True, exist_ok=True)

# Avoid committing downloaded support files.
gitignore = REPO / ".gitignore"
if gitignore.exists():
    g = gitignore.read_text()
else:
    g = ""
if "external/" not in g:
    gitignore.write_text(g.rstrip() + "\nexternal/\n")

SEARCH_ROOTS = [
    REPO / "external",
    REPO / "reports",
    DATA,
]

URLS_TO_SCAN = [
    "https://www.cancerimagingarchive.net/collection/ispy2/",
    "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230072",
    "https://portal.imaging.datacommons.cancer.gov/collections/ispy2/",
]

KEYWORDS = [
    "roi", "mask", "tumor", "tumour", "lesion", "ftv",
    "functional", "volume", "pe", "ser", "enhancement",
    "analysis", "image-analysis", "image_analysis",
    "segmentation", "parametric", "kinetic", "ktrans",
    "pharmacokinetic", "adc", "dwi", "nifti", "nii",
    "nrrd", "mha", "npz", "parquet"
]

TABLE_EXTS = [".csv", ".tsv", ".txt", ".xlsx", ".xls"]
IMAGE_EXTS = [".nii", ".nii.gz", ".nrrd", ".mha", ".mhd", ".npz", ".parquet", ".json"]
ARCHIVE_EXTS = [".zip"]
ALL_EXTS = TABLE_EXTS + IMAGE_EXTS + ARCHIVE_EXTS

def lower_name(p):
    return str(p).lower()

def keyword_hits(text):
    low = str(text).lower()
    return [k for k in KEYWORDS if k in low]

def short_name_from_url(url):
    parsed = urllib.parse.urlparse(url)
    name = Path(urllib.parse.unquote(parsed.path)).name
    if not name:
        name = "downloaded_support"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    ext = Path(name).suffix
    if not ext:
        ext = ".dat"
    return f"{Path(name).stem[:45]}_{digest}{ext}"

def fetch_html(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=45) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

def find_links(html, base):
    links = set()
    for m in re.finditer(r'href=["\']([^"\']+)["\']', html, flags=re.I):
        href = m.group(1)
        full = urllib.parse.urljoin(base, href)
        low = full.lower()
        if any(k in low for k in KEYWORDS) or any(low.split("?")[0].endswith(e) for e in ALL_EXTS):
            links.add(full)
    return sorted(links)

def download_small(url, max_mb=120):
    out = EXT / short_name_from_url(url)
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
        return None, "download_failed: " + str(e)[:120]

def extract_zip(zip_path, max_each_mb=120):
    extracted = []
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            for info in z.infolist():
                name = info.filename
                low = name.lower()
                if not any(low.endswith(e) for e in TABLE_EXTS + IMAGE_EXTS):
                    continue
                if not keyword_hits(low):
                    continue
                if info.file_size > max_each_mb * 1024 * 1024:
                    continue

                safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)
                digest = hashlib.sha1((str(zip_path) + name).encode("utf-8")).hexdigest()[:10]
                out = EXT / f"zip_{digest}_{safe}"
                with z.open(info) as src:
                    out.write_bytes(src.read())
                extracted.append(str(out))
    except Exception:
        pass
    return extracted

def should_skip_dir(path):
    low = str(path).lower()
    # Avoid walking millions of DICOM images inside MR/SEG series folders.
    base = Path(path).name.lower()
    if base.startswith("mr_") or base.startswith("seg_"):
        return True
    if "__pycache__" in low or ".git" in low:
        return True
    return False

def candidate_type(path, cols_text=""):
    low = str(path).lower() + " " + str(cols_text).lower()

    if "ftv" in low or "functional_tumor_volume" in low or "functional tumor volume" in low:
        return "FTV_candidate"
    if "pe" in low and "ser" in low:
        return "PE_SER_candidate"
    if "roi" in low:
        return "ROI_candidate"
    if "mask" in low or "segmentation" in low or "tumor" in low or "lesion" in low:
        return "mask_or_segmentation_candidate"
    if "adc" in low or "dwi" in low:
        return "DWI_ADC_candidate"
    if "clinical" in low or "pcr" in low or "response" in low:
        return "clinical_label_candidate"
    if "parquet" in low or "supervoxel" in low:
        return "supervoxel_candidate"
    return "other_support_candidate"

def read_table_preview(path):
    path = Path(path)
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, dtype=str, nrows=5).fillna("")
        elif path.suffix.lower() == ".tsv":
            df = pd.read_csv(path, sep="\t", dtype=str, nrows=5).fillna("")
        elif path.suffix.lower() == ".txt":
            df = pd.read_csv(path, sep=None, engine="python", dtype=str, nrows=5).fillna("")
        elif path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path, dtype=str, nrows=5).fillna("")
        else:
            return "", "", "not_table"
        cols = ";".join([str(c) for c in df.columns[:40]])
        values = ";".join([str(v)[:50] for v in df.head(1).values.flatten()[:20]])
        return cols, values, "read_ok"
    except Exception as e:
        return "", "", "read_failed: " + str(e)[:80]

def inspect_npz(path):
    try:
        data = np.load(path, allow_pickle=False)
        keys = list(data.keys())
        desc = []
        for k in keys[:20]:
            arr = data[k]
            desc.append(f"{k}:{arr.shape}:{arr.dtype}")
        return ";".join(desc), "npz_read_ok"
    except Exception as e:
        return "", "npz_read_failed: " + str(e)[:80]

def inspect_json(path):
    try:
        obj = json.loads(Path(path).read_text(errors="ignore"))
        if isinstance(obj, dict):
            keys = list(obj.keys())[:40]
            return ";".join(map(str, keys)), "json_read_ok"
        if isinstance(obj, list):
            return f"list_len={len(obj)}", "json_read_ok"
        return type(obj).__name__, "json_read_ok"
    except Exception as e:
        return "", "json_read_failed: " + str(e)[:80]

# Step 1: scan online pages for small support links.
page_rows = []
download_rows = []
downloaded_files = []

for url in URLS_TO_SCAN:
    html = fetch_html(url)
    links = find_links(html, url) if html else []
    page_rows.append({
        "source_url": url,
        "html_read": "yes" if html else "no",
        "candidate_links_found": len(links)
    })

    for link in links:
        hits = keyword_hits(link)
        low = link.lower()
        if not hits and not any(low.split("?")[0].endswith(e) for e in ALL_EXTS):
            continue
        out, status = download_small(link, max_mb=120)
        download_rows.append({
            "url": link,
            "keyword_hits": ";".join(hits),
            "local_file": str(out) if out else "",
            "status": status
        })
        if out:
            if out.suffix.lower() == ".zip":
                downloaded_files.extend(extract_zip(out))
            else:
                downloaded_files.append(str(out))

# Step 2: scan local support-like files.
candidate_rows = []
seen = set()

def add_candidate(path, source):
    path = Path(path)
    if str(path) in seen:
        return
    seen.add(str(path))

    low = str(path).lower()
    ext = "".join(path.suffixes[-2:]).lower() if low.endswith(".nii.gz") else path.suffix.lower()
    hits = keyword_hits(str(path))

    # Keep only support-like files. Do not include raw DICOM.
    if ext == ".dcm":
        return
    if ext not in ALL_EXTS and not hits:
        return

    try:
        size_mb = path.stat().st_size / (1024 * 1024)
    except Exception:
        size_mb = np.nan

    columns = ""
    preview = ""
    read_status = ""

    if ext in TABLE_EXTS:
        columns, preview, read_status = read_table_preview(path)
    elif ext == ".npz":
        columns, read_status = inspect_npz(path)
    elif ext == ".json":
        columns, read_status = inspect_json(path)
    else:
        read_status = "not_opened"

    ctype = candidate_type(path, columns)

    score = 0
    if ctype in ["FTV_candidate", "PE_SER_candidate", "ROI_candidate", "mask_or_segmentation_candidate", "supervoxel_candidate"]:
        score += 50
    score += len(hits) * 5
    if pd.notna(size_mb) and size_mb > 0:
        score += 1
    if read_status.endswith("read_ok") or "read_ok" in read_status:
        score += 5

    candidate_rows.append({
        "file_path": str(path),
        "source": source,
        "extension": ext,
        "size_mb": round(size_mb, 4) if pd.notna(size_mb) else "",
        "candidate_type": ctype,
        "keyword_hits": ";".join(hits),
        "read_status": read_status,
        "columns_or_keys_preview": columns,
        "first_row_preview": preview,
        "priority_score": score
    })

# downloaded/extracted files first
for f in downloaded_files:
    add_candidate(f, "downloaded_or_extracted_support")

# repo/external/report roots and careful raw data scan
for root in SEARCH_ROOTS:
    root = Path(root)
    if not root.exists():
        continue

    for dirpath, dirnames, filenames in os.walk(root):
        # prune huge raw DICOM series dirs
        pruned = []
        for d in dirnames:
            full = Path(dirpath) / d
            if should_skip_dir(full):
                continue
            pruned.append(d)
        dirnames[:] = pruned

        # limit raw data scan depth under ~/ispy2
        if root == DATA:
            rel = Path(dirpath).relative_to(root)
            if len(rel.parts) > 4:
                dirnames[:] = []
                continue

        for name in filenames:
            p = Path(dirpath) / name
            low = name.lower()
            if any(k in low for k in KEYWORDS) or any(low.endswith(e) for e in ALL_EXTS):
                add_candidate(p, "local_scan")

page_df = pd.DataFrame(page_rows)
download_df = pd.DataFrame(download_rows)
cand_df = pd.DataFrame(candidate_rows)

if not cand_df.empty:
    cand_df = cand_df.sort_values(["priority_score", "candidate_type", "size_mb"], ascending=[False, True, False])

page_df.to_csv(TABLES / "phase8a_support_page_scan.csv", index=False)
download_df.to_csv(TABLES / "phase8a_support_download_log.csv", index=False)
cand_df.to_csv(TABLES / "phase8a_support_source_candidates.csv", index=False)

# Make markdown previews.
def make_md(csv_path, md_path, title, max_rows=80, max_cols=10):
    if not Path(csv_path).exists():
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
    for _, r in small.iterrows():
        lines.append("| " + " | ".join(clean(v) for v in r.tolist()) + " |")
    md_path.write_text("\n".join(lines))

make_md(TABLES / "phase8a_support_source_candidates.csv", VIEWS / "phase8a_support_source_candidates.md", "Phase 8A Support Source Candidates")
make_md(TABLES / "phase8a_support_download_log.csv", VIEWS / "phase8a_support_download_log.md", "Phase 8A Support Download Log")

# Figures.
plt.figure(figsize=(10, 5))
if not cand_df.empty:
    cand_df["candidate_type"].value_counts().sort_values().plot(kind="barh")
    plt.title("Phase 8A Support Source Candidate Types")
    plt.xlabel("Files")
else:
    plt.text(0.5, 0.5, "No support candidates found", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "230_phase8a_candidate_types.png", dpi=220)
plt.close()

plt.figure(figsize=(10, 5))
if not cand_df.empty:
    top = cand_df.head(20).copy()
    labels = top["candidate_type"].astype(str) + " | " + top["extension"].astype(str)
    plt.barh(labels, pd.to_numeric(top["priority_score"], errors="coerce").fillna(0))
    plt.title("Phase 8A Top Support Candidates")
    plt.xlabel("Priority score")
else:
    plt.text(0.5, 0.5, "No support candidates found", ha="center", va="center")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "231_phase8a_top_candidates.png", dpi=220)
plt.close()

# Summary.
n_total = len(cand_df)
n_ftv = int((cand_df["candidate_type"] == "FTV_candidate").sum()) if n_total else 0
n_peser = int((cand_df["candidate_type"] == "PE_SER_candidate").sum()) if n_total else 0
n_roi = int((cand_df["candidate_type"] == "ROI_candidate").sum()) if n_total else 0
n_mask = int((cand_df["candidate_type"] == "mask_or_segmentation_candidate").sum()) if n_total else 0
n_super = int((cand_df["candidate_type"] == "supervoxel_candidate").sum()) if n_total else 0

summary = []
summary.append("# Phase 8A Support-Data Tumor Source Search Report")
summary.append("")
summary.append("This phase searches for official or local support-data tumor sources after Phase 7F showed that recovered DICOM SEG masks were too limited for reliable supervised ML.")
summary.append("")
summary.append("## Main outputs")
summary.append("")
summary.append("- reports/tables/phase8a_support_source_candidates.csv")
summary.append("- reports/tables/phase8a_support_page_scan.csv")
summary.append("- reports/tables/phase8a_support_download_log.csv")
summary.append("- reports/table_views/phase8a_support_source_candidates.md")
summary.append("- reports/table_views/phase8a_support_download_log.md")
summary.append("- reports/figures/230_phase8a_candidate_types.png")
summary.append("- reports/figures/231_phase8a_top_candidates.png")
summary.append("")
summary.append("## Main counts")
summary.append("")
summary.append("- Total support candidates: " + str(n_total))
summary.append("- FTV candidates: " + str(n_ftv))
summary.append("- PE/SER candidates: " + str(n_peser))
summary.append("- ROI candidates: " + str(n_roi))
summary.append("- mask/segmentation candidates: " + str(n_mask))
summary.append("- supervoxel candidates: " + str(n_super))
summary.append("")
summary.append("## Interpretation")
summary.append("")
if n_ftv + n_peser + n_roi + n_mask + n_super > 0:
    summary.append("Potential support-data sources were found. Open the candidate table and choose the highest-priority ROI, FTV, PE/SER, or supervoxel source for Phase 8B.")
else:
    summary.append("No obvious support-data tumor source was found automatically. You may need to manually download the official TCIA/IDC image-analysis support package and place it under external/ispy2_support_sources/.")
summary.append("")
summary.append("## Next step")
summary.append("")
summary.append("Phase 8B should inspect the best support-data candidate and build a real tumor-source manifest. Do not return to Phase 6 or Phase 7 ML until a stronger tumor source is identified.")

(REPORTS / "Phase8A_Support_Data_Tumor_Source_Search_Report.md").write_text("\n".join(summary))

# Dashboard and README.
dash = REPORTS / "Dashboard.md"
old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
dash_add = "\n\n## Phase 8A support-data tumor source search\n\n- [Support-data tumor source search report](Phase8A_Support_Data_Tumor_Source_Search_Report.md)\n- [Support source candidates](table_views/phase8a_support_source_candidates.md)\n"
if "Phase 8A support-data tumor source search" not in old_dash:
    dash.write_text(old_dash.rstrip() + dash_add)

readme = REPO / "README.md"
old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
readme_add = "\n\n### Phase 8A: support-data tumor source search\n\nMain outputs:\n\n- reports/Phase8A_Support_Data_Tumor_Source_Search_Report.md\n- reports/tables/phase8a_support_source_candidates.csv\n- reports/figures/230_phase8a_candidate_types.png\n"
if "Phase 8A: support-data tumor source search" not in old_readme:
    readme.write_text(old_readme.rstrip() + readme_add)

print("Phase 8A complete.")
print("Total support candidates:", n_total)
print("FTV candidates:", n_ftv)
print("PE/SER candidates:", n_peser)
print("ROI candidates:", n_roi)
print("mask/segmentation candidates:", n_mask)
print("supervoxel candidates:", n_super)
