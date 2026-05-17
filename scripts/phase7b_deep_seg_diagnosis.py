from pathlib import Path
import os
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pydicom

REPO = Path.home() / "ISPY2-3DGCNN"
REPORTS = REPO / "reports"
TABLES = REPORTS / "tables"
FIGS = REPORTS / "figures"
VIEWS = REPORTS / "table_views"

for p in [REPORTS, TABLES, FIGS, VIEWS]:
    p.mkdir(parents=True, exist_ok=True)

LINK_PATH = TABLES / "seg_mr_link_table.csv"

if not LINK_PATH.exists():
    raise SystemExit("Missing reports/tables/seg_mr_link_table.csv")

links = pd.read_csv(LINK_PATH, dtype=str).fillna("")

TUMOR_WORDS = [
    "tumor", "tumour", "lesion", "mass", "target", "cancer",
    "carcinoma", "neoplasm", "enhancing", "ftv", "roi", "volume"
]

EXCLUDE_WORDS = [
    "whole", "breast", "background", "body", "skin", "air",
    "normal", "tissue", "fibroglandular", "fgt", "organ",
    "thorax", "chest"
]

def safe_str(x):
    return "" if x is None else str(x)

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def clean_text(x):
    return safe_str(x).strip().replace("\n", " ").replace("\r", " ")

def get_code_meaning(item, seq_name):
    try:
        seq = getattr(item, seq_name, None)
        if seq and len(seq) > 0:
            return clean_text(getattr(seq[0], "CodeMeaning", ""))
    except Exception:
        pass
    return ""

def get_segment_metadata(ds):
    rows = []
    seq = getattr(ds, "SegmentSequence", None)

    if not seq:
        rows.append({
            "segment_number": "unknown",
            "segment_label": "unknown",
            "segment_description": "",
            "algorithm_type": "",
            "property_category": "",
            "property_type": "",
        })
        return rows

    for item in seq:
        rows.append({
            "segment_number": clean_text(getattr(item, "SegmentNumber", "")),
            "segment_label": clean_text(getattr(item, "SegmentLabel", "")),
            "segment_description": clean_text(getattr(item, "SegmentDescription", "")),
            "algorithm_type": clean_text(getattr(item, "SegmentAlgorithmType", "")),
            "property_category": get_code_meaning(item, "SegmentedPropertyCategoryCodeSequence"),
            "property_type": get_code_meaning(item, "SegmentedPropertyTypeCodeSequence"),
        })

    return rows

def get_frame_segment_numbers(ds, n_frames):
    out = []
    per = getattr(ds, "PerFrameFunctionalGroupsSequence", None)

    for i in range(n_frames):
        seg_num = "unknown"
        try:
            fg = per[i]
            seq = getattr(fg, "SegmentIdentificationSequence", None)
            if seq:
                seg_num = clean_text(getattr(seq[0], "ReferencedSegmentNumber", "unknown"))
        except Exception:
            pass
        out.append(seg_num)

    return out

def read_seg_pixel_array(seg_file):
    ds = pydicom.dcmread(seg_file, force=True)
    arr = np.asarray(ds.pixel_array)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 4:
        arr = np.squeeze(arr)
        if arr.ndim == 4:
            arr = arr[..., 0]

    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected SEG pixel array shape: {arr.shape}")

    return ds, arr

def classify_segment(text, max_frac, median_frac, total_frac, nonzero_frames):
    t = text.lower()

    has_tumor_word = any(w in t for w in TUMOR_WORDS)
    has_exclude_word = any(w in t for w in EXCLUDE_WORDS)

    if max_frac >= 0.95:
        return "full_slice_mask"
    if max_frac >= 0.50:
        return "large_non_tumor_likely"
    if nonzero_frames == 0:
        return "empty_segment"
    if max_frac < 0.00001:
        return "tiny_noise"
    if has_tumor_word and max_frac < 0.50:
        return "tumor_label_plausible_size"
    if has_exclude_word:
        return "exclude_label"
    if 0.00001 <= max_frac < 0.25:
        return "size_plausible_unknown_label"
    return "unclear"

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
    lines.append("| " + " | ".join(clean(c) for c in small.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(small.columns)) + " |")

    for _, row in small.iterrows():
        lines.append("| " + " | ".join(clean(v) for v in row.tolist()) + " |")

    md_path.write_text("\n".join(lines))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=300)
    args = parser.parse_args()

    work = links.copy()

    for col in ["seg_first_dcm", "patient_folder", "study_folder", "seg_series_uid"]:
        if col not in work.columns:
            raise SystemExit(f"Missing column in seg_mr_link_table.csv: {col}")

    work = work[work["seg_first_dcm"].astype(str) != ""].copy()

    if "link_confidence" in work.columns:
        priority = {"high": 0, "medium": 1, "weak": 2}
        work["link_priority"] = work["link_confidence"].map(priority).fillna(9)
        work = work.sort_values(["link_priority", "patient_folder"])
    else:
        work = work.sort_values("patient_folder")

    work = work.head(args.max_cases)

    case_rows = []
    segment_rows = []

    for _, row in work.iterrows():
        case_no = len(case_rows) + 1

        patient = row.get("patient_folder", "")
        study = row.get("study_folder", "")
        seg_uid = row.get("seg_series_uid", "")
        seg_file = row.get("seg_first_dcm", "")

        case = {
            "case_no": case_no,
            "patient_folder": patient,
            "study_folder": study,
            "seg_series_uid": seg_uid,
            "seg_file": seg_file,
            "link_confidence": row.get("link_confidence", ""),
            "diagnosis_status": "not_started",
        }

        try:
            ds, arr = read_seg_pixel_array(seg_file)
            mask = arr > 0

            n_frames, rows, cols = mask.shape
            frame_area = rows * cols
            frame_seg_nums = get_frame_segment_numbers(ds, n_frames)
            segment_meta = get_segment_metadata(ds)
            meta_by_num = {str(x["segment_number"]): x for x in segment_meta}

            unique_values = np.unique(arr[:min(10, n_frames)])
            unique_values_text = ";".join([str(v) for v in unique_values[:20]])

            case.update({
                "segmentation_type": clean_text(getattr(ds, "SegmentationType", "")),
                "series_description": clean_text(getattr(ds, "SeriesDescription", "")),
                "content_label": clean_text(getattr(ds, "ContentLabel", "")),
                "content_description": clean_text(getattr(ds, "ContentDescription", "")),
                "n_frames": int(n_frames),
                "rows": int(rows),
                "cols": int(cols),
                "frame_area": int(frame_area),
                "pixel_unique_values_first_frames": unique_values_text,
                "n_segment_sequence_items": len(segment_meta),
                "n_frame_segment_numbers": len(set(frame_seg_nums)),
            })

            if all(x == "unknown" for x in frame_seg_nums):
                segment_numbers = ["unknown"]
            else:
                segment_numbers = sorted(set(frame_seg_nums))

            segment_classifications = []
            plausible_count = 0
            full_count = 0

            for seg_num in segment_numbers:
                if seg_num == "unknown":
                    idx = list(range(n_frames))
                else:
                    idx = [i for i, s in enumerate(frame_seg_nums) if str(s) == str(seg_num)]

                seg_mask = np.zeros_like(mask, dtype=bool)
                seg_mask[idx] = mask[idx]

                per_frame_pixels = seg_mask.reshape(n_frames, -1).sum(axis=1)
                nz = per_frame_pixels[per_frame_pixels > 0]

                nonzero_frames = int((per_frame_pixels > 0).sum())
                max_pixels = int(per_frame_pixels.max()) if len(per_frame_pixels) else 0
                median_pixels = float(np.median(nz)) if len(nz) else 0.0

                max_frac = float(max_pixels / max(frame_area, 1))
                median_frac = float(median_pixels / max(frame_area, 1))
                total_frac = float(seg_mask.sum() / max(seg_mask.size, 1))

                meta = meta_by_num.get(str(seg_num), {})
                label = meta.get("segment_label", "")
                desc = meta.get("segment_description", "")
                prop_cat = meta.get("property_category", "")
                prop_type = meta.get("property_type", "")

                text = " ".join([label, desc, prop_cat, prop_type])
                seg_class = classify_segment(text, max_frac, median_frac, total_frac, nonzero_frames)

                if seg_class in ["tumor_label_plausible_size", "size_plausible_unknown_label"]:
                    plausible_count += 1

                if seg_class == "full_slice_mask":
                    full_count += 1

                segment_classifications.append(seg_class)

                segment_rows.append({
                    "case_no": case_no,
                    "patient_folder": patient,
                    "study_folder": study,
                    "seg_series_uid": seg_uid,
                    "segment_number": seg_num,
                    "segment_label": label,
                    "segment_description": desc,
                    "property_category": prop_cat,
                    "property_type": prop_type,
                    "algorithm_type": meta.get("algorithm_type", ""),
                    "nonzero_frames": nonzero_frames,
                    "max_frame_pixels": max_pixels,
                    "median_nonzero_frame_pixels": median_pixels,
                    "max_frame_fraction": max_frac,
                    "median_nonzero_frame_fraction": median_frac,
                    "total_mask_fraction_3d": total_frac,
                    "segment_classification": seg_class,
                    "seg_file": seg_file,
                })

            case["plausible_segment_count"] = plausible_count
            case["full_slice_segment_count"] = full_count
            case["segment_classifications"] = ";".join(sorted(set(segment_classifications)))

            if plausible_count > 0:
                case["diagnosis_status"] = "has_plausible_tumor_like_segment"
            elif full_count == len(segment_numbers):
                case["diagnosis_status"] = "only_full_slice_segments"
            else:
                case["diagnosis_status"] = "no_tumor_like_segment_found"

        except Exception as e:
            case["diagnosis_status"] = "failed_exception"
            case["error"] = str(e)[:300]

        case_rows.append(case)

    case_df = pd.DataFrame(case_rows)
    seg_df = pd.DataFrame(segment_rows)

    case_csv = TABLES / "phase7b_seg_case_deep_diagnosis.csv"
    seg_csv = TABLES / "phase7b_seg_segment_deep_diagnosis.csv"

    case_df.to_csv(case_csv, index=False)
    seg_df.to_csv(seg_csv, index=False)

    make_md(case_csv, VIEWS / "phase7b_seg_case_deep_diagnosis.md", "Phase 7B SEG Case Deep Diagnosis")
    make_md(seg_csv, VIEWS / "phase7b_seg_segment_deep_diagnosis.md", "Phase 7B SEG Segment Deep Diagnosis")

    # Figure 180: case diagnosis status
    plt.figure(figsize=(9, 5))
    if len(case_df):
        case_df["diagnosis_status"].value_counts().sort_values().plot(kind="barh")
        plt.title("Phase 7B SEG Case Diagnosis Status")
        plt.xlabel("Number of cases")
    else:
        plt.text(0.5, 0.5, "No cases diagnosed", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "180_phase7b_case_diagnosis_status.png", dpi=220)
    plt.close()

    # Figure 181: segment classification status
    plt.figure(figsize=(9, 5))
    if len(seg_df):
        seg_df["segment_classification"].value_counts().sort_values().plot(kind="barh")
        plt.title("Phase 7B Segment Classification")
        plt.xlabel("Number of segments")
    else:
        plt.text(0.5, 0.5, "No segments diagnosed", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "181_phase7b_segment_classification.png", dpi=220)
    plt.close()

    # Figure 182: mask fraction distribution
    plt.figure(figsize=(8, 5))
    vals = pd.to_numeric(seg_df.get("max_frame_fraction", pd.Series([])), errors="coerce").dropna()
    if len(vals):
        vals.plot(kind="hist", bins=40)
        plt.axvline(0.50, linestyle="--")
        plt.title("Phase 7B Max Frame Mask Fraction")
        plt.xlabel("Largest frame mask fraction")
        plt.ylabel("Segments")
    else:
        plt.text(0.5, 0.5, "No mask fraction values", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "182_phase7b_mask_fraction_distribution.png", dpi=220)
    plt.close()

    # Figure 183: labels
    plt.figure(figsize=(10, 7))
    if len(seg_df):
        labels = seg_df["segment_label"].replace("", "blank").fillna("blank")
        labels.value_counts().head(20).sort_values().plot(kind="barh")
        plt.title("Phase 7B Top SEG Segment Labels")
        plt.xlabel("Count")
    else:
        plt.text(0.5, 0.5, "No segment labels", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "183_phase7b_top_segment_labels.png", dpi=220)
    plt.close()

    n_cases = len(case_df)
    n_plausible_cases = int((case_df["diagnosis_status"] == "has_plausible_tumor_like_segment").sum()) if n_cases else 0
    n_only_full = int((case_df["diagnosis_status"] == "only_full_slice_segments").sum()) if n_cases else 0
    n_failed = int((case_df["diagnosis_status"] == "failed_exception").sum()) if n_cases else 0

    n_segments = len(seg_df)
    n_plausible_segments = int(seg_df["segment_classification"].isin(["tumor_label_plausible_size", "size_plausible_unknown_label"]).sum()) if n_segments else 0
    n_full_segments = int((seg_df["segment_classification"] == "full_slice_mask").sum()) if n_segments else 0

    report = []
    report.append("# Phase 7B Deep SEG Diagnosis Report")
    report.append("")
    report.append("This phase diagnoses whether the downloaded DICOM SEG files contain real tumor-only segments or mostly full-slice masks.")
    report.append("")
    report.append("## Main outputs")
    report.append("")
    report.append("- reports/tables/phase7b_seg_case_deep_diagnosis.csv")
    report.append("- reports/tables/phase7b_seg_segment_deep_diagnosis.csv")
    report.append("- reports/table_views/phase7b_seg_case_deep_diagnosis.md")
    report.append("- reports/table_views/phase7b_seg_segment_deep_diagnosis.md")
    report.append("- reports/figures/180_phase7b_case_diagnosis_status.png")
    report.append("- reports/figures/181_phase7b_segment_classification.png")
    report.append("- reports/figures/182_phase7b_mask_fraction_distribution.png")
    report.append("- reports/figures/183_phase7b_top_segment_labels.png")
    report.append("")
    report.append("## Main counts")
    report.append("")
    report.append("- SEG cases diagnosed: " + str(n_cases))
    report.append("- Cases with plausible tumor-like segment: " + str(n_plausible_cases))
    report.append("- Cases with only full-slice segments: " + str(n_only_full))
    report.append("- Failed cases: " + str(n_failed))
    report.append("- Segment rows diagnosed: " + str(n_segments))
    report.append("- Plausible tumor-like segments: " + str(n_plausible_segments))
    report.append("- Full-slice segments: " + str(n_full_segments))
    report.append("")
    report.append("## Interpretation")
    report.append("")

    if n_plausible_cases > 0:
        report.append("Some plausible tumor-like segments were found. The next step should create overlays for those specific segments and use only those for tumor feature extraction.")
    else:
        report.append("No plausible tumor-like DICOM SEG segments were found in this diagnostic sample. This means the current SEG files are probably not tumor masks, or the tumor masks are stored in a different support-data file or different DICOM series.")
        report.append("")
        report.append("Do not use the current SEG files for tumor volume, vascular density, or graph-node construction until a real tumor mask source is identified.")

    report.append("")
    report.append("## Next step")
    report.append("")
    report.append("If plausible segments exist, run a Phase 7C overlay builder for those segments. If none exist, search the ISPY2 support data for tumor masks, FTV maps, PE/SER maps, or ROI files instead of using the current SEG files.")

    (REPORTS / "Phase7B_Deep_SEG_Diagnosis_Report.md").write_text("\n".join(report))

    # Dashboard update
    dash = REPORTS / "Dashboard.md"
    old_dash = dash.read_text() if dash.exists() else "# ISPY2 4D Atlas Dashboard\n"
    dash_add = "\n\n## Phase 7B deep SEG diagnosis\n\n- [Deep SEG diagnosis report](Phase7B_Deep_SEG_Diagnosis_Report.md)\n- [SEG case diagnosis](table_views/phase7b_seg_case_deep_diagnosis.md)\n- [SEG segment diagnosis](table_views/phase7b_seg_segment_deep_diagnosis.md)\n"
    if "Phase 7B deep SEG diagnosis" not in old_dash:
        dash.write_text(old_dash.rstrip() + dash_add)

    readme = REPO / "README.md"
    old_readme = readme.read_text() if readme.exists() else "# ISPY2-3DGCNN\n"
    readme_add = "\n\n### Phase 7B: deep SEG diagnosis\n\nMain outputs:\n\n- reports/Phase7B_Deep_SEG_Diagnosis_Report.md\n- reports/tables/phase7b_seg_case_deep_diagnosis.csv\n- reports/tables/phase7b_seg_segment_deep_diagnosis.csv\n"
    if "Phase 7B: deep SEG diagnosis" not in old_readme:
        readme.write_text(old_readme.rstrip() + readme_add)

    print("Phase 7B complete.")
    print("SEG cases diagnosed:", n_cases)
    print("Cases with plausible tumor-like segment:", n_plausible_cases)
    print("Cases with only full-slice segments:", n_only_full)
    print("Failed cases:", n_failed)
    print("Plausible tumor-like segments:", n_plausible_segments)
    print("Full-slice segments:", n_full_segments)

if __name__ == "__main__":
    main()
