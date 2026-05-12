#!/usr/bin/env bash
set -euo pipefail

REPO="$HOME/ISPY2-3DGCNN"
DATA_ROOT="$HOME/ispy2/ispy2"

echo "Checking repo..."
if [ ! -d "$REPO" ]; then
  echo "ERROR: Repo folder not found: $REPO"
  exit 1
fi

echo "Checking ISPY2 data..."
if [ ! -d "$DATA_ROOT" ]; then
  echo "ERROR: ISPY2 data folder not found: $DATA_ROOT"
  exit 1
fi

source /home/mdsubhan01/miniforge3/etc/profile.d/conda.sh
conda activate 3dgcnn

cd "$REPO"

mkdir -p scripts configs reports/figures reports/tables reports/logs

# Keep logs out of GitHub
grep -qxF "reports/logs/" .gitignore || echo "reports/logs/" >> .gitignore
grep -qxF "*.tmp" .gitignore || echo "*.tmp" >> .gitignore

cat > configs/phase2_seg_mr_linking_config.yaml <<'EOF'
project_name: ISPY2_4D_Vascular_Perfusion_Tumor_Atlas
phase: phase_2_seg_mr_linking
repo_root: /home/mdsubhan01/ISPY2-3DGCNN
data_root: /home/mdsubhan01/ispy2/ispy2
input_inventory: reports/tables/ispy2_series_inventory.csv
input_candidate_mr: reports/tables/candidate_mr_series_best_per_patient.csv
main_goal: >
  Link verified DICOM SEG masks with MR visits and create quality-control tables.
  This is the required bridge between Phase 1 cohort inventory and later tumor/perfusion biomarkers.
outputs:
  - SEG metadata table
  - SEG-to-MR linking table
  - SEG mask QC features
  - figures for SEG availability, link status, segment labels, and decoded mask projection
important_limitation: >
  Decoded DICOM SEG frame counts are quality-control features. Final tumor volume features
  should be computed after spatial registration and verified MR-SEG alignment.
EOF

cat > scripts/phase2_seg_mr_linking.py <<'PY'
#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm

try:
    import pydicom
except Exception as e:
    raise SystemExit(f"pydicom import failed: {e}")


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


def read_meta_dataset(path: str):
    return pydicom.dcmread(path, stop_before_pixels=True, force=True)


def collect_series_uids(ds) -> List[str]:
    found = set()

    def walk(obj):
        try:
            for elem in obj:
                if getattr(elem, "keyword", "") == "SeriesInstanceUID":
                    found.add(safe_str(elem.value))
                if getattr(elem, "VR", "") == "SQ":
                    try:
                        for child in elem.value:
                            walk(child)
                    except Exception:
                        pass
        except Exception:
            pass

    walk(ds)
    own_uid = safe_str(getattr(ds, "SeriesInstanceUID", ""))
    found.discard(own_uid)
    return sorted([x for x in found if x])


def get_segment_sequence(ds) -> Tuple[str, str]:
    labels = []
    numbers = []

    seq = getattr(ds, "SegmentSequence", None)
    if not seq:
        return "", ""

    for item in seq:
        number = safe_str(getattr(item, "SegmentNumber", ""))
        label = safe_str(getattr(item, "SegmentLabel", ""))
        algorithm = safe_str(getattr(item, "SegmentAlgorithmType", ""))

        if number:
            numbers.append(number)

        if label or number:
            labels.append(f"{number}:{label}:{algorithm}")

    return ";".join(numbers), ";".join(labels)


def extract_seg_metadata(seg_df: pd.DataFrame, out_tables: Path) -> pd.DataFrame:
    rows = []

    for _, r in tqdm(seg_df.iterrows(), total=len(seg_df), desc="Reading SEG metadata"):
        first_dcm = safe_str(r.get("first_dcm", ""))
        row = {
            "patient_folder": safe_str(r.get("patient_folder", "")),
            "study_folder": safe_str(r.get("study_folder", "")),
            "series_folder": safe_str(r.get("series_folder", "")),
            "series_path": safe_str(r.get("series_path", "")),
            "first_dcm": first_dcm,
            "seg_series_uid_inventory": safe_str(r.get("SeriesInstanceUID", "")),
            "seg_study_uid_inventory": safe_str(r.get("StudyInstanceUID", "")),
            "dicom_read_status": "not_read",
        }

        try:
            ds = read_meta_dataset(first_dcm)

            own_uid = safe_str(getattr(ds, "SeriesInstanceUID", ""))
            study_uid = safe_str(getattr(ds, "StudyInstanceUID", ""))
            patient_id = safe_str(getattr(ds, "PatientID", ""))
            sop_uid = safe_str(getattr(ds, "SOPInstanceUID", ""))
            frame_uid = safe_str(getattr(ds, "FrameOfReferenceUID", ""))
            rows_value = safe_str(getattr(ds, "Rows", ""))
            cols_value = safe_str(getattr(ds, "Columns", ""))
            frames_value = safe_str(getattr(ds, "NumberOfFrames", ""))
            content_date = safe_str(getattr(ds, "ContentDate", ""))
            content_time = safe_str(getattr(ds, "ContentTime", ""))

            seg_nums, seg_labels = get_segment_sequence(ds)
            ref_uids = collect_series_uids(ds)

            row.update({
                "dicom_read_status": "success",
                "PatientID": patient_id,
                "SOPInstanceUID": sop_uid,
                "seg_series_uid": own_uid,
                "seg_study_uid": study_uid,
                "FrameOfReferenceUID": frame_uid,
                "Rows": rows_value,
                "Columns": cols_value,
                "NumberOfFrames": frames_value,
                "ContentDate": content_date,
                "ContentTime": content_time,
                "segment_numbers": seg_nums,
                "segment_labels": seg_labels,
                "referenced_series_uids": ";".join(ref_uids),
                "n_referenced_series_uids": len(ref_uids),
            })

        except Exception as e:
            row.update({
                "dicom_read_status": "failed",
                "read_error": safe_str(e)[:300],
            })

        rows.append(row)

    meta = pd.DataFrame(rows)
    meta.to_csv(out_tables / "seg_series_metadata.csv", index=False)
    return meta


def score_candidate_series(row: pd.Series) -> float:
    desc = (safe_str(row.get("SeriesDescription", "")) + " " + safe_str(row.get("ProtocolName", ""))).lower()

    positive = ["dce", "dynamic", "dyn", "vibrant", "post", "pre", "contrast", "sub", "t1", "fs", "ax", "sag"]
    negative = ["localizer", "scout", "calibration", "survey", "asset", "mip"]

    score = 0.0
    for kw in positive:
        if kw in desc:
            score += 2.0
    for kw in negative:
        if kw in desc:
            score -= 5.0

    try:
        count = float(row.get("dcm_count", 0))
    except Exception:
        count = 0.0

    score += min(np.log1p(count), 10.0)

    if safe_str(row.get("Modality", "")).upper() == "MR":
        score += 5.0

    return score


def split_semicolon(x: str) -> List[str]:
    if not isinstance(x, str) or not x.strip():
        return []
    return [i.strip() for i in x.split(";") if i.strip()]


def build_seg_mr_links(
    inventory: pd.DataFrame,
    seg_meta: pd.DataFrame,
    candidate_best: pd.DataFrame,
    out_tables: Path,
) -> pd.DataFrame:

    inv = inventory.copy()
    inv["Modality"] = inv["Modality"].fillna("").astype(str)
    mr = inv[inv["Modality"].str.upper().eq("MR")].copy()

    if "candidate_score" not in mr.columns:
        mr["candidate_score"] = mr.apply(score_candidate_series, axis=1)

    mr_by_uid = {}
    for _, r in mr.iterrows():
        uid = safe_str(r.get("SeriesInstanceUID", ""))
        if uid and uid not in mr_by_uid:
            mr_by_uid[uid] = r

    candidate_by_patient = {}
    if candidate_best is not None and not candidate_best.empty:
        for _, r in candidate_best.iterrows():
            candidate_by_patient[safe_str(r.get("patient_folder", ""))] = r

    links = []

    for _, s in seg_meta.iterrows():
        patient = safe_str(s.get("patient_folder", ""))
        study = safe_str(s.get("study_folder", ""))
        ref_uids = split_semicolon(safe_str(s.get("referenced_series_uids", "")))

        matched = None
        link_status = "unmatched"
        link_confidence = "low"

        for uid in ref_uids:
            if uid in mr_by_uid:
                matched = mr_by_uid[uid]
                link_status = "referenced_series_uid_match"
                link_confidence = "high"
                break

        if matched is None:
            same_study = mr[
                (mr["patient_folder"].astype(str) == patient)
                & (mr["study_folder"].astype(str) == study)
            ].copy()

            if not same_study.empty:
                same_study = same_study.sort_values(["candidate_score", "dcm_count"], ascending=[False, False])
                matched = same_study.iloc[0]
                link_status = "same_patient_same_study_best_mr"
                link_confidence = "medium"

        if matched is None and patient in candidate_by_patient:
            matched = candidate_by_patient[patient]
            link_status = "same_patient_best_candidate_mr"
            link_confidence = "weak"

        row = {
            "patient_folder": patient,
            "study_folder": study,
            "seg_series_path": safe_str(s.get("series_path", "")),
            "seg_first_dcm": safe_str(s.get("first_dcm", "")),
            "seg_series_uid": safe_str(s.get("seg_series_uid", "")),
            "seg_study_uid": safe_str(s.get("seg_study_uid", "")),
            "seg_frame_of_reference_uid": safe_str(s.get("FrameOfReferenceUID", "")),
            "seg_rows": safe_str(s.get("Rows", "")),
            "seg_columns": safe_str(s.get("Columns", "")),
            "seg_number_of_frames": safe_str(s.get("NumberOfFrames", "")),
            "segment_labels": safe_str(s.get("segment_labels", "")),
            "referenced_series_uids": ";".join(ref_uids),
            "link_status": link_status,
            "link_confidence": link_confidence,
        }

        if matched is not None:
            row.update({
                "matched_mr_series_uid": safe_str(matched.get("SeriesInstanceUID", "")),
                "matched_mr_study_uid": safe_str(matched.get("StudyInstanceUID", "")),
                "matched_mr_series_path": safe_str(matched.get("series_path", "")),
                "matched_mr_first_dcm": safe_str(matched.get("first_dcm", "")),
                "matched_mr_series_description": safe_str(matched.get("SeriesDescription", "")),
                "matched_mr_protocol_name": safe_str(matched.get("ProtocolName", "")),
                "matched_mr_dcm_count": safe_str(matched.get("dcm_count", "")),
            })
        else:
            row.update({
                "matched_mr_series_uid": "",
                "matched_mr_study_uid": "",
                "matched_mr_series_path": "",
                "matched_mr_first_dcm": "",
                "matched_mr_series_description": "",
                "matched_mr_protocol_name": "",
                "matched_mr_dcm_count": "",
            })

        links.append(row)

    link_df = pd.DataFrame(links)
    link_df.to_csv(out_tables / "seg_mr_link_table.csv", index=False)
    return link_df


def get_pixel_measures(ds) -> Tuple[float, float, float]:
    row_spacing = 1.0
    col_spacing = 1.0
    slice_thickness = 1.0

    try:
        pm = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
        ps = getattr(pm, "PixelSpacing", [1.0, 1.0])
        row_spacing = float(ps[0])
        col_spacing = float(ps[1])
        slice_thickness = float(getattr(pm, "SliceThickness", 1.0) or 1.0)
    except Exception:
        pass

    return row_spacing, col_spacing, slice_thickness


def frame_segment_numbers(ds, n_frames: int) -> List[str]:
    nums = []

    try:
        per = getattr(ds, "PerFrameFunctionalGroupsSequence", [])
        for i in range(n_frames):
            num = ""
            try:
                fg = per[i]
                seq = getattr(fg, "SegmentIdentificationSequence", None)
                if seq:
                    num = safe_str(getattr(seq[0], "ReferencedSegmentNumber", ""))
            except Exception:
                num = ""
            nums.append(num)
    except Exception:
        nums = [""] * n_frames

    if len(nums) < n_frames:
        nums.extend([""] * (n_frames - len(nums)))

    return nums[:n_frames]


def decode_seg_qc(seg_meta: pd.DataFrame, out_tables: Path, out_figures: Path, decode_limit: int = 120):
    records = []
    demo_payload = None

    readable = seg_meta[seg_meta["dicom_read_status"].eq("success")].copy()

    if decode_limit and decode_limit > 0:
        readable = readable.head(decode_limit)

    for _, r in tqdm(readable.iterrows(), total=len(readable), desc="Decoding SEG sample"):
        first_dcm = safe_str(r.get("first_dcm", ""))
        base = {
            "patient_folder": safe_str(r.get("patient_folder", "")),
            "study_folder": safe_str(r.get("study_folder", "")),
            "seg_series_uid": safe_str(r.get("seg_series_uid", "")),
            "seg_series_path": safe_str(r.get("series_path", "")),
            "first_dcm": first_dcm,
            "decode_status": "not_started",
        }

        try:
            ds = pydicom.dcmread(first_dcm, force=True)
            arr = np.asarray(ds.pixel_array)
            arr = np.squeeze(arr)

            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]
            elif arr.ndim == 4:
                arr = np.squeeze(arr)
                if arr.ndim == 4:
                    arr = arr[..., 0]

            if arr.ndim != 3:
                raise RuntimeError(f"Unexpected pixel_array shape after squeeze: {arr.shape}")

            mask = arr > 0
            n_frames = int(mask.shape[0])
            rows = int(mask.shape[1])
            cols = int(mask.shape[2])

            seg_nums = frame_segment_numbers(ds, n_frames)
            row_spacing, col_spacing, slice_thickness = get_pixel_measures(ds)
            voxel_proxy_mm3 = row_spacing * col_spacing * slice_thickness

            frame_nonzero = mask.reshape(n_frames, -1).sum(axis=1)

            by_seg: Dict[str, int] = {}
            by_seg_frames: Dict[str, int] = {}

            for i, vox in enumerate(frame_nonzero):
                seg_num = seg_nums[i] if i < len(seg_nums) and seg_nums[i] else "unknown"
                by_seg[seg_num] = by_seg.get(seg_num, 0) + int(vox)
                by_seg_frames[seg_num] = by_seg_frames.get(seg_num, 0) + int(vox > 0)

            total_mask_pixels = int(mask.sum())
            nonzero_frames = int((frame_nonzero > 0).sum())

            rec = dict(base)
            rec.update({
                "decode_status": "success",
                "n_frames": n_frames,
                "rows": rows,
                "columns": cols,
                "nonzero_frames": nonzero_frames,
                "total_mask_pixels": total_mask_pixels,
                "row_spacing_mm": row_spacing,
                "col_spacing_mm": col_spacing,
                "slice_thickness_mm": slice_thickness,
                "volume_proxy_mm3": float(total_mask_pixels * voxel_proxy_mm3),
                "segment_pixel_counts": ";".join([f"{k}:{v}" for k, v in sorted(by_seg.items())]),
                "segment_nonzero_frame_counts": ";".join([f"{k}:{v}" for k, v in sorted(by_seg_frames.items())]),
            })
            records.append(rec)

            if demo_payload is None and total_mask_pixels > 0:
                demo_payload = {
                    "patient": rec["patient_folder"],
                    "study": rec["study_folder"],
                    "seg_uid": rec["seg_series_uid"],
                    "mask_mip": mask.max(axis=0).astype(float),
                    "mask_sum": mask.sum(axis=0).astype(float),
                    "frame_nonzero": frame_nonzero.astype(float),
                    "segment_labels": safe_str(r.get("segment_labels", "")),
                }

        except Exception as e:
            rec = dict(base)
            rec.update({
                "decode_status": "failed",
                "decode_error": safe_str(e)[:400],
            })
            records.append(rec)

    qc = pd.DataFrame(records)
    qc.to_csv(out_tables / "seg_mask_qc_features.csv", index=False)

    make_seg_qc_figure(qc, demo_payload, out_figures)

    return qc


def make_seg_qc_figure(qc: pd.DataFrame, demo_payload, out_figures: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    if qc.empty:
        axes[0].text(0.5, 0.5, "No SEG decode attempts", ha="center", va="center")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_figures / "13_seg_qc_projection_demo.png", dpi=220)
        plt.close()
        return

    status_counts = qc["decode_status"].fillna("unknown").value_counts()
    axes[0].bar(status_counts.index.astype(str), status_counts.values)
    axes[0].set_title("SEG Decode QC Status")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=20)

    success = qc[qc["decode_status"].eq("success")].copy()
    if not success.empty:
        success["total_mask_pixels"] = pd.to_numeric(success["total_mask_pixels"], errors="coerce")
        axes[1].hist(success["total_mask_pixels"].dropna(), bins=30)
        axes[1].set_title("SEG Mask Pixel Count Distribution")
        axes[1].set_xlabel("Mask pixels")
        axes[1].set_ylabel("Count")
    else:
        axes[1].text(0.5, 0.5, "No successful decode", ha="center", va="center")
        axes[1].axis("off")

    if demo_payload is not None:
        axes[2].imshow(demo_payload["mask_mip"], cmap="gray")
        axes[2].set_title("Decoded SEG Mask MIP")
        axes[2].axis("off")

        axes[3].plot(demo_payload["frame_nonzero"])
        axes[3].set_title("Mask Area by SEG Frame")
        axes[3].set_xlabel("Frame")
        axes[3].set_ylabel("Nonzero pixels")
    else:
        axes[2].text(0.5, 0.5, "No decoded mask projection", ha="center", va="center")
        axes[2].axis("off")
        axes[3].text(0.5, 0.5, "No frame profile", ha="center", va="center")
        axes[3].axis("off")

    plt.suptitle("Phase 2 SEG Mask Quality-Control Demo")
    plt.tight_layout()
    plt.savefig(out_figures / "13_seg_qc_projection_demo.png", dpi=220)
    plt.close()


def make_phase2_figures(seg_meta: pd.DataFrame, links: pd.DataFrame, qc: pd.DataFrame, out_figures: Path) -> None:
    patient_seg = seg_meta.groupby("patient_folder").size().reset_index(name="n_seg_series")

    plt.figure(figsize=(7, 4))
    patient_seg["n_seg_series"].plot(kind="hist", bins=30)
    plt.title("SEG Series Count per Patient")
    plt.xlabel("SEG series count")
    plt.ylabel("Number of patients")
    plt.tight_layout()
    plt.savefig(out_figures / "10_seg_series_per_patient.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 4))
    counts = links["link_status"].fillna("unknown").value_counts().sort_values()
    counts.plot(kind="barh")
    plt.title("SEG-to-MR Link Status")
    plt.xlabel("Number of SEG series")
    plt.tight_layout()
    plt.savefig(out_figures / "11_seg_mr_link_status.png", dpi=220)
    plt.close()

    labels = []
    for text in seg_meta["segment_labels"].fillna("").astype(str):
        for item in text.split(";"):
            item = item.strip()
            if not item:
                continue
            parts = item.split(":")
            label = parts[1] if len(parts) > 1 and parts[1] else item
            labels.append(label)

    plt.figure(figsize=(9, 6))
    if labels:
        label_counts = pd.Series(labels).value_counts().head(20).sort_values()
        label_counts.plot(kind="barh")
        plt.title("Top SEG Segment Labels")
        plt.xlabel("Count")
    else:
        plt.text(0.5, 0.5, "No segment labels found", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_figures / "12_top_seg_segment_labels.png", dpi=220)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.axis("off")
    txt = (
        "Phase 2 pipeline\n\n"
        "DICOM SEG metadata\n"
        "        ↓\n"
        "Referenced SeriesInstanceUID matching\n"
        "        ↓\n"
        "Same-study MR fallback matching\n"
        "        ↓\n"
        "SEG pixel decode QC sample\n"
        "        ↓\n"
        "Verified MR-SEG linking table\n"
        "        ↓\n"
        "Next: tumor, peritumor, vesselness, and longitudinal delta features"
    )
    plt.text(0.5, 0.5, txt, ha="center", va="center", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_figures / "14_phase2_pipeline_diagram.png", dpi=220)
    plt.close()


def make_summary(out_dir: Path, seg_meta: pd.DataFrame, links: pd.DataFrame, qc: pd.DataFrame) -> None:
    figures = sorted((out_dir / "figures").glob("1*.png"))
    tables = [
        out_dir / "tables" / "seg_series_metadata.csv",
        out_dir / "tables" / "seg_mr_link_table.csv",
        out_dir / "tables" / "seg_mask_qc_features.csv",
    ]

    n_seg = len(seg_meta)
    n_seg_patients = seg_meta["patient_folder"].nunique() if not seg_meta.empty else 0
    n_high = int((links["link_confidence"] == "high").sum()) if not links.empty else 0
    n_med = int((links["link_confidence"] == "medium").sum()) if not links.empty else 0
    n_weak = int((links["link_confidence"] == "weak").sum()) if not links.empty else 0
    n_unmatched = int((links["link_status"] == "unmatched").sum()) if not links.empty else 0
    n_decoded = int((qc["decode_status"] == "success").sum()) if not qc.empty else 0
    n_failed = int((qc["decode_status"] == "failed").sum()) if not qc.empty else 0

    lines = []
    lines.append("# Phase 2 Output Summary")
    lines.append("")
    lines.append("This phase connects DICOM SEG masks with MR visits for the 4D vascular-perfusion tumor atlas.")
    lines.append("")
    lines.append("## Main results")
    lines.append("")
    lines.append(f"- SEG series found: {n_seg}")
    lines.append(f"- Patients with SEG series: {n_seg_patients}")
    lines.append(f"- High-confidence referenced MR links: {n_high}")
    lines.append(f"- Medium-confidence same-study MR links: {n_med}")
    lines.append(f"- Weak same-patient candidate MR links: {n_weak}")
    lines.append(f"- Unmatched SEG series: {n_unmatched}")
    lines.append(f"- Successfully decoded SEG masks in QC sample: {n_decoded}")
    lines.append(f"- Failed SEG decode attempts in QC sample: {n_failed}")
    lines.append("")
    lines.append("## Tables")
    for t in tables:
        if t.exists():
            lines.append(f"- `{t}`")
    lines.append("")
    lines.append("## Figures")
    for f in figures:
        lines.append(f"- `{f}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Phase 2 creates the verified bridge between raw DICOM data and tumor-level feature extraction.")
    lines.append("High-confidence links use referenced DICOM SeriesInstanceUID information.")
    lines.append("Medium-confidence links use the best MR series from the same patient and same study.")
    lines.append("")
    lines.append("## Important limitation")
    lines.append("")
    lines.append("This phase does not yet compute final clinical tumor volumes.")
    lines.append("DICOM SEG frames must be spatially aligned with matched MR images before final tumor, peritumor, and vascular features are reported.")
    lines.append("")
    lines.append("## Next phase")
    lines.append("")
    lines.append("Phase 3 should compute verified tumor and peritumor features from the linked MR-SEG pairs, including shape, radial profile, vesselness, and enhancement statistics.")

    (out_dir / "Phase2_SEG_MR_Linking_Summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--decode-limit", type=int, default=120)
    args = parser.parse_args()

    repo = Path(args.repo_root).expanduser().resolve()
    out_dir = repo / "reports"
    out_tables = out_dir / "tables"
    out_figures = out_dir / "figures"

    inv_path = out_tables / "ispy2_series_inventory.csv"
    cand_path = out_tables / "candidate_mr_series_best_per_patient.csv"

    if not inv_path.exists():
        raise SystemExit(f"Missing Phase 1 inventory table: {inv_path}")

    inventory = pd.read_csv(inv_path, dtype=str)
    inventory["Modality"] = inventory["Modality"].fillna("").astype(str)

    try:
        inventory["dcm_count"] = pd.to_numeric(inventory["dcm_count"], errors="coerce").fillna(0).astype(int)
    except Exception:
        inventory["dcm_count"] = 0

    if cand_path.exists():
        candidate_best = pd.read_csv(cand_path, dtype=str)
        try:
            candidate_best["dcm_count"] = pd.to_numeric(candidate_best["dcm_count"], errors="coerce").fillna(0).astype(int)
        except Exception:
            pass
    else:
        candidate_best = pd.DataFrame()

    seg_df = inventory[inventory["Modality"].str.upper().eq("SEG")].copy()

    if seg_df.empty:
        raise SystemExit("No SEG modality rows found in inventory. Check DICOM inventory or modality parsing.")

    print(f"SEG series found: {len(seg_df)}")
    print(f"Patients with SEG: {seg_df['patient_folder'].nunique()}")

    seg_meta = extract_seg_metadata(seg_df, out_tables)
    links = build_seg_mr_links(inventory, seg_meta, candidate_best, out_tables)
    qc = decode_seg_qc(seg_meta, out_tables, out_figures, decode_limit=args.decode_limit)

    make_phase2_figures(seg_meta, links, qc, out_figures)
    make_summary(out_dir, seg_meta, links, qc)

    print("Phase 2 complete.")
    print(f"SEG metadata table: {out_tables / 'seg_series_metadata.csv'}")
    print(f"SEG-MR link table: {out_tables / 'seg_mr_link_table.csv'}")
    print(f"SEG QC features: {out_tables / 'seg_mask_qc_features.csv'}")
    print(f"Summary: {out_dir / 'Phase2_SEG_MR_Linking_Summary.md'}")


if __name__ == "__main__":
    main()
PY

chmod +x scripts/phase2_seg_mr_linking.py

cat > run_phase2_seg_mr_linking.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

source /home/mdsubhan01/miniforge3/etc/profile.d/conda.sh
conda activate 3dgcnn

cd "$HOME/ISPY2-3DGCNN"

mkdir -p reports/logs reports/figures reports/tables

python scripts/phase2_seg_mr_linking.py \
  --repo-root "$HOME/ISPY2-3DGCNN" \
  --decode-limit 120 \
  > reports/logs/phase2_seg_mr_linking.log 2>&1

echo "Checking for files larger than 50 MB before git add:"
find . -type f -size +50M ! -path "./.git/*" | tee reports/logs/phase2_large_file_check.txt

if [ -s reports/logs/phase2_large_file_check.txt ]; then
  echo "ERROR: Large files found. Not committing. See reports/logs/phase2_large_file_check.txt"
  exit 1
fi

git add .gitignore
git add configs/phase2_seg_mr_linking_config.yaml
git add scripts/phase2_seg_mr_linking.py
git add run_phase2_seg_mr_linking.sh
git add bootstrap_phase2_seg_mr_linking.sh
git add bootstrap_phase1_4d_vascular_atlas.sh 2>/dev/null || true
git add reports/figures
git add reports/tables
git add reports/Phase2_SEG_MR_Linking_Summary.md

git status --short

if git diff --cached --quiet; then
  echo "No new changes to commit."
else
  git commit -m "Add phase 2 SEG MR linking outputs"
  git push
fi

echo "Phase 2 completed and pushed to GitHub."
EOF

chmod +x run_phase2_seg_mr_linking.sh

echo "Phase 2 bootstrap created."
echo "Next command:"
echo "nohup bash run_phase2_seg_mr_linking.sh > reports/logs/run_phase2_nohup.log 2>&1 &"
