#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

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

try:
    from skimage import filters, morphology, measure
    from scipy import ndimage as ndi
except Exception as e:
    raise SystemExit(f"image analysis import failed: {e}")


META_FIELDS = [
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Modality",
    "SeriesDescription",
    "ProtocolName",
    "StudyDate",
    "AcquisitionDate",
    "AcquisitionTime",
    "Manufacturer",
    "BodyPartExamined",
    "MagneticFieldStrength",
    "SliceThickness",
    "RepetitionTime",
    "EchoTime",
    "FlipAngle",
]


def count_and_first_dicom(series_dir: Path) -> Tuple[int, str]:
    count = 0
    first = ""
    for root, _, files in os.walk(series_dir):
        for name in files:
            if name.lower().endswith(".dcm"):
                count += 1
                if not first:
                    first = str(Path(root) / name)
    return count, first


def read_dicom_meta(path: str) -> Dict[str, str]:
    out = {k: "" for k in META_FIELDS}
    if not path:
        return out
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        for k in META_FIELDS:
            v = getattr(ds, k, "")
            out[k] = str(v) if v is not None else ""
    except Exception as e:
        out["read_error"] = str(e)[:200]
    return out


def build_inventory(data_root: Path, out_tables: Path, out_figures: Path, max_patients: int = 0) -> pd.DataFrame:
    patient_dirs = [p for p in sorted(data_root.iterdir()) if p.is_dir()]
    if max_patients and max_patients > 0:
        patient_dirs = patient_dirs[:max_patients]

    rows = []

    for pdir in tqdm(patient_dirs, desc="Scanning patients"):
        study_dirs = [s for s in sorted(pdir.iterdir()) if s.is_dir()]
        for study_dir in study_dirs:
            series_dirs = [x for x in sorted(study_dir.iterdir()) if x.is_dir()]
            for series_dir in series_dirs:
                dcm_count, first_dcm = count_and_first_dicom(series_dir)
                if dcm_count == 0:
                    continue

                meta = read_dicom_meta(first_dcm)
                prefix = series_dir.name.split("_", 1)[0] if "_" in series_dir.name else ""

                row = {
                    "patient_folder": pdir.name,
                    "study_folder": study_dir.name,
                    "series_folder": series_dir.name,
                    "series_path": str(series_dir),
                    "first_dcm": first_dcm,
                    "dcm_count": int(dcm_count),
                    "folder_modality_prefix": prefix,
                }
                row.update(meta)

                if not row.get("Modality"):
                    row["Modality"] = prefix

                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No DICOM series found. Check data path.")

    df.to_csv(out_tables / "ispy2_series_inventory.csv", index=False)

    patient_summary = df.groupby("patient_folder").agg(
        n_studies=("study_folder", "nunique"),
        n_series=("series_folder", "nunique"),
        n_dicom=("dcm_count", "sum"),
        has_mr=("Modality", lambda x: int(any(str(v).upper() == "MR" for v in x))),
        has_seg=("Modality", lambda x: int(any(str(v).upper() == "SEG" for v in x))),
    ).reset_index()

    patient_summary.to_csv(out_tables / "ispy2_patient_summary.csv", index=False)

    make_inventory_figures(df, patient_summary, out_figures)

    return df


def make_inventory_figures(df: pd.DataFrame, patient_summary: pd.DataFrame, out_figures: Path) -> None:
    plt.figure(figsize=(7, 4))
    df["Modality"].fillna("UNKNOWN").replace("", "UNKNOWN").value_counts().plot(kind="bar")
    plt.title("ISPY2 Series by Modality")
    plt.ylabel("Number of series")
    plt.tight_layout()
    plt.savefig(out_figures / "01_modality_counts.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 4))
    patient_summary["n_dicom"].plot(kind="hist", bins=40)
    plt.title("DICOM File Count per Patient")
    plt.xlabel("DICOM files")
    plt.ylabel("Number of patients")
    plt.tight_layout()
    plt.savefig(out_figures / "02_dicom_files_per_patient.png", dpi=220)
    plt.close()

    top_desc = df["SeriesDescription"].fillna("UNKNOWN").replace("", "UNKNOWN").value_counts().head(20).sort_values()
    plt.figure(figsize=(9, 7))
    top_desc.plot(kind="barh")
    plt.title("Top 20 Series Descriptions")
    plt.xlabel("Number of series")
    plt.tight_layout()
    plt.savefig(out_figures / "03_top_series_descriptions.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 4))
    patient_summary["n_studies"].plot(kind="hist", bins=20)
    plt.title("Visit Count per Patient")
    plt.xlabel("Number of study folders")
    plt.ylabel("Number of patients")
    plt.tight_layout()
    plt.savefig(out_figures / "04_visit_count_distribution.png", dpi=220)
    plt.close()


def build_visit_tracker(df: pd.DataFrame, out_tables: Path, out_figures: Path, max_patients: int = 80) -> pd.DataFrame:
    visits = df[["patient_folder", "study_folder", "StudyDate", "Modality", "dcm_count"]].copy()
    visits["StudyDate"] = visits["StudyDate"].fillna("").astype(str)

    visits = visits.groupby(["patient_folder", "study_folder", "StudyDate"], as_index=False).agg(
        n_series=("Modality", "count"),
        n_dicom=("dcm_count", "sum"),
        has_mr=("Modality", lambda x: int(any(str(v).upper() == "MR" for v in x))),
        has_seg=("Modality", lambda x: int(any(str(v).upper() == "SEG" for v in x))),
    )

    visits = visits.sort_values(["patient_folder", "StudyDate", "study_folder"])
    visits["visit_index"] = visits.groupby("patient_folder").cumcount()
    visits.to_csv(out_tables / "cohort_visit_tracker_table.csv", index=False)

    patients = sorted(visits["patient_folder"].unique())[:max_patients]
    sub = visits[visits["patient_folder"].isin(patients)]
    max_visit = int(sub["visit_index"].max()) + 1 if len(sub) else 1

    mat = np.zeros((len(patients), max_visit), dtype=float)
    pat_to_i = {p: i for i, p in enumerate(patients)}

    for _, r in sub.iterrows():
        value = 2.0 if int(r["has_seg"]) else 1.0
        mat[pat_to_i[r["patient_folder"]], int(r["visit_index"])] = value

    plt.figure(figsize=(9, max(4, len(patients) * 0.08)))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest", vmin=0, vmax=2)
    plt.title("Visit Tracking Atlas: MR Visit Present, SEG Available")
    plt.xlabel("Visit index")
    plt.ylabel("Patient")
    plt.yticks(range(len(patients)), patients, fontsize=5)
    cbar = plt.colorbar(im)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["missing", "MR", "MR + SEG"])
    plt.tight_layout()
    plt.savefig(out_figures / "05_visit_tracking_heatmap.png", dpi=220)
    plt.close()

    return visits


def score_candidate_series(row: pd.Series) -> float:
    desc = (str(row.get("SeriesDescription", "")) + " " + str(row.get("ProtocolName", ""))).lower()

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
        dcm_count = float(row.get("dcm_count", 0))
    except Exception:
        dcm_count = 0.0

    score += min(np.log1p(dcm_count), 10.0)

    if str(row.get("Modality", "")).upper() == "MR":
        score += 5.0

    return score


def select_candidate_series(df: pd.DataFrame, out_tables: Path, out_figures: Path) -> pd.DataFrame:
    mr = df[df["Modality"].astype(str).str.upper().eq("MR")].copy()

    if mr.empty:
        raise SystemExit("No MR series found. Candidate series selection failed.")

    mr["candidate_score"] = mr.apply(score_candidate_series, axis=1)
    mr = mr.sort_values(["patient_folder", "candidate_score", "dcm_count"], ascending=[True, False, False])

    top3 = mr.groupby("patient_folder", as_index=False).head(3).copy()
    best = mr.groupby("patient_folder", as_index=False).head(1).copy()

    top3.to_csv(out_tables / "candidate_mr_series_top3_per_patient.csv", index=False)
    best.to_csv(out_tables / "candidate_mr_series_best_per_patient.csv", index=False)

    plt.figure(figsize=(7, 4))
    best["candidate_score"].plot(kind="hist", bins=40)
    plt.title("Candidate DCE/T1 MR Series Score Distribution")
    plt.xlabel("Candidate score")
    plt.ylabel("Number of patients")
    plt.tight_layout()
    plt.savefig(out_figures / "06_candidate_score_distribution.png", dpi=220)
    plt.close()

    top_desc = top3["SeriesDescription"].fillna("UNKNOWN").replace("", "UNKNOWN").value_counts().head(20).sort_values()
    plt.figure(figsize=(9, 7))
    top_desc.plot(kind="barh")
    plt.title("Top Candidate MR Series Descriptions")
    plt.xlabel("Count among top-3 candidate series")
    plt.tight_layout()
    plt.savefig(out_figures / "07_candidate_series_descriptions.png", dpi=220)
    plt.close()

    return best


def list_dicom_files(series_path: str) -> List[str]:
    files = []
    for root, _, names in os.walk(series_path):
        for n in names:
            if n.lower().endswith(".dcm"):
                files.append(str(Path(root) / n))
    return sorted(files)


def dicom_sort_key(path: str) -> Tuple[float, int]:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        inst = getattr(ds, "InstanceNumber", 0)
        ipp = getattr(ds, "ImagePositionPatient", None)

        if ipp is not None and len(ipp) >= 3:
            z = float(ipp[2])
        else:
            z = float(inst) if str(inst).replace(".", "", 1).isdigit() else 0.0

        inst_i = int(inst) if str(inst).isdigit() else 0
        return z, inst_i
    except Exception:
        return 0.0, 0


def load_dicom_stack(series_path: str, max_files: int = 120) -> Tuple[np.ndarray, np.ndarray]:
    files = list_dicom_files(series_path)

    if not files:
        raise RuntimeError("No DICOM files found in selected series.")

    sample_files = files[: min(len(files), 2500)]
    sorted_files = sorted(sample_files, key=dicom_sort_key)

    if len(sorted_files) > max_files:
        idx = np.linspace(0, len(sorted_files) - 1, max_files).astype(int)
        sorted_files = [sorted_files[i] for i in idx]

    arrays = []
    spacings = []
    first_shape = None

    for f in sorted_files:
        try:
            ds = pydicom.dcmread(f, force=True)
            arr = ds.pixel_array.astype(np.float32)

            slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
            intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
            arr = arr * slope + intercept

            if first_shape is None:
                first_shape = arr.shape

            if arr.shape != first_shape:
                continue

            arrays.append(arr)

            px = getattr(ds, "PixelSpacing", [1.0, 1.0])
            th = float(getattr(ds, "SliceThickness", 1.0) or 1.0)
            spacings.append((float(px[0]), float(px[1]), th))
        except Exception:
            continue

    if len(arrays) < 8:
        raise RuntimeError("Too few readable DICOM slices. Pixel decoding may need more codecs or a different series.")

    volume = np.stack(arrays, axis=0)
    spacing = np.median(np.array(spacings), axis=0) if spacings else np.array([1.0, 1.0, 1.0])

    return volume, spacing


def normalize01(volume: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(volume, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((volume - lo) / (hi - lo), 0, 1).astype(np.float32)


def prototype_tumor_mask(vn: np.ndarray) -> np.ndarray:
    crop = vn[int(vn.shape[0] * 0.2): int(vn.shape[0] * 0.8)]

    try:
        threshold = filters.threshold_otsu(crop)
    except Exception:
        threshold = float(np.percentile(crop, 90))

    threshold = max(threshold, float(np.percentile(crop, 88)))
    mask = vn > threshold
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=100)

    labels = measure.label(mask)
    if labels.max() == 0:
        return mask.astype(bool)

    center = np.array(vn.shape) / 2.0
    best_label = None
    best_score = -1e18

    for prop in measure.regionprops(labels):
        dist = np.linalg.norm(np.array(prop.centroid) - center)
        score = prop.area - 0.15 * dist
        if score > best_score:
            best_score = score
            best_label = prop.label

    mask = labels == best_label
    mask = ndi.binary_fill_holes(mask)

    return mask.astype(bool)


def surface_area_mm2(mask: np.ndarray, spacing: np.ndarray) -> float:
    if mask.sum() < 50:
        return float("nan")

    try:
        verts, faces, _, _ = measure.marching_cubes(
            mask.astype(float),
            level=0.5,
            spacing=(spacing[2], spacing[0], spacing[1]),
        )
        tri = verts[faces]
        a = tri[:, 1] - tri[:, 0]
        b = tri[:, 2] - tri[:, 0]
        area = 0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum()
        return float(area)
    except Exception:
        return float("nan")


def make_vascular_demo(best: pd.DataFrame, out_tables: Path, out_figures: Path, max_files: int = 120) -> None:
    row = best.iloc[0].to_dict()
    patient = row["patient_folder"]
    series_path = row["series_path"]

    try:
        volume, spacing = load_dicom_stack(series_path, max_files=max_files)
        vn = normalize01(volume)

        mask = prototype_tumor_mask(vn)

        if mask.sum() < 50:
            raise RuntimeError("Prototype ROI too small. Candidate series may not be suitable for demo.")

        slice_scores = mask.reshape(mask.shape[0], -1).sum(axis=1)
        z = int(np.argmax(slice_scores))

        img2d = vn[z]
        mask2d = mask[z]

        try:
            vesselness = filters.frangi(img2d, sigmas=[1, 2, 3, 4], black_ridges=False)
        except Exception:
            vesselness = np.zeros_like(img2d)

        if np.any(vesselness > 0):
            vth = np.percentile(vesselness[vesselness > 0], 92)
        else:
            vth = 1.0

        vessels = vesselness > vth
        vessels = morphology.remove_small_objects(vessels.astype(bool), min_size=8)
        skeleton = morphology.skeletonize(vessels)

        peri = morphology.binary_dilation(mask2d, morphology.disk(15)) & (~mask2d)

        vessel_density_tumor = float(skeleton[mask2d].sum() / max(mask2d.sum(), 1))
        vessel_density_peri = float(skeleton[peri].sum() / max(peri.sum(), 1))

        voxel_mm3 = float(spacing[0] * spacing[1] * spacing[2])
        volume_mm3 = float(mask.sum() * voxel_mm3)
        area_mm2 = surface_area_mm2(mask, spacing)

        if np.isfinite(area_mm2) and area_mm2 > 0:
            sphericity = float((np.pi ** (1 / 3)) * ((6 * volume_mm3) ** (2 / 3)) / area_mm2)
            compactness = float(volume_mm3 / (area_mm2 ** 1.5))
        else:
            sphericity = float("nan")
            compactness = float("nan")

        yy, xx = np.indices(img2d.shape)

        cy, cx = ndi.center_of_mass(mask2d.astype(float))
        if not np.isfinite(cx) or not np.isfinite(cy):
            cy, cx = img2d.shape[0] / 2.0, img2d.shape[1] / 2.0

        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

        if mask2d.any():
            max_r = np.percentile(radius[mask2d], 95)
        else:
            max_r = 50

        bins = np.linspace(0, max_r, 20)
        radial_signal = []
        radial_centers = []

        for a, b in zip(bins[:-1], bins[1:]):
            idx = mask2d & (radius >= a) & (radius < b)
            radial_signal.append(float(img2d[idx].mean()) if idx.any() else np.nan)
            radial_centers.append(float((a + b) / 2.0))

        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        axes = axes.ravel()

        axes[0].imshow(img2d, cmap="gray")
        axes[0].set_title(f"Candidate MR slice\n{patient}")
        axes[0].axis("off")

        axes[1].imshow(img2d, cmap="gray")
        axes[1].contour(mask2d, colors="r", linewidths=1)
        axes[1].set_title("Prototype tumor ROI")
        axes[1].axis("off")

        axes[2].imshow(vesselness, cmap="magma")
        axes[2].set_title("Frangi vesselness map")
        axes[2].axis("off")

        axes[3].imshow(img2d, cmap="gray")
        axes[3].contour(mask2d, colors="r", linewidths=1)
        axes[3].imshow(np.ma.masked_where(~skeleton, skeleton), alpha=0.8)
        axes[3].set_title("Vessel skeleton overlay")
        axes[3].axis("off")

        axes[4].plot(radial_centers, radial_signal, marker="o")
        axes[4].set_title("Radial enhancement profile")
        axes[4].set_xlabel("Radius from ROI center")
        axes[4].set_ylabel("Normalized signal")

        text = (
            f"Patient: {patient}\n"
            f"Prototype volume: {volume_mm3:,.1f} mm^3\n"
            f"Surface area: {area_mm2:,.1f} mm^2\n"
            f"Sphericity: {sphericity:.3f}\n"
            f"Compactness: {compactness:.6f}\n"
            f"Tumor vessel density: {vessel_density_tumor:.4f}\n"
            f"Peritumor vessel density: {vessel_density_peri:.4f}\n\n"
            "Important: This ROI is a prototype only.\n"
            "Final study must use verified SEG masks\n"
            "or a validated segmentation model."
        )

        axes[5].text(0.02, 0.98, text, va="top", ha="left", fontsize=10)
        axes[5].axis("off")

        plt.suptitle("4D Vascular-Perfusion Tumor Atlas: One-Patient Visual Demo", y=0.98)
        plt.tight_layout()
        plt.savefig(out_figures / "08_one_patient_vascular_atlas_demo.png", dpi=220)
        plt.close()

        features = pd.DataFrame([{
            "status": "success_prototype_roi_not_clinical",
            "patient_folder": patient,
            "series_path": series_path,
            "used_slices": int(volume.shape[0]),
            "height": int(volume.shape[1]),
            "width": int(volume.shape[2]),
            "voxel_mm3": voxel_mm3,
            "prototype_tumor_voxels": int(mask.sum()),
            "prototype_volume_mm3": volume_mm3,
            "surface_mm2": area_mm2,
            "sphericity": sphericity,
            "compactness": compactness,
            "mean_normalized_signal_roi": float(vn[mask].mean()),
            "vessel_density_tumor_2d": vessel_density_tumor,
            "vessel_density_peritumor_2d": vessel_density_peri,
        }])

        features.to_csv(out_tables / "one_patient_vascular_atlas_features.csv", index=False)

    except Exception as e:
        message = str(e)

        plt.figure(figsize=(9, 4))
        plt.text(
            0.5,
            0.65,
            "Pixel-level vascular atlas demo could not run",
            ha="center",
            va="center",
            fontsize=14,
        )
        plt.text(
            0.5,
            0.42,
            message[:260],
            ha="center",
            va="center",
            fontsize=9,
            wrap=True,
        )
        plt.text(
            0.5,
            0.22,
            "Metadata tables and cohort figures are still valid outputs.",
            ha="center",
            va="center",
            fontsize=10,
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_figures / "08_one_patient_vascular_atlas_demo.png", dpi=220)
        plt.close()

        pd.DataFrame([{
            "status": "pixel_demo_failed",
            "message": message,
            "patient_folder": patient,
            "series_path": series_path,
        }]).to_csv(out_tables / "one_patient_vascular_atlas_features.csv", index=False)


def make_model_table(out_tables: Path, out_figures: Path) -> None:
    model_table = pd.DataFrame([
        {
            "model_head": "Radiomics + XGBoost",
            "input": "tumor volume, shape, radial profile, vessel density, visit deltas",
            "output": "pCR or early response probability",
            "role": "primary interpretable prediction model",
            "why": "best first model for structured atlas biomarkers",
        },
        {
            "model_head": "Temporal visit graph",
            "input": "patient visits, missing timepoints, longitudinal features",
            "output": "patient trajectory embedding",
            "role": "longitudinal tracking and missingness modeling",
            "why": "captures patient-level time structure",
        },
        {
            "model_head": "3D CNN or 3D U-Net",
            "input": "DCE-MRI volume and verified mask labels",
            "output": "tumor mask or image embedding",
            "role": "future automated segmentation and representation learning",
            "why": "use after mask quality control, not before",
        },
        {
            "model_head": "Offline RL",
            "input": "state = atlas features, action = treatment arm, reward = response/toxicity",
            "output": "policy hypothesis",
            "role": "later-stage treatment policy simulation",
            "why": "only valid after careful causal/offline policy design",
        },
    ])

    model_table.to_csv(out_tables / "unified_model_heads_for_atlas.csv", index=False)

    plt.figure(figsize=(10, 5))
    y = np.arange(len(model_table))
    plt.barh(y, [4, 3, 2, 1])
    plt.yticks(y, model_table["model_head"])
    plt.xlabel("Recommended order of use")
    plt.title("Unified Atlas Downstream Model Plan")
    plt.tight_layout()
    plt.savefig(out_figures / "09_model_head_plan.png", dpi=220)
    plt.close()


def make_summary(out_dir: Path) -> None:
    figures = sorted((out_dir / "figures").glob("*.png"))
    tables = sorted((out_dir / "tables").glob("*.csv"))

    lines = []
    lines.append("# Phase 1 Output Summary")
    lines.append("")
    lines.append("This is the first reproducible output of the 4D vascular-perfusion tumor atlas project.")
    lines.append("")
    lines.append("## What was created")
    lines.append("")
    lines.append("- Full ISPY2 DICOM series inventory")
    lines.append("- Patient-level summary table")
    lines.append("- Visit tracking table and heatmap")
    lines.append("- Candidate DCE/T1 MR series selection table")
    lines.append("- One-patient prototype vascular atlas visual demo")
    lines.append("- Unified downstream model table")
    lines.append("")
    lines.append("## Figures")
    for f in figures:
        lines.append(f"- `{f}`")
    lines.append("")
    lines.append("## Tables")
    for t in tables:
        lines.append(f"- `{t}`")
    lines.append("")

    inv = out_dir / "tables" / "ispy2_series_inventory.csv"
    if inv.exists():
        df = pd.read_csv(inv)
        lines.append("## Inventory counts")
        lines.append(f"- Patients: {df['patient_folder'].nunique()}")
        lines.append(f"- Series: {len(df)}")
        lines.append(f"- DICOM files counted: {int(df['dcm_count'].sum()):,}")
        lines.append("")

    lines.append("## Important limitation")
    lines.append("")
    lines.append("The one-patient tumor ROI is a prototype visual ROI. It is not a final clinical segmentation.")
    lines.append("The final study should use verified SEG masks or a validated segmentation model before reporting tumor volume, shape, or vascular biomarkers.")
    lines.append("")
    lines.append("## Next phase")
    lines.append("")
    lines.append("Phase 2 should connect SEG masks to each MR visit, compute verified tumor and peritumor features, then create longitudinal delta features across treatment visits.")

    (out_dir / "Phase1_Output_Summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--max-pixel-files", type=int, default=120)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir)
    out_tables = out_dir / "tables"
    out_figures = out_dir / "figures"

    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    print("Building inventory...")
    inventory = build_inventory(data_root, out_tables, out_figures, max_patients=args.max_patients)

    print("Building visit tracker...")
    build_visit_tracker(inventory, out_tables, out_figures)

    print("Selecting candidate MR series...")
    best = select_candidate_series(inventory, out_tables, out_figures)

    print("Building one-patient vascular atlas demo...")
    make_vascular_demo(best, out_tables, out_figures, max_files=args.max_pixel_files)

    print("Making model table...")
    make_model_table(out_tables, out_figures)

    print("Making summary...")
    make_summary(out_dir)

    print("Phase 1 complete.")
    print(f"Figures: {out_figures}")
    print(f"Tables: {out_tables}")


if __name__ == "__main__":
    main()
