# Phase 7B Deep SEG Diagnosis Report

This phase diagnoses whether the downloaded DICOM SEG files contain real tumor-only segments or mostly full-slice masks.

## Main outputs

- reports/tables/phase7b_seg_case_deep_diagnosis.csv
- reports/tables/phase7b_seg_segment_deep_diagnosis.csv
- reports/table_views/phase7b_seg_case_deep_diagnosis.md
- reports/table_views/phase7b_seg_segment_deep_diagnosis.md
- reports/figures/180_phase7b_case_diagnosis_status.png
- reports/figures/181_phase7b_segment_classification.png
- reports/figures/182_phase7b_mask_fraction_distribution.png
- reports/figures/183_phase7b_top_segment_labels.png

## Main counts

- SEG cases diagnosed: 300
- Cases with plausible tumor-like segment: 0
- Cases with only full-slice segments: 300
- Failed cases: 0
- Segment rows diagnosed: 300
- Plausible tumor-like segments: 0
- Full-slice segments: 300

## Interpretation

No plausible tumor-like DICOM SEG segments were found in this diagnostic sample. This means the current SEG files are probably not tumor masks, or the tumor masks are stored in a different support-data file or different DICOM series.

Do not use the current SEG files for tumor volume, vascular density, or graph-node construction until a real tumor mask source is identified.

## Next step

If plausible segments exist, run a Phase 7C overlay builder for those segments. If none exist, search the ISPY2 support data for tumor masks, FTV maps, PE/SER maps, or ROI files instead of using the current SEG files.