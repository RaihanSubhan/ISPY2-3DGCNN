# Phase 1 Output Summary

This is the first reproducible output of the 4D vascular-perfusion tumor atlas project.

## What was created

- Full ISPY2 DICOM series inventory
- Patient-level summary table
- Visit tracking table and heatmap
- Candidate DCE/T1 MR series selection table
- One-patient prototype vascular atlas visual demo
- Unified downstream model table

## Figures
- `reports/figures/01_modality_counts.png`
- `reports/figures/02_dicom_files_per_patient.png`
- `reports/figures/03_top_series_descriptions.png`
- `reports/figures/04_visit_count_distribution.png`
- `reports/figures/05_visit_tracking_heatmap.png`
- `reports/figures/06_candidate_score_distribution.png`
- `reports/figures/07_candidate_series_descriptions.png`
- `reports/figures/08_one_patient_vascular_atlas_demo.png`
- `reports/figures/09_model_head_plan.png`

## Tables
- `reports/tables/candidate_mr_series_best_per_patient.csv`
- `reports/tables/candidate_mr_series_top3_per_patient.csv`
- `reports/tables/cohort_visit_tracker_table.csv`
- `reports/tables/ispy2_patient_summary.csv`
- `reports/tables/ispy2_series_inventory.csv`
- `reports/tables/one_patient_vascular_atlas_features.csv`
- `reports/tables/unified_model_heads_for_atlas.csv`

## Inventory counts
- Patients: 719
- Series: 32411
- DICOM files counted: 5,586,493

## Important limitation

The one-patient tumor ROI is a prototype visual ROI. It is not a final clinical segmentation.
The final study should use verified SEG masks or a validated segmentation model before reporting tumor volume, shape, or vascular biomarkers.

## Next phase

Phase 2 should connect SEG masks to each MR visit, compute verified tumor and peritumor features, then create longitudinal delta features across treatment visits.