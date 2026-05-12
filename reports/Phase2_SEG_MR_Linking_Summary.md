# Phase 2 Output Summary

This phase connects DICOM SEG masks with MR visits for the 4D vascular-perfusion tumor atlas.

## Main results

- SEG series found: 2688
- Patients with SEG series: 719
- High-confidence referenced MR links: 0
- Medium-confidence same-study MR links: 2688
- Weak same-patient candidate MR links: 0
- Unmatched SEG series: 0
- Successfully decoded SEG masks in QC sample: 120
- Failed SEG decode attempts in QC sample: 0

## Tables
- `/home/mdsubhan01/ISPY2-3DGCNN/reports/tables/seg_series_metadata.csv`
- `/home/mdsubhan01/ISPY2-3DGCNN/reports/tables/seg_mr_link_table.csv`
- `/home/mdsubhan01/ISPY2-3DGCNN/reports/tables/seg_mask_qc_features.csv`

## Figures
- `/home/mdsubhan01/ISPY2-3DGCNN/reports/figures/10_seg_series_per_patient.png`
- `/home/mdsubhan01/ISPY2-3DGCNN/reports/figures/11_seg_mr_link_status.png`
- `/home/mdsubhan01/ISPY2-3DGCNN/reports/figures/12_top_seg_segment_labels.png`
- `/home/mdsubhan01/ISPY2-3DGCNN/reports/figures/13_seg_qc_projection_demo.png`
- `/home/mdsubhan01/ISPY2-3DGCNN/reports/figures/14_phase2_pipeline_diagram.png`

## Interpretation

Phase 2 creates the verified bridge between raw DICOM data and tumor-level feature extraction.
High-confidence links use referenced DICOM SeriesInstanceUID information.
Medium-confidence links use the best MR series from the same patient and same study.

## Important limitation

This phase does not yet compute final clinical tumor volumes.
DICOM SEG frames must be spatially aligned with matched MR images before final tumor, peritumor, and vascular features are reported.

## Next phase

Phase 3 should compute verified tumor and peritumor features from the linked MR-SEG pairs, including shape, radial profile, vesselness, and enhancement statistics.