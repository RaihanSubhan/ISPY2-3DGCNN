# Phase 7A Tumor SEG Decoding Fix Report

This phase fixes the earlier segmentation QC problem by selecting tumor-like DICOM SEG segments instead of blindly using all nonzero SEG frames.

## Why this fix was needed

Earlier overlays could look acceptable even when the selected mask covered most or all of the image slice. This phase rejects full-slice masks and records which SEG segment was selected.

## Main outputs

- reports/tables/phase7a_seg_segment_label_audit.csv
- reports/tables/phase7a_tumor_mask_candidate_table.csv
- reports/table_views/phase7a_seg_segment_label_audit.md
- reports/table_views/phase7a_tumor_mask_candidate_table.md
- reports/figures/170_phase7a_tumor_mask_overlay_grid.png
- reports/figures/171_phase7a_fix_status_counts.png
- reports/figures/172_phase7a_mask_fraction_distribution.png

## Main counts

- Cases audited: 80
- Tumor mask candidate pass cases: 0
- Full-slice masks rejected: 80
- Review-needed cases: 0
- Failed cases: 0

## Rule used

A chosen tumor candidate is rejected if the selected frame mask fraction is >= 0.50, because a tumor mask should not cover the whole image slice.

## Interpretation

No reliable tumor-like mask candidates were found. The SEG label interpretation or MR matching must be reviewed before feature extraction.

## Next step

Open the Phase 7A overlay grid in GitHub. If the contours now sit on tumor-like regions, rebuild Phase 6A using the Phase 7A tumor mask table instead of the older segmentation QC table.