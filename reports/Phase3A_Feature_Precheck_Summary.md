# Phase 3A Feature Precheck Summary

This step creates a safe precheck table for verified tumor and peritumor feature extraction.

## Main outputs

- `reports/tables/phase3a_feature_precheck.csv`
- `reports/figures/20_phase3a_link_confidence.png`
- `reports/figures/21_phase3a_mask_volume_proxy.png`
- `reports/figures/22_phase3a_top_mr_descriptions.png`

## Main counts

- SEG-MR linked pairs checked: 2688
- Unique patients represented: 719
- Basic Phase 3 ready pairs: 2688
- SEG decode QC success cases already available: 120

## Interpretation

Phase 3A does not compute final clinical biomarkers yet. It checks whether each linked SEG-MR pair has enough metadata and file paths for verified feature extraction. This is needed before measuring tumor volume, shape, vesselness, and longitudinal change.

## Next step

Phase 3B should decode SEG masks and align them with matched MR image geometry for a small test group first. After that, the full cohort can be processed.
