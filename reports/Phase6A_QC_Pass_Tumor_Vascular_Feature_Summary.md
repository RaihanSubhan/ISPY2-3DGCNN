# Phase 6A QC-Pass Tumor and Vascular Feature Summary

This phase extracts tumor, peritumor, radial signal, habitat, and vesselness features from visual-QC-passed SEG-MR cases.

## Main outputs

- reports/tables/phase6a_qc_pass_tumor_vascular_features.csv
- reports/table_views/phase6a_qc_pass_tumor_vascular_features.md
- reports/figures/140_phase6a_tumor_area_mm2.png
- reports/figures/141_phase6a_vessel_density_scatter.png
- reports/figures/142_phase6a_signal_ratio.png
- reports/figures/143_phase6a_habitat_fractions.png
- reports/figures/144_phase6a_feature_correlation_heatmap.png
- reports/figures/145_phase6a_vascular_feature_overlay_grid.png

## Main counts

- QC-passed cases attempted: 12
- Feature extraction success cases: 12
- Failed cases: 0

## Features created

- tumor area and volume proxy
- circularity, solidity, eccentricity, bounding box
- tumor and peritumor signal
- tumor-to-peritumor signal ratio
- radial signal profile
- low, middle, and high enhancement habitat fractions
- Frangi vesselness and vessel skeleton density

## Interpretation

These features are stronger than the earlier proxy-only table because they are extracted from visually QC-passed segmentation overlays.

## Limitation

This phase still uses representative 2D tumor slices plus 3D volume proxy. A future phase should extend the same feature extraction over full 3D matched tumor volumes and across longitudinal visits.

## Next step

Phase 6B should merge these vascular features with pCR labels and rerun pilot ML to test whether vascular features improve AUROC and AUPRC.