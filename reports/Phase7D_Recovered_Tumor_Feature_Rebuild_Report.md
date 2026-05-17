# Phase 7D Recovered Tumor Feature Rebuild Report

This phase rebuilds tumor and vascular features using only Phase 7C fractional tumor candidate pass cases.

## Main outputs

- reports/tables/phase7d_recovered_tumor_vascular_features.csv
- reports/table_views/phase7d_recovered_tumor_vascular_features.md
- reports/figures/200_phase7d_recovered_overlay_grid.png
- reports/figures/201_phase7d_feature_status.png
- reports/figures/202_phase7d_tumor_area.png
- reports/figures/203_phase7d_vessel_density.png

## Main counts

- Phase 7C pass cases attempted: 8
- Feature extraction success cases: 8
- Failed or rejected cases: 0

## Interpretation

Recovered fractional tumor masks produced usable tumor and vascular features. These are still a small pilot set and should be visually reviewed before modeling.

## Next step

Run Phase 7E to compare recovered tumor features with pCR labels. If both pCR classes are present, run a small feasibility model.