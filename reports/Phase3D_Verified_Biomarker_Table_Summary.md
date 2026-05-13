# Phase 3D Verified Biomarker Table Summary

This phase creates a compact modeling-ready biomarker table from the Phase 3C geometry matching QC output.

## Main outputs

- reports/tables/phase3d_verified_biomarker_table.csv
- reports/table_views/phase3d_verified_biomarker_table.md
- reports/figures/50_phase3d_biomarker_tier.png
- reports/figures/51_phase3d_volume_proxy_cm3.png
- reports/figures/52_phase3d_z_error_vs_volume.png
- reports/figures/53_phase3d_records_per_patient.png

## Main counts

- Phase 3C cases reviewed: 120
- Strict geometry-ready cases: 43
- Biomarker-table cases included: 120
- Patients represented: 33

## Interpretation

The strict tier should be used for primary analysis. Warning-tier records can be used for sensitivity analysis only.

## Important limitation

This table is still based on geometry QC and volume-proxy features. It is not yet the final vascular-perfusion feature table.

## Next step

Phase 4 should add modeling labels and train baseline prediction models such as logistic regression, SVM, random forest, and XGBoost.