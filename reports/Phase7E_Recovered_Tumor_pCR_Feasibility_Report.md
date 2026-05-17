# Phase 7E Recovered Tumor pCR Feasibility Report

This phase merges recovered Phase 7D tumor/vascular features with pCR labels.

## Main outputs

- reports/tables/phase7e_recovered_tumor_label_merge.csv
- reports/tables/phase7e_recovered_tumor_modeling_dataset.csv
- reports/tables/phase7e_recovered_tumor_model_performance.csv
- reports/table_views/phase7e_recovered_tumor_label_merge.md
- reports/table_views/phase7e_recovered_tumor_modeling_dataset.md
- reports/figures/210_phase7e_label_distribution.png
- reports/figures/211_phase7e_top_feature_by_label.png
- reports/figures/212_phase7e_model_performance.png

## Main counts

- Recovered feature rows: 94
- Unique recovered feature patients: 44
- Labeled recovered records: 2
- Labeled recovered patients: 2
- pCR positive patients: 1
- non-pCR negative patients: 1
- Numeric recovered tumor features: 34

## Feasibility result

Supervised ML was not run because the recovered tumor cohort is still too small or lacks both pCR classes.

## Next step

Run Phase 7F to expand fractional threshold recovery beyond the current 8 recovered cases. The goal is to obtain more labeled pCR and non-pCR patients before model training.