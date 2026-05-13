# Phase 4B Label Merge and Baseline ML Summary

This phase merges the best available pCR/response label with the biomarker table and trains baseline ML models.

## Main outputs

- reports/tables/phase4b_label_merge_candidates.csv
- reports/tables/phase4b_modeling_dataset.csv
- reports/tables/phase4b_model_performance.csv
- reports/tables/phase4b_model_predictions.csv
- reports/figures/70_phase4b_label_distribution.png
- reports/figures/71_phase4b_model_auroc.png
- reports/figures/72_phase4b_model_auprc.png

## Label source

- File: /home/mdsubhan01/ISPY2-3DGCNN/reports/tables/phase4a_required_label_template.csv
- Column: pCR_label

## Main counts

- Labeled patients used: 33
- Positive labels: 20
- Negative labels: 13
- Numeric biomarker features: 16
- CV folds: 5

## Interpretation

These are pilot baseline models. Because the cohort is small, results should be treated as feasibility results, not final clinical performance.

## Next step

Phase 4C should add model calibration, SHAP-style interpretation, and a cleaner train/test design after labels are verified.