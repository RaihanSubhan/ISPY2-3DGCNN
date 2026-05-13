# Phase 5A Model Calibration and Interpretation Summary

This phase evaluates baseline pCR prediction models using cross-validated probabilities, calibration metrics, and feature importance.

## Main outputs

- reports/tables/phase5a_model_calibration_metrics.csv
- reports/tables/phase5a_oof_predictions.csv
- reports/tables/phase5a_feature_importance.csv
- reports/tables/phase5a_model_recommendation.csv
- reports/figures/110_phase5a_roc_curves.png
- reports/figures/111_phase5a_pr_curves.png
- reports/figures/112_phase5a_calibration_curve.png
- reports/figures/113_phase5a_feature_importance.png
- reports/figures/114_phase5a_confusion_matrix.png

## Main counts

- Labeled patients: 33
- Positive pCR labels: 20
- Negative non-pCR labels: 13
- Numeric biomarker features: 16
- CV folds: 5
- XGBoost available: not_available

## Recommended pilot model

- Model: LogisticRegression
- AUROC: 0.5462
- AUPRC: 0.7506
- Brier score: 0.2662

## Interpretation

This is still a pilot model because only 33 labeled patients are available. The current result can support feasibility and method development, but it should not be claimed as final clinical performance.

## Next step

Phase 5B should create a final manuscript-ready results report with the best model, limitations, feature interpretation, and next validation plan.