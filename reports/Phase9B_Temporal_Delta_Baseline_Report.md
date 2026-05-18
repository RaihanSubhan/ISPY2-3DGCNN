# Phase 9B Temporal-Delta Baseline Model Report

This phase trains a simple temporal-delta baseline model using longitudinal FTV graph features.

## Main outputs

- reports/tables/phase9b_temporal_delta_modeling_dataset.csv
- reports/tables/phase9b_temporal_delta_model_performance.csv
- reports/tables/phase9b_temporal_delta_model_predictions.csv
- reports/tables/phase9b_temporal_delta_feature_importance.csv
- reports/tables/phase9b_vs_phase8c_comparison.csv
- reports/figures/290_phase9b_label_distribution.png
- reports/figures/291_phase9b_model_auroc.png
- reports/figures/292_phase9b_model_auprc.png
- reports/figures/293_phase9b_feature_importance.png
- reports/figures/294_phase9b_vs_phase8c_comparison.png

## Main counts

- Patients: 13
- pCR positive: 7
- non-pCR negative: 6
- Temporal-delta features: 10

## Modeling result

ML was run successfully.
- Best model: RandomForest
- AUROC: 0.5714
- AUPRC: 0.6060
- Brier score: 0.2724

## Interpretation

This is the first temporal baseline for the future graph-learning direction. It tests whether simple first-to-last visit changes carry pCR signal before building a temporal GNN.

## Next step

Phase 9C should compare Phase 8C and Phase 9B, then decide whether a temporal GNN is justified or whether the classical FTV SVM remains the strongest pilot model.