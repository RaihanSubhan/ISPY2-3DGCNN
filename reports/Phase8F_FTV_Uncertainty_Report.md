# Phase 8F: FTV pCR Model Uncertainty

Bootstrap 95% confidence intervals and a label-permutation p-value for the FTV pCR
model, computed from the saved per-patient predictions (no retraining).

## Best model: SVM_RBF
- AUROC 0.6667 (95% CI 0.3056 to 1.0)
- Permutation p-value (AUROC vs chance): 0.1653
- 95% CI includes chance (0.5): YES
- Statistically significant at 0.05: NO

## Honest interpretation
The 95% CI includes 0.5 and the permutation test is not significant, so with this
cohort the model's discrimination cannot be distinguished from chance. Report the
result strictly as a feasibility signal, never as a working pCR predictor. The wide
interval is the direct consequence of n=13 and is the core argument for 'pilot, not
validation'.

## Outputs
- reports/tables/phase8f_ftv_metric_confidence_intervals.csv
- reports/tables/phase8f_ftv_permutation_pvalues.csv
- reports/figures/270_phase8f_ftv_auroc_ci_forest.png

## Limitation
n=13 labeled FTV patients. CIs are wide; this is a pilot feasibility result.
