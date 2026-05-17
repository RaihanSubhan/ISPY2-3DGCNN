# Phase 8D FTV vs Baseline Manuscript-Ready Result

## Summary

This phase compares the earlier Phase 5A baseline model against the Phase 8C FTV support-data model.

## Main result

The Phase 5A best model was **LogisticRegression**, with AUROC = 0.5462, AUPRC = 0.7506, and Brier score = 0.2662.

The Phase 8C FTV best model was **SVM_RBF**, with AUROC = 0.6667, AUPRC = 0.7528, and Brier score = 0.2405.

The FTV path improved AUROC by **0.1205** and AUPRC by **0.0021** compared with the Phase 5A baseline. The Brier score changed by **-0.0257**; lower Brier is better.

## Scientific interpretation

The result supports moving the project away from weak DICOM SEG recovery and toward official support-data FTV / MRI features. The FTV path gives a cleaner tumor-response signal and a working pCR prediction pipeline.

## Important limitation

This is still a pilot result. The Phase 8C FTV model used only 13 labeled patients. Therefore, this should not be claimed as final clinical performance.

## Safe manuscript claim

A safe claim is: official MRI / FTV support features produced a stronger pilot pCR prediction signal than the earlier internal proxy biomarker pipeline.

## Claim to avoid

Do not claim that the model is clinically validated or ready for decision-making.

## Next step

Phase 8E should prepare the final research proposal/manuscript structure, including background, methods, results, limitations, and future temporal graph model plan.

## Main outputs

- reports/tables/phase8d_baseline_vs_ftv_model_comparison.csv
- reports/tables/phase8d_metric_delta_summary.csv
- reports/tables/phase8d_safe_claims_and_limitations.csv
- reports/figures/260_phase8d_baseline_vs_ftv_metrics.png
- reports/figures/261_phase8d_cohort_size_comparison.png
- reports/figures/262_phase8d_all_model_auroc.png
- reports/figures/263_phase8d_ftv_feature_importance.png
- reports/figures/264_phase8d_decision_flow.png