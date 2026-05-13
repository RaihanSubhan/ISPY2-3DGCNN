# Phase 5B Manuscript-Ready Results Report

## Working title

A 4D Vascular-Perfusion Tumor Atlas for Early Prediction of Pathologic Complete Response in Breast Cancer Using I-SPY2 MRI

## Study aim

This study aims to build a longitudinal imaging atlas that links patient visits, DICOM SEG tumor masks, MR imaging, geometry-verified tumor biomarkers, and pCR outcome labels.

## Current cohort

The current supervised pilot cohort contains 33 labeled patients. Among them, 20 are pCR positive and 13 are non-pCR negative. The pCR rate is 0.606.

## Modeling method

Baseline supervised models were trained using cross-validated probability estimates. The tested models included logistic regression, SVM, KNN, random forest, and gradient boosting. XGBoost was not available in the current environment.

## Main result

The recommended pilot model was **LogisticRegression**. Its AUROC was **0.5462**, AUPRC was **0.7506**, and Brier score was **0.2662**.

## Interpretation

The current AUROC shows weak discrimination. This does not mean the project failed. It means the current feature table and labeled cohort are still at a feasibility stage. The result supports a complete working pipeline, from raw ISPY2 DICOM data to GitHub-tracked model outputs, but it should not be presented as final clinical performance.

## Why logistic regression is acceptable here

Logistic regression is suitable as the current pilot model because the labeled sample size is small. With only 33 labeled patients, more flexible models may overfit. Logistic regression gives a stable and interpretable baseline before moving to XGBoost, temporal models, or deep learning.

## Main limitations

1. The labeled cohort is small.
2. The current biomarkers are still proxy and QC-driven features.
3. The model has no external validation.
4. The DCE vascular-perfusion component still needs stronger pharmacokinetic and vesselness features.
5. The current result should be treated as method development, not clinical deployment.

## Recommended next phase

Phase 6 should improve the scientific strength of the project by expanding the biomarker feature set. The next technical target is to compute verified tumor and peritumor vascular features across more geometry-ready cases. These should include tumor volume, shape, peritumor rim features, Frangi vesselness, radial signal, and longitudinal change features.

## Manuscript claim that is safe

A safe claim is: the study developed a reproducible HPC and GitHub-tracked pipeline for ISPY2 DICOM inventory, SEG-MR linking, geometry QC, pCR label mapping, and pilot ML modeling.

## Manuscript claim to avoid

Do not claim that the model is clinically accurate yet. The current AUROC is not strong enough for that claim.

## Files generated

- `reports/tables/phase5b_key_results_summary.csv`
- `reports/tables/phase5b_limitations_and_next_steps.csv`
- `reports/figures/120_phase5b_pipeline_summary.png`
- `reports/figures/121_phase5b_model_performance_summary.png`
- `reports/figures/122_phase5b_top_feature_importance.png`