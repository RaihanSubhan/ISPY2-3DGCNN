# Phase 8C FTV Feature Modeling Report

This phase uses the strongest Phase 8B support-data source, the multi-feature MRI NACT / FTV table, to build a patient-level pCR modeling dataset.

## Selected source

- File: /home/mdsubhan01/ISPY2-3DGCNN/external/ispy2_clinical_support/Multi-feature_MRI_NACT_Data_b3cb2e5ecfa7.xlsx
- Patient column: CLINICAL-TRIAL-SUBJECT-ID
- Rows in source: 384
- Numeric source features: 28

## Main outputs

- reports/tables/phase8c_ftv_source_selection.csv
- reports/tables/phase8c_ftv_modeling_dataset.csv
- reports/tables/phase8c_ftv_feature_completeness.csv
- reports/tables/phase8c_ftv_model_performance.csv
- reports/table_views/phase8c_ftv_modeling_dataset.md
- reports/figures/250_phase8c_ftv_label_distribution.png
- reports/figures/251_phase8c_ftv_model_auroc.png

## Main counts

- Labeled FTV patients: 13
- pCR positive: 7
- non-pCR negative: 6
- Usable numeric FTV features: 28

## Modeling status

ML was run successfully.
- Best model: SVM_RBF
- AUROC: 0.6667
- AUPRC: 0.7528
- Brier score: 0.2405

## Interpretation

This is now a stronger path than the recovered DICOM SEG path because it uses official multi-feature MRI/FTV support data rather than weak fractional DICOM SEG recovery.

## Next step

Phase 8D should compare the FTV support-data model against Phase 5A and prepare a clean manuscript-ready result.