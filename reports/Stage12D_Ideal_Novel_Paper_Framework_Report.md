# Stage 12D Ideal Novel Paper Framework Report

This stage builds the final paper framework based on the three attached paper directions, your ISPY2 pipeline, and the exploratory temporal 3DGCNN result.

## Main decision

The paper should not claim that the 3DGCNN is the best model. The paper should claim that tumor-source validation and FTV-based pCR prediction are the current main contributions, with temporal 3DGCNN as an exploratory future-ready framework.

## Final model ranking

- Rank 1: Phase 8C FTV support-data model / SVM_RBF with AUROC 0.6667, AUPRC 0.7528, and Brier 0.2405.
- Rank 2: Phase 9B temporal-delta baseline / RandomForest with AUROC 0.5714, AUPRC 0.6060, and Brier 0.2724.
- Rank 3: Stage 12B exploratory temporal 3DGCNN / Exploratory_Temporal_3DGCNN with AUROC 0.5714, AUPRC 0.5897, and Brier 0.3909.

## Ideal title

Tumor-Source Validation and FTV-Based pCR Prediction in I-SPY2 Breast MRI with an Exploratory Temporal 3DGCNN Framework

## Novelty

The novelty is not only the model. The novelty is the complete path: validate tumor source first, reject weak DICOM SEG as the main path, use official MRI/FTV features for the main pilot model, and build a temporal graph framework for future 3DGCNN work.

## Main outputs

- reports/tables/stage12d_final_model_ranking.csv
- reports/tables/stage12d_three_paper_alignment.csv
- reports/tables/stage12d_novel_contribution_map.csv
- reports/tables/stage12d_final_publication_claims.csv
- reports/tables/stage12d_ideal_paper_outline.csv
- reports/figures/530_stage12d_final_model_ranking.png
- reports/figures/531_stage12d_novel_paper_framework.png
- reports/figures/532_stage12d_claim_map.png
- reports/figures/533_stage12d_ideal_paper_structure.png
- reports/final_package/Stage12D_Ideal_Novel_Paper_Draft.md

## Next step

Use this framework to write the manuscript. Do not run more models unless the goal is to expand the cohort or add richer PE/SER tumor-region features.