# Stage 12C 3DGCNN Decision and Novelty Report

This stage interprets the Stage 12B exploratory temporal 3DGCNN result and positions the novel paper idea.

## Main conclusion

The exploratory temporal 3DGCNN ran successfully, but it did not outperform the Phase 8C FTV SVM model. Therefore, the FTV SVM remains the current main pilot result, and the 3DGCNN should be reported as an exploratory prototype and future-work direction.

## Model ranking

- Rank 1: Phase 8C FTV SVM baseline / SVM_RBF with AUROC 0.6667, AUPRC 0.7528, Brier 0.2405.
- Rank 2: Phase 9B temporal-delta baseline / RandomForest with AUROC 0.5714, AUPRC 0.6060, Brier 0.2724.
- Rank 3: Stage 12B exploratory temporal 3DGCNN / Exploratory_Temporal_3DGCNN with AUROC 0.5714, AUPRC 0.5897, Brier 0.3909.

## Novel paper idea

A strong paper should focus on tumor-source validation and FTV-based pCR prediction, with an exploratory temporal 3DGCNN prototype. This is stronger than claiming the 3DGCNN is already the best model.

## Why this is novel

The project does not blindly apply deep learning. It first validates the tumor source, rejects weak DICOM SEG masks as the main path, uses official MRI/FTV support features for the primary pilot model, and then builds a graph-ready longitudinal framework for future 3DGCNN work.

## Safe manuscript claim

Official MRI/FTV support features gave the strongest current pilot pCR prediction result. A temporal 3DGCNN prototype was implemented but remains exploratory because of small sample size.

## Claim to avoid

Do not claim that the 3DGCNN is clinically validated or that it is currently better than the FTV SVM model.

## Next step

Use Stage 12C to update the paper outline. The title should mention FTV-based pCR prediction, tumor-source validation, and exploratory temporal 3DGCNN.