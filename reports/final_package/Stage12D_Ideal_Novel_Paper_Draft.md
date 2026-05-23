# Tumor-Source Validation and FTV-Based pCR Prediction in I-SPY2 Breast MRI with an Exploratory Temporal 3DGCNN Framework

## Abstract draft

This pilot study develops a reproducible Cradle-to-GitHub pipeline for I-SPY2 breast MRI analysis. The study first evaluates tumor-source reliability, showing that raw DICOM SEG masks are not strong enough as the main tumor-source pathway. It then uses official MRI/FTV support-data features for pCR prediction and implements an exploratory temporal 3DGCNN framework over patient-visit graphs. The strongest current model is an FTV support-data SVM_RBF model, with AUROC 0.6667, AUPRC 0.7528, and Brier score 0.2405. The temporal 3DGCNN prototype ran successfully but did not outperform the FTV SVM model. Therefore, FTV-based prediction is the current main pilot result, while temporal 3DGCNN is positioned as future work.

## Main hypothesis

Official MRI/FTV support features provide a more reliable pilot pCR prediction signal than weak DICOM SEG-derived features, and these features can later support temporal 3DGCNN modeling when a larger longitudinal graph cohort is available.

## Main contribution

1. Tumor-source validation before modeling.
2. Official MRI/FTV support-data pCR modeling.
3. Graph-ready longitudinal patient-visit representation.
4. Exploratory temporal 3DGCNN prototype.
5. Visual 3D tumor and transport-proxy explanation.

## Main result

- Rank 1: Phase 8C FTV support-data model using SVM_RBF gave AUROC 0.6667, AUPRC 0.7528, and Brier 0.2405.
- Rank 2: Phase 9B temporal-delta baseline using RandomForest gave AUROC 0.5714, AUPRC 0.6060, and Brier 0.2724.
- Rank 3: Stage 12B exploratory temporal 3DGCNN using Exploratory_Temporal_3DGCNN gave AUROC 0.5714, AUPRC 0.5897, and Brier 0.3909.

## Safe conclusion

The official MRI/FTV support-data SVM model is the strongest current pilot model. The temporal 3DGCNN is a working exploratory framework, but it is not yet the main model.

## Limitation

This is pilot feasibility work based on a small labeled cohort. It is not clinical validation.

## Future work

Future work should expand the longitudinal graph cohort and add richer PE/SER, tumor-region, and validated segmentation features before training a full 3DGCNN as the main model.