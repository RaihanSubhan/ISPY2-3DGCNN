# Stage 10F Professor Meeting Talk Track

## 60-second summary

I built a full ISPY2 breast MRI pipeline on Cradle and saved it in GitHub. I started with raw DICOM and DICOM SEG files, but I found that the SEG masks were not reliable tumor-only masks. Because of that, I moved to official MRI/FTV support-data features. That path gave the strongest pilot pCR prediction result. The best current model is SVM_RBF using FTV support features, with AUROC 0.6667, AUPRC 0.7528, and Brier score 0.2405. I also built a graph-ready longitudinal FTV dataset, but the temporal-delta model was weaker than the FTV model. So I am keeping temporal GNN as future work, not the current main model.

## Questions for professor

1. Is this title acceptable: Pilot FTV-Based Prediction of pCR in I-SPY2 Breast MRI?
2. Should Phase 8C FTV SVM_RBF be the main pilot model?
3. Can I present DICOM SEG as a diagnostic negative result?
4. Should I describe the obstacle map as conceptual contrast-transport disturbance only?
5. Is the limitation statement strong enough: only 13 labeled FTV patients, pilot only?
6. Should temporal GNN be placed as future work instead of current main model?
7. Are Figures 340, 342, 260, 300, and 283 enough for presentation?

## Safe final claim

Official MRI/FTV support-data features produced a stronger pilot pCR prediction signal than the earlier proxy and temporal-delta approaches.

## Claim to avoid

This model is clinically validated or ready for clinical use.