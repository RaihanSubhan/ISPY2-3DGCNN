# Final Report Package

## Title
Pilot FTV-Based Prediction of Pathologic Complete Response in I-SPY2 Breast MRI with Tumor-Source Validation and a Future Temporal Graph Learning Framework

## Main idea
This project built a reproducible Cradle-to-GitHub pipeline for I-SPY2 breast MRI, tested DICOM SEG masks, found that the raw SEG path was not reliable enough for the main model, and then moved to official MRI/FTV support-data features for pCR prediction.

## Main result
The best current pilot model is SVM_RBF using official MRI/FTV support features. It achieved AUROC 0.6667, AUPRC 0.7528, and Brier score 0.2405 on 13 labeled patients.

## Model decision
The temporal-delta model used RandomForest and reached AUROC 0.5714. Since it was weaker than the FTV model, temporal GNN should be future work, not the main current model.

## Safe claim
Official MRI/FTV support-data features produced a stronger pilot pCR prediction signal than the earlier proxy and temporal-delta approaches.

## Limitation
This is pilot work only. It is not clinical validation because the strongest FTV model used a small labeled cohort.

## Final outputs
- final_package/ISPY2_FTV_pCR_Final_Report.docx
- final_package/ISPY2_FTV_pCR_Final_Presentation.pptx
- final_package/ISPY2_FTV_pCR_One_Page_Summary.md
- reports/Stage10E_Final_Word_PowerPoint_Package_Report.md
