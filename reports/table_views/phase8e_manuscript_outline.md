# Phase 8E Manuscript Outline

Rows: 9
Columns: 3

| section | content | purpose |
| --- | --- | --- |
| Title | Pilot FTV-Based pCR Prediction and Tumor-Source Validation in I-SPY2 Breast MRI | Make the project focus clear. |
| Background | Breast MRI can monitor treatment response during neoadjuvant therapy, but reliable tumo... | Explain why this work matters. |
| Problem | Initial DICOM SEG masks were not reliable tumor-only masks because they often represent... | Explain the key technical challenge. |
| Hypothesis | Official MRI/FTV support-data features provide a cleaner pilot signal for pCR predictio... | State the testable idea. |
| Methods | Download ISPY2 on Cradle, build DICOM inventory, diagnose SEG masks, search support dat... | Summarize the pipeline. |
| Models | Logistic regression, SVM, random forest, gradient boosting, and other classical ML mode... | Explain model choice. |
| Main result | FTV SVM_RBF model achieved AUROC 0.6667, AUPRC 0.7528, and Brier 0.2405 in a small pilo... | Show the key result. |
| Limitations | Only 13 labeled FTV patients were used. This is not clinical validation. | Keep claims honest. |
| Future work | Build temporal graph learning using longitudinal FTV, PE/SER, and tumor-change features... | Connect to 3DGCNN / temporal graph direction. |