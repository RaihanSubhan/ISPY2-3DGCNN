# Final Send Message to Professor

Subject: ISPY2 breast MRI pilot project package

Dear Professor,

I have completed the first full research cycle for my ISPY2 breast MRI project. The final package includes a Word report, PowerPoint presentation, one-page summary, and review bundle.

The main current result is a pilot pCR prediction model using official MRI/FTV support-data features. The best model is SVM_RBF with AUROC 0.6667, AUPRC 0.7528, and Brier score 0.2405.

A key technical finding is that the raw DICOM SEG mask path was not reliable enough for the main tumor-source pathway, so the project moved toward official MRI/FTV support-data features.

I also built a graph-ready longitudinal FTV dataset, but the temporal-delta model was weaker than the FTV model. Therefore, temporal GNN is included as future work rather than the current main model.

I would appreciate your feedback on the title, main claim, limitation wording, and future graph-learning direction.

Best regards,
Raihan