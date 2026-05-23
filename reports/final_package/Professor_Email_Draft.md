# Professor Email Draft

Subject: ISPY2 breast MRI pilot results and proposal draft

Dear Professor,

I have completed the first full research cycle for my ISPY2 breast MRI project. The main current result is a pilot pCR prediction model using official MRI/FTV support-data features. The best model is SVM_RBF with AUROC 0.6667, AUPRC 0.7528, and Brier score 0.2405.

During the work, I also tested the raw DICOM SEG mask path and found that it was not reliable enough as the main tumor-source path. Many masks behaved like full-slice analysis masks. Because of that, I moved to the official FTV/MRI support-data features as the main modeling source.

I also built a graph-ready longitudinal FTV dataset and tested a temporal-delta baseline. That model was weaker than the FTV support-data model, so I am treating temporal GNN as future work rather than the current main model.

I am attaching the Word report and PowerPoint. I would like your feedback on the title, the main claim, the limitation wording, and whether the future temporal graph direction is appropriate.

Best regards,
Raihan
