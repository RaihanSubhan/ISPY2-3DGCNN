# Phase 8E Final Research Proposal and Manuscript Plan

## Final research direction

The strongest current direction is not to continue with weak DICOM SEG masks. The stronger path is to use official MRI / FTV support-data features for pCR prediction, then extend the work toward temporal graph learning once stable longitudinal tumor features are available.

## Working title

Pilot FTV-Based Prediction of Pathologic Complete Response in I-SPY2 Breast MRI with a Future Temporal Graph Learning Framework

## What has been done

A full Cradle-to-GitHub pipeline was built. The project downloaded ISPY2 data, created a DICOM inventory, linked SEG and MR data, diagnosed segmentation problems, recovered limited fractional tumor masks, found official MRI/FTV support data, mapped pCR labels, and trained pilot ML models.

## Key technical lesson

The DICOM SEG files were not reliable tumor-only masks for the main modeling pipeline. Phase 7 showed that many masks behaved like full-slice analysis masks. This changed the project direction toward official support-data FTV / MRI features.

## Main result

The Phase 5A baseline best model was LogisticRegression, with AUROC 0.5462, AUPRC 0.7506, and Brier score 0.2662.

The Phase 8C FTV best model was SVM_RBF, with AUROC 0.6667, AUPRC 0.7528, and Brier score 0.2405.

The FTV path improved AUROC by 0.1205. It improved AUPRC by 0.0021. It reduced Brier score by 0.0257.

## Safe conclusion

Official MRI / FTV support-data features produced a stronger pilot pCR prediction signal than the earlier internal proxy biomarker pipeline.

## Main limitation

This is still a pilot result. The Phase 8C FTV model used only 13 labeled patients. It should not be described as clinically validated.

## Recommended manuscript structure

### Title
Pilot FTV-Based pCR Prediction and Tumor-Source Validation in I-SPY2 Breast MRI

### Background
Breast MRI can monitor treatment response during neoadjuvant therapy, but reliable tumor-source selection is needed before model building.

### Problem
Initial DICOM SEG masks were not reliable tumor-only masks because they often represented full-slice analysis masks.

### Hypothesis
Official MRI/FTV support-data features provide a cleaner pilot signal for pCR prediction than weak DICOM SEG-derived proxy features.

### Methods
Download ISPY2 on Cradle, build DICOM inventory, diagnose SEG masks, search support data, build FTV feature table, merge pCR labels, and train baseline ML.

### Models
Logistic regression, SVM, random forest, gradient boosting, and other classical ML models were compared.

### Main result
FTV SVM_RBF model achieved AUROC 0.6667, AUPRC 0.7528, and Brier 0.2405 in a small pilot cohort.

### Limitations
Only 13 labeled FTV patients were used. This is not clinical validation.

### Future work
Build temporal graph learning using longitudinal FTV, PE/SER, and tumor-change features once more stable patient-visit features are available.

## Future temporal graph plan

The next scientific step is to build a longitudinal patient-visit graph. Each patient can be represented by serial visits. Each visit can include FTV, PE/SER, tumor shape, and treatment-response features. Edges can encode temporal order and similarity of tumor-change patterns. A temporal graph model can then be tested against the FTV SVM baseline.

## Next practical action

Use this Phase 8E report as the base for a Word proposal and PowerPoint presentation. Then plan a new technical phase focused on longitudinal FTV / PE-SER graph modeling.