# Stage 12B Exploratory Temporal 3DGCNN Report

This stage creates a real exploratory temporal graph convolutional model using the Phase 9A graph-ready FTV dataset.

## Why this was created

The attached 3D MRI papers motivate patient-specific 3D modeling and validated segmentation. This code tests a lightweight 3DGCNN-style graph model for breast MRI visits, while keeping the Phase 8C FTV SVM as the current main pilot result.

## Graph design

- One patient = one graph
- One visit = one node
- Temporal edges connect consecutive visits
- Node features include FTV volume, sphericity, longest diameter, BPE, visit index, and feature availability flags

## Main counts

- Patient graphs: 13
- pCR positive graphs: 7
- non-pCR graphs: 6
- Node features: 9

## Modeling result

The exploratory temporal 3DGCNN ran on 13 patient graphs. AUROC = 0.5714, AUPRC = 0.5897, Brier score = 0.3909.

## Important interpretation

This 3DGCNN is exploratory. It should not replace the Phase 8C FTV SVM as the main model unless it clearly outperforms it on a larger cohort.

## Main outputs

- reports/tables/stage12b_3dgcnn_model_performance.csv
- reports/tables/stage12b_3dgcnn_oof_predictions.csv
- reports/tables/stage12b_3dgcnn_vs_existing_models.csv
- reports/figures/510_stage12b_graph_label_distribution.png
- reports/figures/511_stage12b_model_comparison_auroc.png
- reports/figures/514_stage12b_3dgcnn_schema.png

## Next step

If the 3DGCNN underperforms the FTV SVM, keep it as a method-prototype figure. If it improves results, use it as an exploratory future-work experiment, not clinical validation.