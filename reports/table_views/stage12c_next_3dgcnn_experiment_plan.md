# Stage 12C Next 3DGCNN Experiment Plan

Rows: 5
Columns: 4

| priority | next_experiment | why | deliverable |
| --- | --- | --- | --- |
| 1 | Keep Phase 8C FTV SVM as main paper result | It ranks best by AUROC, AUPRC, and Brier score. | Main results table and figure. |
| 2 | Report Stage 12B 3DGCNN as exploratory prototype | It demonstrates feasibility but does not outperform the FTV model. | Methods/future-work subsection. |
| 3 | Create larger patient-visit graph cohort | A true 3DGCNN paper needs more than 13 patient graphs. | Expanded graph dataset. |
| 4 | Add richer PE/SER or voxel/region features | The current graph nodes are visit-level features, not true 3D tumor-region nodes. | Tumor-region graph nodes. |
| 5 | Retest GNN only after cohort expansion | Training deep graph models on 13 graphs risks overfitting. | Stage 13 or future paper. |