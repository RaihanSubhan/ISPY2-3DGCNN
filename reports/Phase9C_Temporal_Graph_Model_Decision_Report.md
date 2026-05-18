# Phase 9C Temporal Graph Model Decision Report

This phase compares the current best FTV support-data model against the temporal-delta baseline and decides whether a temporal GNN should be the next main model.

## Main result

Phase 8C FTV best model: SVM_RBF, AUROC 0.6667, AUPRC 0.7528, Brier 0.2405.

Phase 9B temporal-delta best model: RandomForest, AUROC 0.5714, AUPRC 0.6060, Brier 0.2724.

## Decision

Decision: **do_not_train_gnn_yet_small_n**.

Do not make temporal GNN the main model yet. Keep Phase 8C FTV SVM as the main pilot result and describe GNN as future work.

## Interpretation

The temporal-delta baseline did not beat the FTV support-data model. This means the temporal graph idea is still scientifically useful, but it should not be the main result yet.

## Why this is the right decision

A temporal GNN needs enough patients and stable node features. The current graph-ready dataset has only 13 patient graphs. Training a deep graph model now would likely overfit and weaken the paper.

## Safe paper statement

The official FTV support-data SVM is the current strongest pilot model. The longitudinal graph dataset is a future-work bridge, not the current main clinical model.

## Next practical step

Prepare the final Word report and PowerPoint using Phase 8C, Phase 8D, Phase 8E, Phase 9A, Phase 9B, and Phase 9C. The future-work section should explain how a larger temporal GNN would be built.