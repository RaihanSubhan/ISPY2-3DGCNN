# Stage 14A Cohort Tumor-Graph Feature Dataset Report

Expands the Stage 13 single-case 3D tumor graph (ISPY2-109623) into a cohort-level
tumor-graph feature dataset from the recovered fractional tumor masks. Each patient's
best-quality recovered mask becomes a spatial tumor-node graph with the same method as
Stage 13 (marching-cubes surface, KMeans nodes, spatial kNN edges).

## Graph settings
- Max nodes per graph: 50
- kNN per node: 5

## Cohort counts
- Patients with a usable tumor graph: 44
- Built from a real recovered 3D mask: 44
- Built from a pseudo-3D extrusion of a real 2D outline: 0

## Labeled cohort (the decision gate)
- Patients with a tumor graph AND a pCR label: 2
- pCR-positive graphs: 1
- non-pCR graphs: 1

## Decision
- Verdict code: **NOT_ENOUGH_keep_3dgcnn_as_future_work**
- NOT enough labeled tumor graphs for graph-feature modelling or a 3DGCNN. This is the empirical justification for keeping the temporal 3DGCNN as future work. The cohort tumor-graph dataset is a feasibility asset, ready to grow with more labels.

## Main outputs
- reports/tables/stage14a_cohort_graph_features.csv
- reports/tables/stage14a_cohort_label_merge.csv
- reports/tables/stage14a_cohort_tumor_graph_nodes.csv
- reports/tables/stage14a_cohort_tumor_graph_edges.csv
- reports/figures/700_stage14a_cohort_graph_overview.png

## Safe wording
Use: cohort tumor-graph feature dataset, spatial tumor-node graph, recovered-mask-derived graph.
Avoid: measured blood flow, vessel graph, clinically validated 3D tumor segmentation.

## Limitation
Tumor graphs come from recovered fractional masks, not expert segmentations. Graph edges are
spatial links, not vessels or measured flow. Pilot feasibility work.
