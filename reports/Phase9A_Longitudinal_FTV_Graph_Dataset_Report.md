# Phase 9A Longitudinal FTV Graph Dataset Report

This phase converts the Phase 8C FTV modeling table into a graph-ready longitudinal patient-visit dataset.

## Main outputs

- reports/tables/phase9a_longitudinal_visit_nodes.csv
- reports/tables/phase9a_temporal_edges.csv
- reports/tables/phase9a_patient_graph_summary.csv
- reports/tables/phase9a_temporal_delta_features.csv
- reports/figures/280_phase9a_visit_availability.png
- reports/figures/281_phase9a_ftv_trajectories.png
- reports/figures/282_phase9a_mean_ftv_by_label.png
- reports/figures/283_phase9a_graph_schema.png

## Main counts

- Patient graphs: 13
- Visit nodes: 52
- Temporal edges: 39
- Patient delta rows: 13
- pCR positive graphs: 7
- non-pCR graphs: 6

## Interpretation

The project now has a graph-ready representation. Each patient is represented by visit nodes and temporal edges. This is the correct bridge between the current FTV support-data model and the future temporal graph learning model.

## Limitation

This graph dataset is still small because it uses the 13 labeled patients from Phase 8C. It is ready for method design and pilot testing, not final clinical claims.

## Next step

Phase 9B should train a simple temporal-delta baseline model and compare it with Phase 8C. After that, a temporal graph neural network can be designed.