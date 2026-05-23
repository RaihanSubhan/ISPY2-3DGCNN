# Stage 13B Publication-Grade 3D Tumor Graph QC Report

This stage refines the Stage 13A real 3D tumor graph into a cleaner publication-grade graph and adds graph quality control.

## What was improved

- Smoothed the real-mask-derived tumor surface.
- Rebuilt graph nodes with spatial clustering.
- Ensured graph connectivity.
- Added graph QC metrics, degree distribution, and edge-length distribution.
- Added tensor-ready node and edge tables for future 3DGCNN experiments.

## Main outputs

- reports/figures/610_stage13b_publication_tumor_graph_panel.png
- reports/figures/611_stage13b_node_degree_distribution.png
- reports/figures/612_stage13b_edge_length_distribution.png
- reports/figures/613_stage13b_node_feature_map.png
- reports/figures/614_stage13b_rotating_publication_tumor_graph.gif
- reports/figures/615_stage13b_publication_graph_schema.png
- reports/interactive/616_stage13b_publication_tumor_graph.html
- reports/tables/stage13b_publication_tumor_graph_nodes.csv
- reports/tables/stage13b_publication_tumor_graph_edges.csv
- reports/tables/stage13b_graph_qc_metrics.csv
- reports/tables/stage13b_graph_tensor_features.csv
- reports/tables/stage13b_graph_tensor_edge_index.csv

## Selected graph

- Patient: ISPY2-109623
- Mask mode: real_recovered_3d_mask
- Graph nodes: 50
- Graph edges: 143
- Connected components: 1.0
- Mean degree: 5.72
- Mean edge distance: 0.20189429361412814
- Interactive HTML created: True

## Interpretation

This is the cleanest current 3D tumor graph output. It should replace earlier ellipsoid or schematic tumor-shape figures when discussing the 3DGCNN direction.

## Limitation

The tumor graph is derived from a recovered fractional mask, not a manual expert segmentation. Graph edges are spatial links, not vessels or measured flow.

## Next step

Use Stage 13B as the final 3D tumor graph visualization. If more work is needed, the next scientific step is to train a graph model on multiple patient tumor graphs, not just one visual case.