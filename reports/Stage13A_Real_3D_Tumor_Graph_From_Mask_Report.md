# Stage 13A Real 3D Tumor Graph From Recovered Mask Report

This stage fixes the problem that earlier 3D tumor visuals looked schematic or unrealistic.

## What this stage does

- Uses the recovered tumor mask instead of an artificial ellipsoid.
- Builds a real 3D tumor surface mesh using marching cubes.
- Converts tumor voxels into graph nodes using spatial clustering.
- Connects nodes using 3D nearest-neighbor edges.
- Creates static, animated, and interactive visual outputs.

## Main outputs

- reports/tables/stage13a_tumor_graph_nodes.csv
- reports/tables/stage13a_tumor_graph_edges.csv
- reports/figures/600_stage13a_real_tumor_surface_graph.png
- reports/figures/601_stage13a_tumor_graph_zoom.png
- reports/figures/602_stage13a_graph_node_feature_map.png
- reports/figures/603_stage13a_rotating_tumor_graph.gif
- reports/figures/605_stage13a_tumor_graph_schema.png
- reports/interactive/604_stage13a_interactive_tumor_graph.html

## Selected tumor graph case

- Patient: ISPY2-109623
- Mask mode: real_recovered_3d_mask
- Tumor voxels: 284829
- Graph nodes: 40
- Graph edges: 91
- Interactive HTML created: True

## Important interpretation

This is now a true tumor graph visualization because the graph is built from recovered tumor-mask voxels. It is not a fake ellipsoid.

## Limitation

The mask is still a recovered fractional mask. If mask_mode is pseudo_3d_from_real_2d_mask_outline, then the 3D thickness is a visualization approximation based on the real 2D outline.

## Safe wording

Use: real-mask-derived tumor graph, spatial tumor-node graph, and contrast-transport proxy.

Avoid: measured blood flow, milk flow, vessel graph, or clinically validated 3D tumor segmentation.

## Next step

Use Stage 13A tumor graph figures as the main 3D tumor graph visuals. Do not use earlier ellipsoid figures as final tumor-shape evidence.