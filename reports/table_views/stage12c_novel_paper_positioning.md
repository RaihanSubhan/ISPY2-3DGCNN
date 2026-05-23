# Stage 12C Novel Paper Positioning

Rows: 5
Columns: 4

| novelty_axis | what_you_did | why_it_matters | paper_role |
| --- | --- | --- | --- |
| Tumor-source validation | Diagnosed DICOM SEG masks and showed the raw SEG path was not reliable enough for main ... | Many imaging ML projects assume tumor masks are valid. Your work checks the tumor sourc... | Core contribution |
| Official MRI/FTV support-data modeling | Selected official MRI/FTV support features and trained pCR prediction models. | FTV features gave the strongest pilot signal. | Main result |
| Graph-ready longitudinal representation | Built patient-visit graphs with visit nodes and temporal edges. | This creates the bridge toward future temporal 3DGCNN/GNN models. | Future-work bridge |
| Exploratory temporal 3DGCNN | Implemented a lightweight graph convolution model over patient visits. | It proves the model framework can run on Cradle, but it does not beat FTV SVM yet. | Prototype, not main claim |
| 3D visual explanation | Created breast-pair, contrast-transport proxy, 3D tumor surface, vessel schematic, and ... | These help explain the biological intuition and future graph idea. | Methods visualization |