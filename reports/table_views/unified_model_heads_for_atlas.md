# unified_model_heads_for_atlas.csv

Original CSV: `reports/tables/unified_model_heads_for_atlas.csv`

Rows: 4
Columns: 5

This page shows the first 50 rows and first 10 columns only.
Open the CSV file for the full GitHub interactive table.

| model_head | input | output | role | why |
| --- | --- | --- | --- | --- |
| Radiomics + XGBoost | tumor volume, shape, radial profile, vessel density, visit deltas | pCR or early response probability | primary interpretable prediction model | best first model for structured atlas biomarkers |
| Temporal visit graph | patient visits, missing timepoints, longitudinal features | patient trajectory embedding | longitudinal tracking and missingness modeling | captures patient-level time structure |
| 3D CNN or 3D U-Net | DCE-MRI volume and verified mask labels | tumor mask or image embedding | future automated segmentation and representation learning | use after mask quality control, not before |
| Offline RL | state = atlas features, action = treatment arm, reward = response/toxicity | policy hypothesis | later-stage treatment policy simulation | only valid after careful causal/offline policy design |
