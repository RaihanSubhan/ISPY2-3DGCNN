# ISPY2-3DGCNN

This repository stores code, scripts, SLURM job files, notebooks, reports, and notes for ISPY2 breast MRI research on Cradle/HPC.

## Dataset

The ISPY2 dataset is stored on Cradle and is not included in this GitHub repository.

Dataset path on Cradle:

/home/mdsubhan01/ispy2

## Environment

To activate the Cradle Python environment:

source /home/mdsubhan01/miniforge3/etc/profile.d/conda.sh
conda activate 3dgcnn

## Important Rule

Do not commit DICOM files, NIfTI files, model checkpoints, processed data, logs, or the full ISPY2 dataset.

### Phase 3B: tumor and peritumor feature pilot

Main outputs:

- reports/tables/phase3b_tumor_peritumor_features_pilot.csv
- reports/table_views/phase3b_tumor_peritumor_features_pilot.md
- reports/Phase3B_Tumor_Peritumor_Features_Summary.md

### Phase 3D: verified biomarker table

Main outputs:

- reports/tables/phase3d_verified_biomarker_table.csv
- reports/table_views/phase3d_verified_biomarker_table.md
- reports/Phase3D_Verified_Biomarker_Table_Summary.md

### Phase 4A: label and modeling readiness

Main outputs:

- reports/tables/phase4a_label_source_inventory.csv
- reports/tables/phase4a_required_label_template.csv
- reports/Phase4A_Label_and_Modeling_Readiness_Summary.md

### Phase 4C: pCR label resolution

Main outputs:

- reports/tables/phase4a_required_label_template.csv
- reports/tables/phase4c_pcr_label_template.csv
- reports/Phase4C_PCR_Label_Resolution_Summary.md

### Phase 5A: model calibration and interpretation

Main outputs:

- reports/tables/phase5a_model_calibration_metrics.csv
- reports/tables/phase5a_feature_importance.csv
- reports/Phase5A_Model_Calibration_Interpretation_Summary.md
