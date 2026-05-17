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

### Phase 5B: manuscript-ready pilot report

Main outputs:

- reports/Phase5B_Manuscript_Ready_Results_Report.md
- reports/tables/phase5b_key_results_summary.csv
- reports/figures/120_phase5b_pipeline_summary.png

### Segmentation QC fix

Main outputs:

- reports/Segmentation_QC_Report.md
- reports/tables/segmentation_qc_case_audit.csv
- reports/figures/130_segmentation_qc_overlay_grid.png

### Phase 6A: QC-pass tumor and vascular features

Main outputs:

- reports/tables/phase6a_qc_pass_tumor_vascular_features.csv
- reports/Phase6A_QC_Pass_Tumor_Vascular_Feature_Summary.md

### Phase 7A: tumor SEG decoding fix

Main outputs:

- reports/Phase7A_Tumor_SEG_Decoding_Fix_Report.md
- reports/tables/phase7a_tumor_mask_candidate_table.csv
- reports/figures/170_phase7a_tumor_mask_overlay_grid.png

### Phase 7B: deep SEG diagnosis

Main outputs:

- reports/Phase7B_Deep_SEG_Diagnosis_Report.md
- reports/tables/phase7b_seg_case_deep_diagnosis.csv
- reports/tables/phase7b_seg_segment_deep_diagnosis.csv

### Phase 7C: fractional SEG threshold recovery

Main outputs:

- reports/Phase7C_Fractional_SEG_Threshold_Recovery_Report.md
- reports/tables/phase7c_fractional_tumor_mask_candidate_table.csv
- reports/figures/192_phase7c_fractional_overlay_grid.png

### Phase 7D: recovered tumor feature rebuild

Main outputs:

- reports/Phase7D_Recovered_Tumor_Feature_Rebuild_Report.md
- reports/tables/phase7d_recovered_tumor_vascular_features.csv
- reports/figures/200_phase7d_recovered_overlay_grid.png

### Phase 7E: recovered tumor pCR feasibility

Main outputs:

- reports/Phase7E_Recovered_Tumor_pCR_Feasibility_Report.md
- reports/tables/phase7e_recovered_tumor_label_merge.csv
- reports/tables/phase7e_recovered_tumor_model_performance.csv
