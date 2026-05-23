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

### Phase 7F: expanded recovered tumor cohort

Main outputs:

- reports/Phase7F_Expanded_Recovered_Tumor_Cohort_Report.md
- reports/tables/phase7f_expanded_recovered_cohort_summary.csv
- reports/figures/220_phase7f_expansion_summary.png

### Phase 8A: support-data tumor source search

Main outputs:

- reports/Phase8A_Support_Data_Tumor_Source_Search_Report.md
- reports/tables/phase8a_support_source_candidates.csv
- reports/figures/230_phase8a_candidate_types.png

### Phase 8B: tumor source manifest

Main outputs:

- reports/Phase8B_Tumor_Source_Manifest_Report.md
- reports/tables/phase8b_tumor_source_manifest.csv
- reports/tables/phase8b_support_source_triage.csv

### Phase 8C: FTV feature modeling

Main outputs:

- reports/Phase8C_FTV_Feature_Modeling_Report.md
- reports/tables/phase8c_ftv_modeling_dataset.csv
- reports/tables/phase8c_ftv_model_performance.csv

### Phase 8D: FTV vs baseline manuscript-ready result

Main outputs:

- reports/Phase8D_FTV_vs_Baseline_Manuscript_Result.md
- reports/tables/phase8d_baseline_vs_ftv_model_comparison.csv
- reports/figures/260_phase8d_baseline_vs_ftv_metrics.png

### Phase 8E: final research proposal and manuscript plan

Main outputs:

- reports/Phase8E_Final_Research_Proposal_and_Manuscript_Plan.md
- reports/tables/phase8e_final_project_status.csv
- reports/figures/270_phase8e_project_pipeline.png

### Phase 9A: longitudinal FTV graph-ready dataset

Main outputs:

- reports/Phase9A_Longitudinal_FTV_Graph_Dataset_Report.md
- reports/tables/phase9a_longitudinal_visit_nodes.csv
- reports/tables/phase9a_temporal_edges.csv

### Phase 9B: temporal-delta baseline model

Main outputs:

- reports/Phase9B_Temporal_Delta_Baseline_Report.md
- reports/tables/phase9b_temporal_delta_model_performance.csv
- reports/tables/phase9b_vs_phase8c_comparison.csv

### Phase 9C: temporal graph model decision

Main outputs:

- reports/Phase9C_Temporal_Graph_Model_Decision_Report.md
- reports/tables/phase9c_model_decision_summary.csv
- reports/figures/300_phase9c_model_decision.png

### Stage 10A: dataset structure and visual preprocessing

Main outputs:

- reports/Stage10A_Dataset_Visual_Preprocessing_Report.md
- reports/figures/311_stage10a_preprocess_without_with_mask_grid.png
- reports/figures/312_stage10a_segmentation_top_bottom_grid.png
- reports/figures/313_stage10a_breast_pair_obstacle_grid.png

### Stage 10B: expanded preprocessing visual outputs

Main outputs:

- reports/Stage10B_Expanded_Preprocessed_Visual_Output_Report.md
- reports/figures/320_stage10b_without_mask_grid.png
- reports/figures/321_stage10b_with_mask_grid.png
- reports/figures/322_stage10b_four_panel_grid.png

### Stage 10C: breast-pair and contrast-transport visualization

Main outputs:

- reports/Stage10C_Breast_Pair_Obstacle_Visualization_Report.md
- reports/figures/330_stage10c_breast_pair_grid.png
- reports/figures/331_stage10c_obstacle_proxy_grid.png
- reports/figures/332_stage10c_transport_field_grid.png

### Stage 10D: thesis methods figure package

Main outputs:

- reports/Stage10D_Thesis_Methods_Figure_Package_Report.md
- reports/figures/340_stage10d_master_methods_panel.png
- reports/figures/341_stage10d_selected_case_contact_sheet.png
- reports/figures/342_stage10d_final_pipeline_and_model_result.png

### Stage 10E: final Word report and PowerPoint

Main outputs:

- reports/final_package/ISPY2_FTV_pCR_Final_Report.docx
- reports/final_package/ISPY2_FTV_pCR_Final_Presentation.pptx
- reports/Stage10E_Final_Word_PowerPoint_Package_Report.md

### Stage 10F: final package QC and professor review

Main outputs:

- reports/Stage10F_Final_Package_QC_Professor_Review_Report.md
- reports/Final_GitHub_Output_Index.md
- reports/final_package/ISPY2_FTV_pCR_Professor_Review_Bundle.zip

### Stage 10G: final professor handoff fix

Main outputs:

- reports/Stage10G_Final_Handoff_Fix_Report.md
- reports/final_package/ISPY2_FTV_pCR_Professor_Review_Bundle.zip
- reports/final_package/Professor_Email_Draft.md

### Stage 10H: final repository freeze

Main outputs:

- FINAL_STATUS.md
- reports/Stage10H_Final_Repository_Freeze_Report.md
- reports/final_package/RELEASE_NOTES_v1.0-pilot-ftv-pcr.md

### Stage 10I: post-release polish

Main outputs:

- reports/Stage10I_Post_Release_Polish_Report.md
- reports/final_package/FINAL_HANDOFF_READY.md
- reports/tables/stage10i_post_release_polish_qc.csv

### Stage 10J: final after-push verification

Main outputs:

- reports/Stage10J_Final_After_Push_Verification_Report.md
- reports/tables/stage10j_final_after_push_qc.csv
- reports/tables/stage10j_release_status.csv

### Stage 10K: final visual methods QC

Main outputs:

- reports/Stage10K_Final_Visual_Methods_QC_Report.md
- reports/tables/stage10k_four_task_completion_checklist.csv
- reports/figures/401_stage10k_best_visual_case_panel.png

### Stage 11A: 3D flow-vessel-growth visualization

Main outputs:

- reports/Stage11A_3D_Flow_Vessel_Growth_Visualization_Report.md
- reports/figures/410_stage11a_bidirectional_flow_proxy.png
- reports/figures/411_stage11a_3d_tumor_vessel_schematic.png
- reports/figures/412_stage11a_multivisit_3d_tumor_growth.png
