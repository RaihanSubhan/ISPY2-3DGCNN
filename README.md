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
