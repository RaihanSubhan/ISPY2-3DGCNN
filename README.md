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
