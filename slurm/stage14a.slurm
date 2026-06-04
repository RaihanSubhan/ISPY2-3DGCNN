#!/bin/bash
#SBATCH --job-name=stage14a
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=reports/logs/stage14a_%j.out
#SBATCH --error=reports/logs/stage14a_%j.err

source /home/mdsubhan01/miniforge3/etc/profile.d/conda.sh
conda activate 3dgcnn
cd "$HOME/ISPY2-3DGCNN"

python scripts/stage14a_cohort_tumor_graph_feature_dataset.py \
  --label-table reports/tables/phase4c_pcr_label_template.csv

echo "Stage 14A finished with exit code $?"
