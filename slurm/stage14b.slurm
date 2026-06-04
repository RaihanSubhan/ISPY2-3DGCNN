#!/bin/bash
#SBATCH --job-name=stage14b
#SBATCH --partition=kimq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=reports/logs/stage14b_%j.out
#SBATCH --error=reports/logs/stage14b_%j.err

source /home/mdsubhan01/miniforge3/etc/profile.d/conda.sh
conda activate 3dgcnn
cd "$HOME/ISPY2-3DGCNN"
python scripts/stage14b_cohort_3d_tumor_gallery.py --n 6
echo "Stage 14B finished with exit code $?"
