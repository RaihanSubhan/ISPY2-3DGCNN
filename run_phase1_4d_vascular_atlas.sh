#!/usr/bin/env bash
set -euo pipefail

source /home/mdsubhan01/miniforge3/etc/profile.d/conda.sh
conda activate 3dgcnn

cd "$HOME/ISPY2-3DGCNN"

DATA_ROOT="$HOME/ispy2/ispy2"

mkdir -p reports/logs reports/figures reports/tables

python scripts/phase1_4d_vascular_atlas.py \
  --data-root "$DATA_ROOT" \
  --out-dir reports \
  --max-patients 0 \
  --max-pixel-files 120 \
  > reports/logs/phase1_4d_vascular_atlas.log 2>&1

echo "Checking for files larger than 50 MB before git add:"
find . -type f -size +50M | tee reports/logs/large_file_check.txt

if [ -s reports/logs/large_file_check.txt ]; then
  echo "ERROR: Large files found. Not committing. See reports/logs/large_file_check.txt"
  exit 1
fi

git add requirements_atlas.txt
git add configs/vascular_atlas_config.yaml
git add scripts/phase1_4d_vascular_atlas.py
git add run_phase1_4d_vascular_atlas.sh
git add reports/figures
git add reports/tables
git add reports/Phase1_Output_Summary.md

git status --short

if git diff --cached --quiet; then
  echo "No new changes to commit."
else
  git commit -m "Add phase 1 4D vascular perfusion atlas outputs"
  git push
fi

echo "Phase 1 completed and pushed to GitHub."
