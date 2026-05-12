#!/usr/bin/env bash
set -euo pipefail

source /home/mdsubhan01/miniforge3/etc/profile.d/conda.sh
conda activate 3dgcnn

cd "$HOME/ISPY2-3DGCNN"

mkdir -p reports/logs reports/figures reports/tables

python scripts/phase2_seg_mr_linking.py \
  --repo-root "$HOME/ISPY2-3DGCNN" \
  --decode-limit 120 \
  > reports/logs/phase2_seg_mr_linking.log 2>&1

echo "Checking for files larger than 50 MB before git add:"
find . -type f -size +50M ! -path "./.git/*" | tee reports/logs/phase2_large_file_check.txt

if [ -s reports/logs/phase2_large_file_check.txt ]; then
  echo "ERROR: Large files found. Not committing. See reports/logs/phase2_large_file_check.txt"
  exit 1
fi

git add .gitignore
git add configs/phase2_seg_mr_linking_config.yaml
git add scripts/phase2_seg_mr_linking.py
git add run_phase2_seg_mr_linking.sh
git add bootstrap_phase2_seg_mr_linking.sh
git add bootstrap_phase1_4d_vascular_atlas.sh 2>/dev/null || true
git add reports/figures
git add reports/tables
git add reports/Phase2_SEG_MR_Linking_Summary.md

git status --short

if git diff --cached --quiet; then
  echo "No new changes to commit."
else
  git commit -m "Add phase 2 SEG MR linking outputs"
  git push
fi

echo "Phase 2 completed and pushed to GitHub."
