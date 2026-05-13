# Phase 4D Label Source Deep Inspection Summary

This phase inspects candidate pCR or response-label files more deeply.

## Main outputs

- reports/tables/phase4d_label_candidate_deep_inspection.csv
- reports/tables/phase4d_label_candidate_preview.csv
- reports/table_views/phase4d_label_candidate_preview.md
- reports/figures/90_phase4d_candidate_overlap.png

## Main counts

- Candidate rows inspected: 17
- Candidate files read successfully: 17
- Candidates with label-like columns: 13

## What to do next

Open `reports/tables/phase4d_label_candidate_deep_inspection.csv` and `reports/table_views/phase4d_label_candidate_preview.md` in GitHub.

Look for a real outcome column such as pCR, pathologic complete response, RCB, responder, or response.

If a true pCR column is visible, we can map it into `reports/tables/phase4c_pcr_label_template.csv` and rerun Phase 4B.

If no true pCR column is visible, then the official clinical support data still needs to be downloaded or obtained.