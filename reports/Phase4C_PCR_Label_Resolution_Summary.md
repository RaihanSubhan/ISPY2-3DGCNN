# Phase 4C pCR Label Resolution Summary

Phase 4B did not find a usable two-class pCR or response label automatically. Phase 4C creates a clean label template for the biomarker-ready patients.

## Main outputs

- reports/tables/phase4a_required_label_template.csv
- reports/tables/phase4c_pcr_label_template.csv
- reports/tables/phase4c_label_candidate_review.csv
- reports/table_views/phase4c_pcr_label_template.md
- reports/table_views/phase4c_label_candidate_review.md
- reports/figures/80_phase4c_label_template_status.png
- reports/figures/81_phase4c_label_candidate_review.png

## Main counts

- Patients needing labels: 33
- Patients with pCR_label already filled: 0
- Patients still blank: 33
- Candidate label rows reviewed: 17

## How to fill the template

Fill `pCR_label` using this rule:

- 1 = pCR / responder
- 0 = non-pCR / non-responder
- blank = unknown

## Next step

After pCR labels are filled, rerun Phase 4B. Then baseline ML models can train.