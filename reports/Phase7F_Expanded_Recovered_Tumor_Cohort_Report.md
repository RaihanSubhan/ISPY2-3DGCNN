# Phase 7F Expanded Recovered Tumor Cohort Report

This phase reruns Phase 7C, Phase 7D, and Phase 7E on a larger SEG cohort.

## Main outputs

- reports/tables/phase7f_expanded_recovered_cohort_summary.csv
- reports/table_views/phase7f_expanded_recovered_cohort_summary.md
- reports/figures/220_phase7f_expansion_summary.png
- reports/figures/221_phase7f_label_counts.png
- reports/figures/222_phase7f_mask_fraction_distribution.png

## Main counts

- Phase 7C | SEG cases tested: 2688
- Phase 7C | fractional tumor candidate pass: 94
- Phase 7C | no threshold candidate: 2339
- Phase 7C | review candidates: 255
- Phase 7C | failed: 0
- Phase 7D | feature rows: 94
- Phase 7D | feature success rows: 94
- Phase 7D | unique patients: 44
- Phase 7E | labeled modeling patients: 2
- Phase 7E | pCR positive: 1
- Phase 7E | non-pCR negative: 1
- Phase 7E | ML status: not_run
- Phase 7E | best model: 
- Phase 7E | best AUROC: 
- Phase 7E | best AUPRC: 

## Interpretation

The expanded recovered tumor cohort is still too small for reliable supervised ML.
If this remains true after all SEG cases are tested, the project should switch to official support-data ROI, FTV, PE, or SER maps instead of relying only on DICOM SEG fractional recovery.

## Next step

If counts are still low, run Phase 8A to search support-data ROI/FTV/PE/SER tumor sources.