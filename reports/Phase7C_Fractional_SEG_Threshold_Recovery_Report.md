# Phase 7C Fractional SEG Threshold Recovery Report

This phase tests whether the full-slice DICOM SEG problem can be fixed by thresholding fractional SEG values instead of using all nonzero pixels.

## Why this was needed

Phase 7B showed that `arr > 0` selected full-slice analysis masks. Since these SEG files are marked FRACTIONAL, the tumor-like region may require a higher threshold.

## Main outputs

- reports/tables/phase7c_fractional_threshold_audit.csv
- reports/tables/phase7c_fractional_tumor_mask_candidate_table.csv
- reports/table_views/phase7c_fractional_threshold_audit.md
- reports/table_views/phase7c_fractional_tumor_mask_candidate_table.md
- reports/figures/190_phase7c_recovery_status.png
- reports/figures/191_phase7c_selected_mask_fraction.png
- reports/figures/192_phase7c_fractional_overlay_grid.png

## Main counts

- Cases tested: 2688
- Fractional tumor candidate pass cases: 94
- No-threshold-candidate cases: 2339
- Review-needed cases: 255
- Failed cases: 0

## Interpretation

Fractional thresholding recovered tumor-like mask candidates. Open the overlay grid and visually inspect whether contours sit on tumor regions.

## Next step

Run Phase 7D to rebuild tumor and vascular features using only Phase 7C pass cases.