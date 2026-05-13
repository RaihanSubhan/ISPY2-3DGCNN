# Phase 3C Geometry Matching QC Summary

This phase checks whether decoded SEG frames can be matched to MR slices using DICOM z-position geometry.

## Main outputs

- reports/tables/phase3c_geometry_matching_qc.csv
- reports/table_views/phase3c_geometry_matching_qc.md
- reports/figures/40_phase3c_match_status.png
- reports/figures/41_phase3c_z_error_hist.png
- reports/figures/42_phase3c_within_2mm.png
- reports/figures/43_phase3c_example_overlay.png

## Main counts

- Cases checked: 120
- Geometry-ready cases: 43
- Warning cases: 77
- Failed cases: 0

## Interpretation

This is a QC step before final biomarker extraction. Geometry-ready cases can move toward verified tumor volume, shape, radial signal, and vesselness extraction.

## Next step

Phase 3D should compute final verified tumor and peritumor biomarkers only for geometry-ready cases.