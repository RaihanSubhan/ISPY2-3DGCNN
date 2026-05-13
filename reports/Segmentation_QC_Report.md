# Segmentation QC Report

This report audits whether the DICOM SEG masks are visually and geometrically usable for tumor biomarker extraction.

## Important conclusion

This step verifies existing DICOM SEG masks. It does not train a new segmentation model.

## Main outputs

- reports/tables/segmentation_qc_case_audit.csv
- reports/table_views/segmentation_qc_case_audit.md
- reports/figures/130_segmentation_qc_overlay_grid.png
- reports/figures/131_segmentation_qc_status_counts.png
- reports/figures/132_segmentation_qc_volume_proxy.png
- reports/figures/segmentation_qc_case_001.png and related case images

## Main counts

- Cases audited: 12
- Visual QC pass candidates: 12
- Review-needed cases: 0
- Failed cases: 0

## Interpretation

The segmentation workflow appears usable for pilot analysis, but full manual review is still needed before final claims.

## Next action

Open the overlay grid and each case image in GitHub. Check whether the contour falls inside the breast tumor region and not outside the image or on the wrong anatomy.

If many overlays are unclear, improve MR series selection or SEG-to-MR matching before extracting final biomarkers.