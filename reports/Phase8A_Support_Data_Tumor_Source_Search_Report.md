# Phase 8A Support-Data Tumor Source Search Report

This phase searches for official or local support-data tumor sources after Phase 7F showed that recovered DICOM SEG masks were too limited for reliable supervised ML.

## Main outputs

- reports/tables/phase8a_support_source_candidates.csv
- reports/tables/phase8a_support_page_scan.csv
- reports/tables/phase8a_support_download_log.csv
- reports/table_views/phase8a_support_source_candidates.md
- reports/table_views/phase8a_support_download_log.md
- reports/figures/230_phase8a_candidate_types.png
- reports/figures/231_phase8a_top_candidates.png

## Main counts

- Total support candidates: 141
- FTV candidates: 4
- PE/SER candidates: 23
- ROI candidates: 0
- mask/segmentation candidates: 52
- supervoxel candidates: 0

## Interpretation

Potential support-data sources were found. Open the candidate table and choose the highest-priority ROI, FTV, PE/SER, or supervoxel source for Phase 8B.

## Next step

Phase 8B should inspect the best support-data candidate and build a real tumor-source manifest. Do not return to Phase 6 or Phase 7 ML until a stronger tumor source is identified.