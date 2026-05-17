# Phase 8B Tumor Source Manifest Report

This phase cleans the Phase 8A support-candidate list and separates real external/support sources from generated project outputs.

## Main outputs

- reports/tables/phase8b_support_source_triage.csv
- reports/tables/phase8b_tumor_source_manifest.csv
- reports/table_views/phase8b_support_source_triage.md
- reports/table_views/phase8b_tumor_source_manifest.md
- reports/figures/240_phase8b_source_classes.png
- reports/figures/241_phase8b_top_source_scores.png
- reports/figures/242_phase8b_manifest_decisions.png

## Main counts

- Total Phase 8A candidates triaged: 141
- Generated project outputs excluded: 109
- Documentation-only sources: 3
- Primary candidates: 22
- Secondary candidates: 0
- Manifest rows kept: 25

## Interpretation

Phase 8B found one or more primary tumor-source candidates. Inspect the manifest and select the strongest source for Phase 8C.

## Next step

Run Phase 8C to build a patient/visit tumor-source manifest from the highest-priority primary candidate.