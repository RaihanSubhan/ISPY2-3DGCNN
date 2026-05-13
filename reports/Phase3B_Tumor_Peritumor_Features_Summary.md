# Phase 3B Tumor and Peritumor Feature Pilot Summary

This phase prepares the linked SEG-MR pairs for verified tumor and peritumor feature extraction.

## Main outputs

- reports/tables/phase3b_tumor_peritumor_features_pilot.csv
- reports/table_views/phase3b_tumor_peritumor_features_pilot.md
- reports/figures/30_phase3b_link_confidence.png
- reports/figures/31_phase3b_feature_readiness.png
- reports/figures/32_phase3b_top_mr_descriptions.png
- reports/figures/33_phase3b_feature_roadmap.png

## Main counts

- SEG-MR pairs checked: 2688
- Unique patients represented: 719
- Basic Phase 3 ready pairs: 2688
- High-confidence links: 0
- Medium-confidence links: 2688
- Weak links: 0
- SEG decode QC success cases available: 120

## Interpretation

This step confirms which linked SEG-MR pairs are ready for tumor and peritumor feature extraction. It does not yet claim final clinical tumor volume.

## Next step

Phase 3C should perform geometry-aware matching between SEG frames and MR slices, then compute verified tumor volume, shape, radial signal, Frangi vesselness, peritumor vessel density, and longitudinal change features.