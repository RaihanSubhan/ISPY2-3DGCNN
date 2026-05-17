# Phase 8E Final Project Status

Rows: 7
Columns: 5

| phase | name | status | main_result | paper_use |
| --- | --- | --- | --- | --- |
| Phase 1 | DICOM inventory and visit tracking | complete | Built cohort-level ISPY2 inventory and visit tables. | Methods: data inventory and cohort construction. |
| Phase 2 | SEG-to-MR linking | complete but diagnostic | Linked SEG files to MR series, but high-confidence tumor masks were not confirmed. | Methods and limitation. |
| Phase 7A-7F | DICOM SEG diagnosis and recovered-mask path | diagnostic, not main result | DICOM SEG files were mostly full-slice masks. Fractional thresholding recovered limited... | Important negative result and rationale for support-data path. |
| Phase 8A-8B | Support-data tumor source search | complete | Identified official MRI/FTV support-data source as stronger tumor-feature path. | Methods: tumor feature source selection. |
| Phase 8C | FTV support-data modeling | complete pilot | SVM_RBF achieved AUROC 0.6667 and AUPRC 0.7528 on 13 labeled patients. | Pilot results. |
| Phase 8D | FTV vs baseline comparison | complete | FTV path improved AUROC by 0.1205 over Phase 5A baseline. | Main pilot comparison result. |
| Future work | Temporal graph / 3DGCNN direction | planned | Use stable FTV/PE/SER patient-visit features first, then temporal graph learning. | Future work and novelty. |