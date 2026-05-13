# Phase 5A Model Recommendation

Rows: 1
Columns: 7

| recommended_model | selection_rule | AUROC | AUPRC | Brier_score | why | xgboost_available |
| --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression | highest AUROC, then highest AUPRC, then lowest Brier score | 0.5461538461538461 | 0.7506391256164644 | 0.26616661003690567 | This is a pilot model. It gives the best cross-validated discrimination among the teste... | not_available |