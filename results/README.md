# Saved results

Everything here was produced by the notebooks in `../notebooks` and is kept as saved. See `../docs/PROVENANCE.md` for how each folder was made and what it covers.

| Folder | Contents |
|---|---|
| `si_baseline/<Model>/<Task>/` | Subject-independent baselines (file-group `GroupKFold`): `metrics.csv` (fold means), `classification_report.csv` (pooled), `confusion_matrix.png`. All 21 model/task cells. |
| `sd_baseline/<Model>/<Task>/` | Subject-dependent baselines for Mirror and Stroop (14 cells). Arithmetic has no valid detailed export. |
| `ensembles/` | Voting and Bagging (SI and SD) and Stacking (SD) accuracy per task; `stacking_si/` has the per-fold Stacking (SI) results and confusion matrices. |
| `aggregates/` | Aggregate accuracy summaries (all 7 models x 3 tasks) for SD and SI. |
| `smote_pca_exploratory/` | Exploratory SMOTE + PCA runs, with the baseline accuracy beside each. |
| `TABLES.md` | Tables rebuilt from the files above by `scripts/build_tables.py`. |
