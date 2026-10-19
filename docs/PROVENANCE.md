# Provenance and limits

What produced each saved result, how the splits work, and what these results do and do not show. Read this before citing a number.

## Where each result comes from

| Result | Produced by | Saved in |
|---|---|---|
| Baselines, subject-independent (SI), 7 models x 3 tasks | `notebooks/01_baseline_and_extensions.ipynb`, cells 9-12; same logic as `src/eeg_stress` | `results/si_baseline/` |
| Baselines, subject-dependent (SD), Mirror and Stroop | same notebook, cells 14-20 | `results/sd_baseline/` |
| Baselines, SD, Arithmetic | aggregate summary only | `results/aggregates/SubjectDependent_AllModels.csv` |
| Voting and Bagging (SI and SD), Stacking (SD) | `notebooks/02_ensembles.ipynb` | `results/ensembles/` |
| Stacking (SI) | `notebooks/01_baseline_and_extensions.ipynb`, cells 25-27 | `results/ensembles/stacking_si/` |
| SMOTE + PCA (exploratory) | same notebook, cell 24 | `results/smote_pca_exploratory/` |

`python scripts/build_tables.py` rebuilds the baseline and ensemble tables from these files (`results/TABLES.md`).

The notebooks are Colab / local-runtime exports. Their outputs are kept as saved, but most code cells carry no execution counter, so they document what was run rather than serving as an automatic reproduction. The data path was replaced with a placeholder.

## Re-running

`src/eeg_stress` is a cleaned version of the baseline notebook cells (same models, same settings, same splitting, same scaling). On 19 Oct 2026, `LogisticRegression` on Relax-Arithmetic under the SI protocol was re-run from the raw CSVs and reproduced the saved metrics file exactly (accuracy 0.4649921875, precision 0.47466, recall 0.58017, F1 0.51588, AUC 0.45009).

One supplementary check, **not part of the saved results or the manuscript**: on the same task and model, `--group-mode subject` (matching `_sub_N` files held out together) gave a five-fold mean accuracy of 0.5332 (folds 0.521 to 0.550), against 0.4650 with the file-group keys. So the grouping choice changes the SI number noticeably, and the file-group figures should not be read as the person-held-out accuracy. Only this one model and task were run.

## How the splits work

**SI (`--protocol si`).** Five-fold `GroupKFold`. In the notebooks, the group key is built per file and per class (`label0_sub3`, `label1_sub3`, ...). That gives 80 groups per task (40 Relax files + 40 task files), so each fold holds out 16 file groups and trains on 64. Two things follow:

- A Relax file and the task file with the same `_sub_N` suffix get different keys, so they can fall on opposite sides of a fold. The dataset files do not include a manifest that proves suffix N is the same person in every condition folder, and the notebooks do not join the two files by a common ID.
- The reported SI numbers are therefore best described as **five-fold file-group** results, not as a confirmed leave-subjects-out estimate.

`--group-mode subject` keys groups by the filename suffix only, so matching Relax and task files always stay together. It assumes the suffix is a person ID. It is provided for follow-up work and was **not** used for any saved result.

**SD (`--protocol sd`).** Five-fold shuffled `StratifiedKFold` (`random_state=42`) over individual rows. Rows from the same recording, including neighbouring time samples, can appear in both train and test, so SD accuracy measures within-recording interpolation, not performance on new people.

**Both protocols.** `StandardScaler` is fitted on the training fold only. The index column is dropped. Rows are single time samples (see `docs/DATA.md`), so they are strongly dependent on their neighbours; fold-level and row-level statistics should not be read as independent samples.

## Metrics

- `metrics.csv` is the mean over the five folds (accuracy, precision, recall, F1, AUC; class 1 = task, class 0 = Relax).
- `classification_report.csv` and the confusion matrices pool the held-out predictions of all five folds (384,000 rows per class), so their accuracy can differ slightly from the fold mean.
- Some SI AUC values are below 0.5 (for example Logistic Regression on Relax-Arithmetic, 0.450). That is what the saved run produced and is not adjusted.

## Known gaps and caveats

- **SD Arithmetic** has no usable detailed export. Three exports that were labelled Arithmetic were duplicates of a Random Forest Stroop result and were left out. The Arithmetic SD numbers in the tables come from the aggregate summary and are marked with a dagger.
- **Stacking** (SD in notebook 02, SI in notebook 01) trains the logistic-regression meta-learner on base-model predictions for the same rows the base models were fitted on, instead of out-of-fold predictions. The held-out fold stays untouched, but the meta-learner sees over-confident inputs. The code was kept as run so that it matches the saved numbers.
- **SMOTE + PCA** is a separate exploratory branch. SMOTE and PCA are fitted on training folds only; if SMOTE raises an error the code silently falls back to the unbalanced training fold. Its results are not part of the baseline tables.
- **Statistics, feature importance, diversity and per-group analyses** in notebook 01 (cells 28-37) are exploratory. Their output files are not included, because they were not produced by a documented, reproducible pipeline tied to the baseline results.
- **One dataset.** All results are on SAM 40 with the protocols above. They do not establish how any model would perform on other datasets or on people outside it.
- The notebooks fix `random_state=42` for the models and splits, but the saved Colab runs predate this repository, so a fresh run can differ slightly across library versions and hardware.

## What was removed from the earlier version of this repository

An earlier version of this repository held a Savitzky-Golay + wavelet preprocessing module, a 14-channel x 10-feature extractor, a demonstration classifier and a results table. Those files did not generate the numbers above (the saved experiments classify 32-channel samples directly) and were removed from the working tree; they remain in the git history before the "Rebuild" commit.
