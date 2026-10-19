# EEG stress detection on SAM 40: subject-dependent vs subject-independent evaluation

[![tests](https://github.com/Addy48/EEG-Stress-Detection-/actions/workflows/tests.yml/badge.svg)](https://github.com/Addy48/EEG-Stress-Detection-/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

Code and saved results for the manuscript *Inter-Subject Variability in EEG-Based Stress Detection: A Machine Learning Perspective* (Sharma, Upadhyay, Jadhav; School of Computer Science and Engineering, Manipal University Jaipur). The manuscript is under submission; this repository will link the published version when there is one.

The question: how much does accuracy drop when the classifier is not allowed to see the same recording during training? Seven classical models and three ensembles are evaluated on three binary tasks from the public [SAM 40](docs/DATA.md) dataset (Relax vs Arithmetic, Mirror image, Stroop), under a subject-dependent split and a file-group (subject-independent) split.

## Results at a glance

Accuracy (%), mean over five folds and the three tasks. Full per-task tables are in [`results/TABLES.md`](results/TABLES.md).

| Model | Subject-dependent | File-group (SI) |
|---|---:|---:|
| KNN | 97.39 | 47.42 |
| Random Forest | 92.84 | 49.70 |
| XGBoost | 75.92 | 51.47 |
| CatBoost | 74.62 | 52.46 |
| AdaBoost | 63.77 | 55.29 |
| Linear SVC | 61.38 | 50.51 |
| Logistic Regression | 61.29 | 50.13 |

Voting, Bagging and Stacking are in `TABLES.md`; none of them beats the best single model under SI (best ensemble 53.67%, Bagging).

How to read these numbers:

- **Subject-dependent** (SD) is a shuffled row-level split. Neighbouring time samples of the same recording land in both train and test, so it measures within-recording interpolation.
- **File-group SI** holds out whole files, but each Relax file and task file is its own group. Matching `_sub_N` files can still end up on opposite sides of a fold, so this is **not a confirmed person-held-out estimate**. Details and the other caveats are in [`docs/PROVENANCE.md`](docs/PROVENANCE.md).
- Each row the models see is one time sample of 32 EEG channels ([`docs/DATA.md`](docs/DATA.md)), and all results are for this one dataset.

## Repository layout

```
src/eeg_stress/   data loading, the 7 baselines, SD/SI evaluation, ensembles, CLI
notebooks/        the two notebooks that produced the saved results (outputs kept)
results/          saved metrics, reports and confusion matrices; TABLES.md
scripts/          build_tables.py rebuilds the tables from results/
docs/             DATA.md (what a row is) and PROVENANCE.md (what made what, limits)
tests/            unit tests for loading, splitting and the saved tables
```

## Run it

```bash
git clone https://github.com/Addy48/EEG-Stress-Detection-.git
cd EEG-Stress-Detection-
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m pytest -q
python scripts/build_tables.py            # rebuild the tables from the saved results
```

To re-run an experiment you need the SAM 40 CSVs laid out as described in [`docs/DATA.md`](docs/DATA.md):

```bash
# subject-independent (file-group) baselines for one task and two models
eeg-stress --data-root /path/to/SAM40 --protocol si --task Relax-Arithmetic --models LogisticRegression KNN --out runs

# subject-dependent
eeg-stress --data-root /path/to/SAM40 --protocol sd --task Relax-Stroop --out runs
```

The full seven-model run on all three tasks takes many hours (the saved Colab runs were budgeted at about six). Re-running `LogisticRegression` on Relax-Arithmetic reproduces the saved metrics file exactly.

`--group-mode subject` groups the Relax and task files that share a `_sub_N` suffix. It assumes the suffix is a person ID, is not used for any saved result, and is there for follow-up work.

## Data

SAM 40 is not redistributed. Get it from the dataset paper and cite it:

> Ghosh, R., Deb, N., Sengupta, K., et al. SAM 40: Dataset of 40 subject EEG recordings to monitor the induced-stress while performing Stroop color-word test, arithmetic task, and mirror image recognition task. *Data in Brief* 40 (2022) 107772. [doi:10.1016/j.dib.2021.107772](https://doi.org/10.1016/j.dib.2021.107772)

## Citing

Please cite this repository (see [`CITATION.cff`](CITATION.cff)). Cite the manuscript once it is published.

## License

Code: [MIT](LICENSE). The SAM 40 dataset has its own terms.
