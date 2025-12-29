# Inter-Subject Variability in EEG-Based Stress Detection

[![Paper: Accepted](https://img.shields.io/badge/Paper-Accepted%20(IBCD%202026)-success.svg)](#paper-and-citation)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](requirements.txt)
[![Code: Preserved](https://img.shields.io/badge/Code-Reproducible%20Artifact-brightgreen.svg)](#reproducibility)

Machine-learning benchmark and experimental pipeline investigating **EEG-based stress classification** on the SAM-40 dataset under **subject-dependent (SD)** versus **subject-independent (SI)** cross-validation regimes.

> [!NOTE]
> **Paper Status:** Accepted for publication at the **3rd International Conference on Business Intelligence, Computational Mathematics and Data Analytics (IBCD 2026)**. Camera-ready copy submitted; proceedings publication pending. This repository contains the preserved experimental implementation, preprocessing pipelines, feature extraction, and evaluation metrics corresponding to the accepted manuscript.

**Authors:** Prakriti Sharma, [Aaditya Upadhyay](https://github.com/Addy48), Dr. Akshay Jadhav (corresponding)  
**Affiliation:** School of Computer Science and Engineering, Manipal University Jaipur  
**Venue:** IBCD 2026 (Proceedings forthcoming)

---

## Abstract

EEG-based stress detection reaches high accuracy under controlled, subject-dependent evaluation. Cross-subject generalization is substantially harder. We benchmark seven classical machine learning models (KNN, Random Forest, XGBoost, CatBoost, AdaBoost, Linear SVC, Logistic Regression) and three ensemble architectures (Voting, Bagging, Stacking) on the SAM-40 dataset across three cognitive stressors: Mental Arithmetic, Mirror-Image Tracing, and the Stroop Color-Word task.

Subject-dependent accuracy peaks at **97.39%** (KNN) and **92.84%** (Random Forest). In contrast, subject-independent accuracy peaks at **55.29%** (AdaBoost). Complex ensembles fail to close this 42.1 percentage point generalization deficit. Our findings establish that the cross-subject bottleneck lies in individual neural variability and non-stationary feature distributions rather than model capacity.

---

## Key Results

| Model | Validation Protocol | Arithmetic Acc (%) | Mirror Acc (%) | Stroop Acc (%) | Average Acc (%) |
|---|---|---|---|---|---|
| **KNN** | Subject-Dependent (SD) | 97.30 | 97.31 | 97.56 | **97.39** |
| **Random Forest** | Subject-Dependent (SD) | 93.52 | 94.40 | 90.60 | 92.84 |
| **Stacking Ensemble**| Subject-Dependent (SD) | 80.06 | 80.14 | 83.56 | 81.25 |
| **Voting Ensemble**  | Subject-Dependent (SD) | 76.01 | 76.58 | 78.88 | 77.16 |
| **XGBoost** | Subject-Dependent (SD) | 75.34 | 75.11 | 77.32 | 75.92 |
| **CatBoost** | Subject-Dependent (SD) | 71.71 | 78.19 | 73.95 | 74.62 |
| **AdaBoost** | Subject-Independent (SI) | 54.07 | 53.35 | 58.44 | **55.29** |
| **Bagging Ensemble** | Subject-Independent (SI) | 50.68 | 52.92 | 57.42 | 53.67 |
| **CatBoost** | Subject-Independent (SI) | 48.81 | 52.37 | 56.20 | 52.46 |
| **XGBoost** | Subject-Independent (SI) | 48.07 | 51.94 | 54.39 | 51.47 |
| **Random Forest** | Subject-Independent (SI) | 46.12 | 49.62 | 53.36 | 49.70 |
| **KNN** | Subject-Independent (SI) | 44.83 | 47.78 | 49.65 | 47.42 |

**Generalization Gap:** 42.10 percentage points between SD (97.39%) and SI (55.29%).

---

## Repository Structure

```
├── src/
│   ├── preprocessing.py          # Savitzky-Golay & Daubechies-2 wavelet denoising
│   ├── feature_extraction.py     # 10 statistical & information-theoretic metrics
│   └── eeg_classification.py     # Dual-validation (SD vs SI) training & evaluation
├── notebooks/
│   ├── 01_preprocessing_and_wavelets.ipynb
│   ├── 02_feature_extraction_pipeline.ipynb
│   └── 03_model_benchmarking_and_evaluation.ipynb
├── results/
│   ├── summary.csv               # Empirical metrics across all models and tasks
│   └── benchmark_metrics.json    # Machine-readable evaluation summary
├── tests/                        # Unit test suite verifying signal & feature pipelines
├── docs/                         # SAM-40 acquisition and protocol specifications
├── index.html                    # Interactive web presentation dashboard
├── requirements.txt              # Pinned dependencies
├── CITATION.cff                  # Citation metadata
└── README.md
```

---

## Reproducibility Guide

### 1. Environment Setup
```bash
git clone https://github.com/Addy48/EEG-Stress-Detection-.git
cd EEG-Stress-Detection-
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python3 -m pytest
```

### 3. Execute Benchmarking Pipeline
```bash
python3 src/eeg_classification.py --data_path /path/to/SAM40 --results_path ./results
```

---

## Paper and Citation

```bibtex
@inproceedings{sharma2026eeg,
  title={Inter-Subject Variability in EEG-Based Stress Detection: A Machine Learning Perspective},
  author={Sharma, Prakriti and Upadhyay, Aaditya and Jadhav, Akshay},
  booktitle={Proceedings of the 3rd International Conference on Business Intelligence, Computational Mathematics and Data Analytics (IBCD 2026)},
  year={2026},
  note={Accepted for publication}
}
```

For questions regarding the paper or implementation, please contact the authors.
