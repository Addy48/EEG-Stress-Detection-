"""
EEG Stress Classification Pipeline
-----------------------------------
Classification methodology for EEG-based stress detection using the SAM-40 dataset.

Authors: Prakriti Sharma, Aaditya Upadhyay, Dr. Akshay Jadhav
Manipal University Jaipur
"""

import os
import glob
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

COMPARISONS = [
    ("Relax", "Arithmetic", "Relax-Arithmetic"),
    ("Relax", "Mirror", "Relax-Mirror"),
    ("Relax", "Stroop", "Relax-Stroop"),
]

BASE_MODELS = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "LinearSVC": LinearSVC(max_iter=2000, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
}

def load_data_with_groups(folder_path, label):
    all_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    df_list = []
    groups_list = []
    for group_id, file_path in enumerate(all_files):
        df = pd.read_csv(file_path)
        df_list.append(df)
        groups_list.append(np.full(len(df), f"label{label}_sub{group_id}"))
    X = pd.concat(df_list, axis=0, ignore_index=True).values
    y = np.full(len(X), label)
    groups = np.hstack(groups_list)
    return X, y, groups
