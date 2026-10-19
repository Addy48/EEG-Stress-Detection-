"""Load the SAM40 feature CSVs used in the experiments.

Layout expected under ``data_root``::

    Relax/        Relax_sub_1.csv ... Relax_sub_40.csv
    Arithmetic/   Arithmetic_sub_1.csv ...
    Mirror_image/ Mirror_image_sub_1.csv ...
    Stroop/       Stroop_sub_1.csv ...

Each CSV is 9,600 rows x 33 columns: an index column followed by 32 numeric
columns. The index column is dropped, as in the original notebooks.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

CONDITION_FOLDERS = {
    "Relax": "Relax",
    "Arithmetic": "Arithmetic",
    "Mirror": "Mirror_image",
    "Stroop": "Stroop",
}

TASKS = {
    "Relax-Arithmetic": ("Relax", "Arithmetic"),
    "Relax-Mirror": ("Relax", "Mirror"),
    "Relax-Stroop": ("Relax", "Stroop"),
}

GROUP_MODES = ("file", "subject")

_SUFFIX = re.compile(r"_sub_?(\d+)\.csv$", re.IGNORECASE)


def _list_files(folder: Path) -> list[str]:
    files = sorted(str(p) for p in folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}")
    return files


def load_condition(
    data_root: str | Path,
    condition: str,
    label: int,
    group_mode: str = "file",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X, y, groups)`` for one condition folder.

    ``group_mode="file"`` reproduces the grouping used for the reported results:
    every CSV is its own group, keyed by class label and file order
    (``label0_sub3``, ``label1_sub3``, ...). The Relax file and the task file
    with the same suffix therefore get *different* group keys and can land on
    opposite sides of a fold.

    ``group_mode="subject"`` keys groups by the ``_sub_N`` filename suffix only,
    so the Relax and task files that share a suffix are always held out together.
    This assumes suffix N is the same person in every condition folder, which the
    dataset files themselves do not prove (see docs/PROVENANCE.md).
    """
    if group_mode not in GROUP_MODES:
        raise ValueError(f"group_mode must be one of {GROUP_MODES}")
    folder = Path(data_root) / CONDITION_FOLDERS[condition]
    frames, groups = [], []
    for idx, path in enumerate(_list_files(folder)):
        df = pd.read_csv(path)
        if df.shape[1] > 1:
            df = df.iloc[:, 1:]  # drop the index column
        frames.append(df)
        if group_mode == "file":
            key = f"label{label}_sub{idx}"
        else:
            match = _SUFFIX.search(path)
            if not match:
                raise ValueError(f"Cannot read a subject number from {path}")
            key = f"sub{int(match.group(1))}"
        groups.append(np.full(len(df), key))
    X = pd.concat(frames, ignore_index=True).to_numpy(dtype=np.float32)
    return X, np.full(len(X), label), np.concatenate(groups)


def load_task(
    data_root: str | Path, task: str, group_mode: str = "file"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Relax (label 0) against one task condition (label 1)."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {sorted(TASKS)}")
    rest, stress = TASKS[task]
    X0, y0, g0 = load_condition(data_root, rest, 0, group_mode)
    X1, y1, g1 = load_condition(data_root, stress, 1, group_mode)
    return np.vstack((X0, X1)), np.hstack((y0, y1)), np.hstack((g0, g1))
