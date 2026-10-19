import numpy as np
import pandas as pd
import pytest

from eeg_stress.data import CONDITION_FOLDERS, load_condition, load_task


def _write_condition(root, condition, n_files=4, rows=6, cols=5, seed=0):
    folder = root / CONDITION_FOLDERS[condition]
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for n in range(1, n_files + 1):
        df = pd.DataFrame(rng.normal(size=(rows, cols)), columns=[str(i) for i in range(cols)])
        df.insert(0, "", np.arange(rows))
        df.to_csv(folder / f"{condition if condition != 'Mirror' else 'Mirror_image'}_sub_{n}.csv", index=False)


@pytest.fixture
def root(tmp_path):
    for cond in ("Relax", "Arithmetic"):
        _write_condition(tmp_path, cond)
    return tmp_path


def test_index_column_is_dropped(root):
    X, y, _ = load_condition(root, "Relax", 0)
    assert X.shape == (24, 5)
    assert set(y) == {0}


def test_file_mode_keeps_classes_apart(root):
    _, _, groups = load_task(root, "Relax-Arithmetic", "file")
    assert len(set(groups)) == 8
    assert "label0_sub0" in groups and "label1_sub0" in groups


def test_subject_mode_joins_matching_suffixes(root):
    _, y, groups = load_task(root, "Relax-Arithmetic", "subject")
    assert len(set(groups)) == 4
    for g in set(groups):
        assert set(y[groups == g]) == {0, 1}


def test_unknown_task_or_mode(root):
    with pytest.raises(ValueError):
        load_task(root, "Relax-Nothing")
    with pytest.raises(ValueError):
        load_condition(root, "Relax", 0, group_mode="nope")
