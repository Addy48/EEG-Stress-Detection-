import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_tables", ROOT / "scripts" / "build_tables.py")
build_tables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_tables)


def test_headline_numbers_match_saved_files():
    base = build_tables.baseline_table().set_index("Model")
    assert base.loc["AdaBoost", "SI avg"] == "55.29"
    assert base.loc["KNN", "SD avg"] == "97.39"
    ens, _ = build_tables.ensemble_table()
    row = ens[(ens.Method == "Stacking") & (ens.Protocol == "SD")].iloc[0]
    assert row["Avg"] == "81.25"


def test_si_reports_have_384000_rows_per_class():
    pc = build_tables.si_per_class()
    assert (pc["Support/class"] == 384000).all()
    assert len(pc) == 21
