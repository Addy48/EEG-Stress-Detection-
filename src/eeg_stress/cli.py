"""Command line entry point: ``python -m eeg_stress.cli --help``."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .data import GROUP_MODES, TASKS, load_task
from .evaluate import PROTOCOLS, evaluate
from .models import MODEL_NAMES, build_models


def _save(out_dir: Path, model: str, task: str, protocol: str, group_mode: str, res) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mean, std = res.mean, res.std
    row = {"Model": model, "Task": task, "Protocol": protocol.upper(), "GroupMode": group_mode if protocol == "si" else "n/a"}
    row.update({m.capitalize() if m != "auc" else "AUC": v for m, v in mean.items()})
    row.update({f"{m}_std": v for m, v in std.items()})
    pd.DataFrame([row]).to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(res.report).transpose().to_csv(out_dir / "classification_report.csv")
    pd.DataFrame(res.confusion, index=["Relax", "Task"], columns=["Relax", "Task"]).to_csv(out_dir / "confusion_matrix.csv")
    (out_dir / "folds.json").write_text(json.dumps(res.fold_metrics, indent=2))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Run the SAM40 baseline classifiers under SD or SI cross-validation.")
    p.add_argument("--data-root", required=True, help="Folder holding Relax/, Arithmetic/, Mirror_image/, Stroop/")
    p.add_argument("--task", choices=sorted(TASKS), nargs="+", default=sorted(TASKS))
    p.add_argument("--protocol", choices=PROTOCOLS, default="si")
    p.add_argument("--group-mode", choices=GROUP_MODES, default="file",
                   help="file = grouping used for the reported results; subject = shared _sub_N key (see docs/PROVENANCE.md)")
    p.add_argument("--models", choices=MODEL_NAMES, nargs="+", default=list(MODEL_NAMES))
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--out", default="runs", help="Output folder")
    args = p.parse_args(argv)

    for task in args.task:
        print(f"== {task} | {args.protocol.upper()} | group-mode={args.group_mode}")
        X, y, groups = load_task(args.data_root, task, args.group_mode)
        print(f"   {len(X):,} rows x {X.shape[1]} columns, {len(set(groups))} groups")
        for name in args.models:
            start = time.time()
            model = build_models(args.n_jobs)[name]
            res = evaluate(model, X, y, groups, args.protocol, verbose=True)
            _save(Path(args.out) / args.protocol / args.group_mode / name / task, name, task, args.protocol, args.group_mode, res)
            print(f"   {name}: accuracy={res.mean['accuracy']:.4f} +/- {res.std['accuracy']:.4f} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
