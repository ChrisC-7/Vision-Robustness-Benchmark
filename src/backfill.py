"""Backfill a finished multi-seed experiment into the tracking database.

:func:`multiseed.run_multiseed` trains and evaluates every
(model, seed) pair and writes ``<experiment_dir>/config.json`` and
``result.jsonl``, but never records anything with :mod:`tracking`, so the
API's ``/runs`` and ``/results`` would have nothing to serve. This reads
those files -- it does not retrain or re-evaluate -- and records them.

Run from the repository root, since ``result.jsonl`` stores each
``run_dir`` relative to it:

    python -m src.backfill --experiment-dir runs/multiseed/experiment1
"""

import argparse
import json
import sqlite3
from pathlib import Path

from . import tracking
from .engine import EvalConfig


def load_eval_config(experiment_dir: Path) -> EvalConfig:
    """Rebuilds the ``EvalConfig`` an experiment's evaluations ran with."""
    config_data = json.loads((experiment_dir / "config.json").read_text())
    eval_data = config_data["eval_config"]
    eval_data["conditions"] = tuple(eval_data["conditions"])
    return EvalConfig(**eval_data)


def backfill_experiment(conn: sqlite3.Connection, experiment_dir: str) -> list[str]:
    """Records every run and evaluation in one experiment's ``result.jsonl``.

    Safe to re-run. ``record_train_run`` already updates by ``run_dir``, and
    each evaluation is recorded under a fixed ``eval_id`` derived from its
    ``run_dir`` -- one ``result.jsonl`` line is one evaluation attempt, so
    recording it again is a retry, not a new attempt.

    Args:
        conn: An open connection from :func:`tracking.get_connection`.
        experiment_dir: The directory :func:`multiseed.run_multiseed` wrote,
            e.g. ``runs/multiseed/experiment1``.

    Returns:
        The ``run_dir`` of every run recorded, in ``result.jsonl`` order.
    """
    experiment_dir = Path(experiment_dir)
    config = load_eval_config(experiment_dir)

    run_dirs = []
    for line in (experiment_dir / "result.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        eval_result = json.loads(line)
        run_dir = tracking.record_train_run(conn, eval_result["run_dir"])
        tracking.record_eval_run(
            conn, run_dir, eval_result, config, eval_id=f"backfill:{run_dir}"
        )
        run_dirs.append(run_dir)
    return run_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", default="runs/multiseed/experiment1")
    parser.add_argument("--db", default="experiments.db")
    args = parser.parse_args()

    conn = tracking.get_connection(args.db)
    recorded = backfill_experiment(conn, args.experiment_dir)
    conn.close()
    print(f"Recorded {len(recorded)} runs from {args.experiment_dir} into {args.db}")
