"""Demo of the SQLite tracking layer against real, already-trained runs.

Requires two trained runs under ``runs/`` -- this repo does not ship
checkpoints, so train them first if you have not already:

    python -m src.train --model simple --run-name simple_s42 \\
        --training-seed 42 --split-seed 42 --epochs 30
    python -m src.train --model stronger --run-name stronger_s42 \\
        --training-seed 42 --split-seed 42 --epochs 30

Then, from the repository root (so the ``data/`` and ``runs/`` relative
paths that eval.py already relies on resolve correctly):

    python scripts/tracking_demo.py

This does not retrain anything by itself. It backfills the two runs above,
runs one real evaluation against ``runs/simple_s42``'s existing checkpoint,
and then walks through the retry / independent-attempt / query behavior
described in ``tracking.py``. Writes ``experiments_demo.db`` in the current
directory; delete it any time, it is regenerated from scratch on each run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import tracking
from src.engine import EvalConfig
from src.eval import eval_run

DB_PATH = "experiments_demo.db"
TRAIN_RUN_DIRS = ["runs/simple_s42", "runs/stronger_s42"]


def _require_run_dirs(run_dirs: list[str]) -> None:
    missing = [d for d in run_dirs if not (Path(d) / "config.json").exists()]
    if missing:
        raise SystemExit(
            "Missing trained run(s): "
            + ", ".join(missing)
            + ". Train them first -- see the commands in this script's module"
            " docstring -- then re-run this demo."
        )


def main() -> None:
    _require_run_dirs(TRAIN_RUN_DIRS)
    Path(DB_PATH).unlink(missing_ok=True)
    conn = tracking.get_connection(DB_PATH)

    print(f"== Backfilling already-trained runs into {DB_PATH} (no retraining) ==")
    for run_dir in TRAIN_RUN_DIRS:
        tracking.record_train_run(conn, run_dir)
        print(f"  recorded {run_dir}")

    print("\n== Running one real evaluation against runs/simple_s42's checkpoint ==")
    config = EvalConfig(noise_seed=7, std=0.15)
    eval_result = eval_run("runs/simple_s42", config)
    eval_id = tracking.record_eval_run(conn, "runs/simple_s42", eval_result, config)
    print(f"  recorded eval attempt {eval_id}")
    for condition, outcome in eval_result["conditions"].items():
        print(f"    {condition}: accuracy={outcome['accuracy']:.2f}")

    print("\n== Retrying the exact same write with the same eval_id ==")
    tracking.record_eval_run(conn, "runs/simple_s42", eval_result, config, eval_id=eval_id)
    count = conn.execute(
        "SELECT COUNT(*) FROM eval_runs WHERE eval_id = ?", (eval_id,)
    ).fetchone()[0]
    print(f"  rows for eval_id {eval_id}: {count} (expected 1 -- retry did not duplicate)")

    print("\n== Re-running the same config as a brand-new, independent attempt ==")
    new_eval_id = tracking.record_eval_run(conn, "runs/simple_s42", eval_result, config)
    total = conn.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0]
    print(f"  new attempt id: {new_eval_id} (!= {eval_id})")
    print(f"  total eval_runs rows: {total} (expected 2 -- first attempt + this new one)")

    print("\n== Query: noise condition with std=0.15 ==")
    for row in tracking.query_eval_results(conn, condition="noise", std=0.15):
        print(f"  {row['run_name']} eval_id={row['eval_id'][:8]} accuracy={row['accuracy']:.2f}")

    print("\n== Query: everything recorded for model=simple ==")
    for row in tracking.query_eval_results(conn, model_name="simple"):
        print(
            f"  {row['run_name']} eval_id={row['eval_id'][:8]} "
            f"condition={row['condition']} accuracy={row['accuracy']:.2f}"
        )

    conn.close()


if __name__ == "__main__":
    main()
