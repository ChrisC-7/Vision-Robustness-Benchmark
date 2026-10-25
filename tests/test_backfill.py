"""Tests for src.backfill."""

from src import tracking
from src.backfill import backfill_experiment


def test_backfill_records_every_run_and_condition(experiment_dir):
    conn = tracking.get_connection(":memory:")

    run_dirs = backfill_experiment(conn, str(experiment_dir))

    assert run_dirs == [
        str(experiment_dir / "simple_s42"),
        str(experiment_dir / "simple_s123"),
        str(experiment_dir / "stronger_s42"),
    ]
    blur = tracking.query_eval_results(conn, model_name="stronger", condition="blur")
    assert len(blur) == 1
    assert blur[0]["accuracy"] == 60.0
    assert blur[0]["noise_seed"] == 42


def test_backfill_twice_does_not_duplicate(experiment_dir):
    conn = tracking.get_connection(":memory:")

    backfill_experiment(conn, str(experiment_dir))
    backfill_experiment(conn, str(experiment_dir))

    assert conn.execute("SELECT COUNT(*) FROM train_runs").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM eval_results").fetchone()[0] == 9
