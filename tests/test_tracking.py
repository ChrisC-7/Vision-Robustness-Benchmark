"""Tests for vision_robustness.tracking."""

import json

import pytest

from src import tracking
from src.engine import EvalConfig


def _write_train_run(
    tmp_path, run_name, model_name="simple", training_seed=42, best_accuracy=90.0
):
    """Writes a fake but realistically-shaped run directory under tmp_path."""
    run_dir = tmp_path / run_name
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "training_seed": training_seed,
                "split_seed": 42,
                "batch_size": 64,
                "learning_rate": 0.03,
                "momentum": 0.9,
                "epochs": 2,
                "device": "cpu",
            }
        )
    )
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "epoch": 1,
                        "train_loss": 1.0,
                        "val_loss": 1.0,
                        "val_accuracy": 50.0,
                        "timestamp": 0,
                    }
                ),
                json.dumps(
                    {
                        "epoch": 2,
                        "train_loss": 0.8,
                        "val_loss": 0.9,
                        "val_accuracy": best_accuracy,
                        "timestamp": 1,
                    }
                ),
            ]
        )
    )
    (run_dir / "best.pt").write_bytes(b"fake-checkpoint")
    return run_dir


def _eval_result(run_dir, model_name="simple"):
    """Builds a fake return value shaped like eval.eval_run's."""
    return {
        "run_dir": str(run_dir),
        "model": model_name,
        "conditions": {
            "clean": {"loss": 1.0, "accuracy": 80.0},
            "blur": {"loss": 1.2, "accuracy": 70.0},
            "noise": {"loss": 1.5, "accuracy": 60.0},
        },
    }


@pytest.fixture
def conn():
    return tracking.get_connection(":memory:")


def test_record_train_run_reads_config_and_best_epoch(tmp_path, conn):
    run_dir = _write_train_run(tmp_path, "simple_s42", best_accuracy=91.0)

    tracking.record_train_run(conn, str(run_dir))

    row = conn.execute(
        "SELECT * FROM train_runs WHERE run_dir = ?", (str(run_dir),)
    ).fetchone()
    assert row["model_name"] == "simple"
    assert row["training_seed"] == 42
    assert row["best_epoch"] == 2
    assert row["best_val_accuracy"] == 91.0
    assert row["checkpoint_path"] == str(run_dir / "best.pt")


def test_record_train_run_marks_missing_fields_unknown(tmp_path, conn):
    run_dir = tmp_path / "legacy_run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({"model_name": "simple", "seed": 42}))

    tracking.record_train_run(conn, str(run_dir))

    row = conn.execute(
        "SELECT * FROM train_runs WHERE run_dir = ?", (str(run_dir),)
    ).fetchone()
    assert row["training_seed"] is None
    assert row["split_seed"] is None
    assert row["best_epoch"] is None
    assert row["checkpoint_path"] is None


def test_record_train_run_retry_updates_in_place(tmp_path, conn):
    run_dir = _write_train_run(tmp_path, "simple_s42")

    tracking.record_train_run(conn, str(run_dir))
    tracking.record_train_run(conn, str(run_dir))

    count = conn.execute("SELECT COUNT(*) FROM train_runs").fetchone()[0]
    assert count == 1


def test_record_eval_run_resolves_actual_params_including_defaults(tmp_path, conn):
    run_dir = _write_train_run(tmp_path, "simple_s42")
    tracking.record_train_run(conn, str(run_dir))
    config = EvalConfig(kernel_size=7, noise_seed=123, std=0.2)

    tracking.record_eval_run(conn, str(run_dir), _eval_result(run_dir), config)

    blur_params = json.loads(
        conn.execute(
            "SELECT params_json FROM eval_results WHERE condition = 'blur'"
        ).fetchone()["params_json"]
    )
    # sigma is never in EvalConfig; it must come from the transform's own default.
    assert blur_params == {"kernel_size": 7, "sigma": 1.0}

    noise_params = json.loads(
        conn.execute(
            "SELECT params_json FROM eval_results WHERE condition = 'noise'"
        ).fetchone()["params_json"]
    )
    assert noise_params == {"noise_seed": 123, "mean": 0.0, "std": 0.2}

    clean_params = json.loads(
        conn.execute(
            "SELECT params_json FROM eval_results WHERE condition = 'clean'"
        ).fetchone()["params_json"]
    )
    assert clean_params == {}


def test_record_eval_run_retry_with_same_id_does_not_duplicate(tmp_path, conn):
    run_dir = _write_train_run(tmp_path, "simple_s42")
    tracking.record_train_run(conn, str(run_dir))
    config = EvalConfig()

    eval_id = tracking.record_eval_run(conn, str(run_dir), _eval_result(run_dir), config)
    tracking.record_eval_run(
        conn, str(run_dir), _eval_result(run_dir), config, eval_id=eval_id
    )

    assert conn.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM eval_results").fetchone()[0] == 3


def test_record_eval_run_without_id_creates_independent_attempt(tmp_path, conn):
    run_dir = _write_train_run(tmp_path, "simple_s42")
    tracking.record_train_run(conn, str(run_dir))
    config = EvalConfig()

    first_id = tracking.record_eval_run(conn, str(run_dir), _eval_result(run_dir), config)
    second_id = tracking.record_eval_run(conn, str(run_dir), _eval_result(run_dir), config)

    assert first_id != second_id
    assert conn.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM eval_results").fetchone()[0] == 6


def test_query_eval_results_filters_by_model_seed_condition_and_param(tmp_path, conn):
    simple_dir = _write_train_run(tmp_path, "simple_s42", model_name="simple", training_seed=42)
    stronger_dir = _write_train_run(
        tmp_path, "stronger_s123", model_name="stronger", training_seed=123
    )
    tracking.record_train_run(conn, str(simple_dir))
    tracking.record_train_run(conn, str(stronger_dir))

    tracking.record_eval_run(
        conn, str(simple_dir), _eval_result(simple_dir, "simple"),
        EvalConfig(noise_seed=1, std=0.10),
    )
    tracking.record_eval_run(
        conn, str(stronger_dir), _eval_result(stronger_dir, "stronger"),
        EvalConfig(noise_seed=2, std=0.20),
    )

    by_model = tracking.query_eval_results(conn, model_name="stronger")
    assert {r["model_name"] for r in by_model} == {"stronger"}

    by_seed = tracking.query_eval_results(conn, training_seed=42)
    assert {r["run_name"] for r in by_seed} == {"simple_s42"}

    by_condition = tracking.query_eval_results(conn, condition="noise")
    assert len(by_condition) == 2
    assert all(r["condition"] == "noise" for r in by_condition)

    by_param = tracking.query_eval_results(conn, condition="noise", std=0.20)
    assert len(by_param) == 1
    assert by_param[0]["model_name"] == "stronger"
