"""SQLite-backed tracking for training runs and evaluation results.

This module never trains or evaluates anything itself. It only reads the
artifacts that :mod:`run` and :mod:`eval`
already write to disk (``config.json``, ``metrics.jsonl``, ``best.pt``) and
the return value of :func:`eval.eval_run`, and records
them into a SQLite database so past experiments can be queried, compared,
and traced back to the run that produced them.
"""

import inspect
import json
import sqlite3
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .data import _TRANSFORMS
from .engine import EvalConfig
from .eval import select_kwargs

_SCHEMA = """
CREATE TABLE IF NOT EXISTS train_runs (
    run_dir TEXT PRIMARY KEY,
    run_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    training_seed INTEGER,
    split_seed INTEGER,
    train_config_json TEXT NOT NULL,
    best_epoch INTEGER,
    best_val_accuracy REAL,
    checkpoint_path TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_runs (
    eval_id TEXT PRIMARY KEY,
    run_dir TEXT NOT NULL REFERENCES train_runs(run_dir),
    model_name TEXT NOT NULL,
    noise_seed INTEGER,
    eval_config_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_results (
    eval_id TEXT NOT NULL REFERENCES eval_runs(eval_id),
    condition TEXT NOT NULL,
    loss REAL NOT NULL,
    accuracy REAL NOT NULL,
    params_json TEXT NOT NULL,
    PRIMARY KEY (eval_id, condition)
);
"""


def get_connection(db_path: str = "experiments.db") -> sqlite3.Connection:
    """Opens a tracking database, creating its schema if needed.

    Args:
        db_path: Filesystem path to the SQLite file. Pass ``":memory:"``
            for an ephemeral database; the test suite does this so tests
            never touch a file on disk.

    Returns:
        A connection with row access by column name (``sqlite3.Row``) and
        foreign key enforcement turned on.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    return conn


def _now() -> str:
    """Returns the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _select_best_epoch(
    metrics_path: Path,
) -> tuple[Optional[int], Optional[float]]:
    """Re-derives the best epoch from a run's ``metrics.jsonl``.

    Mirrors the selection rule already documented as
    ``run.CHECKPOINT_SELECTION_RULE``: the checkpoint is replaced only on
    strict improvement, so scanning in epoch order and keeping the first
    strict maximum reproduces exactly which epoch ``best.pt`` came from.

    Args:
        metrics_path: Path to a run directory's ``metrics.jsonl``.

    Returns:
        A ``(best_epoch, best_val_accuracy)`` tuple, or ``(None, None)`` if
        the file is missing or empty.
    """
    if not metrics_path.exists():
        return None, None

    best_epoch, best_val_accuracy = None, None
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if best_val_accuracy is None or record["val_accuracy"] > best_val_accuracy:
            best_epoch = record["epoch"]
            best_val_accuracy = record["val_accuracy"]
    return best_epoch, best_val_accuracy


def record_train_run(conn: sqlite3.Connection, run_dir: str) -> str:
    """Records, or re-records, one training run's artifacts.

    Reads ``run_dir/config.json`` and ``run_dir/metrics.jsonl`` (both
    already written by :func:`run.train_run`) plus the
    presence of ``run_dir/best.pt``. Fields that cannot be determined --
    missing files, or a field absent from an older ``TrainConfig`` shape --
    are stored as ``NULL`` rather than guessed.

    ``run_dir`` is this row's identity: :func:`run.train_run`
    already refuses to reuse a directory that exists (``FileExistsError``),
    so calling this function again for the *same* ``run_dir`` is always a
    retry of the same run, and updates the row in place instead of adding a
    duplicate. Training the same config again under a new ``run_name``
    produces a new ``run_dir`` and therefore a new, independent row.

    Args:
        conn: An open connection from :func:`get_connection`.
        run_dir: Path to a run directory produced by
            :func:`run.train_run`.

    Returns:
        The normalized ``run_dir`` string used as the primary key.

    Raises:
        FileNotFoundError: If ``run_dir/config.json`` does not exist.
    """
    run_dir_path = Path(run_dir)
    config_path = run_dir_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json under {run_dir}")

    config_data = json.loads(config_path.read_text())
    checkpoint_path = run_dir_path / "best.pt"
    best_epoch, best_val_accuracy = _select_best_epoch(run_dir_path / "metrics.jsonl")
    now = _now()

    conn.execute(
        """
        INSERT INTO train_runs (
            run_dir, run_name, model_name, training_seed, split_seed,
            train_config_json, best_epoch, best_val_accuracy,
            checkpoint_path, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_dir) DO UPDATE SET
            run_name=excluded.run_name,
            model_name=excluded.model_name,
            training_seed=excluded.training_seed,
            split_seed=excluded.split_seed,
            train_config_json=excluded.train_config_json,
            best_epoch=excluded.best_epoch,
            best_val_accuracy=excluded.best_val_accuracy,
            checkpoint_path=excluded.checkpoint_path,
            updated_at=excluded.updated_at
        """,
        (
            str(run_dir_path),
            run_dir_path.name,
            config_data["model_name"],
            config_data.get("training_seed"),
            config_data.get("split_seed"),
            json.dumps(config_data),
            best_epoch,
            best_val_accuracy,
            str(checkpoint_path) if checkpoint_path.exists() else None,
            now,
            now,
        ),
    )
    conn.commit()
    return str(run_dir_path)


def _resolve_condition_params(condition: str, config: EvalConfig) -> dict[str, Any]:
    """Fully resolves the kwargs a corruption transform actually ran with.

    ``EvalConfig`` does not carry every parameter a transform accepts -- for
    example ``gaussian_blur_transform``'s ``sigma`` has no ``EvalConfig``
    field and always falls back to the function's own default. Reusing
    :func:`eval.select_kwargs` (the exact helper
    ``eval_run`` uses to build the transform) and then applying the
    transform's own signature defaults reconstructs the complete set of
    parameters that were actually in effect, not just the ones
    ``EvalConfig`` happened to override.

    Args:
        condition: One of ``"clean"``, ``"blur"``, ``"noise"``.
        config: The ``EvalConfig`` an evaluation ran with.

    Returns:
        A JSON-serializable dict of every parameter name to its actual
        value, including defaults ``EvalConfig`` never touched.
    """
    transform_fn = _TRANSFORMS[condition]
    cleaned_args = select_kwargs(transform_fn, vars(config))
    bound = inspect.signature(transform_fn).bind_partial(**cleaned_args)
    bound.apply_defaults()
    return dict(bound.arguments)


def record_eval_run(
    conn: sqlite3.Connection,
    run_dir: str,
    eval_result: dict,
    config: EvalConfig,
    eval_id: Optional[str] = None,
) -> str:
    """Records one evaluation attempt and its per-condition results.

    Args:
        conn: An open connection from :func:`get_connection`.
        run_dir: The training run this evaluation ran against. Must already
            be recorded via :func:`record_train_run`.
        eval_result: The dict returned by
            :func:`eval.eval_run`.
        config: The ``EvalConfig`` that produced ``eval_result``.
        eval_id: Idempotency key for this evaluation *attempt*. Omit it to
            record a fresh, independent attempt (a new row is created).
            Pass back the ``eval_id`` a previous call returned to retry
            recording that exact attempt: the row is updated in place
            instead of duplicated. Content alone can't tell "the same
            write retried" apart from "the same config run again" -- both
            produce identical rows -- so that identity has to come from the
            caller, not from the data.

    Returns:
        The ``eval_id`` used, so the caller can retry with it later.
    """
    eval_id = eval_id or str(uuid.uuid4())
    run_dir = str(Path(run_dir))
    now = _now()

    conn.execute(
        """
        INSERT INTO eval_runs (
            eval_id, run_dir, model_name, noise_seed, eval_config_json,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(eval_id) DO UPDATE SET
            run_dir=excluded.run_dir,
            model_name=excluded.model_name,
            noise_seed=excluded.noise_seed,
            eval_config_json=excluded.eval_config_json,
            updated_at=excluded.updated_at
        """,
        (
            eval_id,
            run_dir,
            eval_result["model"],
            config.noise_seed,
            json.dumps(asdict(config)),
            now,
            now,
        ),
    )

    for condition, outcome in eval_result["conditions"].items():
        params_json = json.dumps(_resolve_condition_params(condition, config))
        conn.execute(
            """
            INSERT INTO eval_results (eval_id, condition, loss, accuracy, params_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(eval_id, condition) DO UPDATE SET
                loss=excluded.loss,
                accuracy=excluded.accuracy,
                params_json=excluded.params_json
            """,
            (eval_id, condition, outcome["loss"], outcome["accuracy"], params_json),
        )

    conn.commit()
    return eval_id


def query_eval_results(
    conn: sqlite3.Connection,
    *,
    model_name: Optional[str] = None,
    training_seed: Optional[int] = None,
    noise_seed: Optional[int] = None,
    condition: Optional[str] = None,
    **param_filters: Any,
) -> list[dict]:
    """Queries recorded evaluation results across all three tables.

    Every filter is optional and combined with AND. ``param_filters`` are
    matched against each result's resolved condition parameters (see
    :func:`_resolve_condition_params`) via SQLite's ``json_extract``, e.g.
    ``query_eval_results(conn, condition="noise", std=0.10)``.

    Args:
        conn: An open connection from :func:`get_connection`.
        model_name: Restrict to this model.
        training_seed: Restrict to this training seed.
        noise_seed: Restrict to this evaluation noise seed.
        condition: Restrict to this corruption condition
            (``"clean"``/``"blur"``/``"noise"``).
        **param_filters: Restrict to rows whose resolved condition
            parameters have this value for the given key, e.g.
            ``kernel_size=5``.

    Returns:
        One dict per matching (run, eval attempt, condition) row.
    """
    sql = """
        SELECT tr.run_dir, tr.run_name, tr.model_name, tr.training_seed,
               er.eval_id, er.noise_seed, ec.condition, ec.loss, ec.accuracy,
               ec.params_json
        FROM eval_results ec
        JOIN eval_runs er ON ec.eval_id = er.eval_id
        JOIN train_runs tr ON er.run_dir = tr.run_dir
        WHERE 1 = 1
    """
    params: list[Any] = []

    filters = {
        "tr.model_name": model_name,
        "tr.training_seed": training_seed,
        "er.noise_seed": noise_seed,
        "ec.condition": condition,
    }
    for column, value in filters.items():
        if value is not None:
            sql += f" AND {column} = ?"
            params.append(value)

    for key, value in param_filters.items():
        sql += " AND json_extract(ec.params_json, ?) = ?"
        params.extend([f"$.{key}", value])

    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]
