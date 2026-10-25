"""Shared fixtures."""

import json
from pathlib import Path

import pytest

EXPERIMENT_DIR = Path("runs/multiseed/exp")

# (model_name, training_seed, clean accuracy) for each fake run.
FAKE_RUNS = [
    ("simple", 42, 41.0),
    ("simple", 123, 35.0),
    ("stronger", 42, 70.0),
]


@pytest.fixture
def experiment_dir(tmp_path, monkeypatch):
    """Writes a fake experiment shaped like multiseed.run_multiseed's output.

    Changes into ``tmp_path`` first, so the relative ``run_dir`` strings in
    ``result.jsonl`` resolve exactly as they do from the repository root.

    Returns:
        The experiment directory, relative to the new working directory.
    """
    monkeypatch.chdir(tmp_path)

    results = []
    for model_name, training_seed, clean_accuracy in FAKE_RUNS:
        run_dir = EXPERIMENT_DIR / f"{model_name}_s{training_seed}"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "training_seed": training_seed,
                    "split_seed": 42,
                    "epochs": 2,
                    "device": "cpu",
                }
            )
        )
        (run_dir / "metrics.jsonl").write_text(
            "\n".join(
                json.dumps({"epoch": epoch, "val_accuracy": accuracy})
                for epoch, accuracy in [(1, 30.0), (2, clean_accuracy)]
            )
        )
        (run_dir / "best.pt").write_bytes(b"fake-checkpoint")
        results.append(
            {
                "run_dir": str(run_dir),
                "model": model_name,
                "conditions": {
                    "clean": {"loss": 1.0, "accuracy": clean_accuracy},
                    "blur": {"loss": 1.5, "accuracy": clean_accuracy - 10},
                    "noise": {"loss": 2.0, "accuracy": clean_accuracy - 15},
                },
            }
        )

    (EXPERIMENT_DIR / "config.json").write_text(
        json.dumps(
            {
                "eval_config": {
                    "conditions": ["clean", "blur", "noise"],
                    "kernel_size": 5,
                    "noise_seed": 42,
                    "mean": 0.0,
                    "std": 0.1,
                    "batch_size": 64,
                    "device": "cpu",
                }
            }
        )
    )
    (EXPERIMENT_DIR / "result.jsonl").write_text(
        "".join(json.dumps(result) + "\n" for result in results)
    )
    return EXPERIMENT_DIR
