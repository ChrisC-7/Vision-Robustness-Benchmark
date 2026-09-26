"""Run-artifact contract: runs/<run-name>/{config.json, metrics.jsonl, best.pt}.

The only module in this package that touches the filesystem.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .data import get_train_val_dataloaders
from .engine import EpochMetrics, TrainConfig, fit
from .models import build_model

CHECKPOINT_SELECTION_RULE = (
    "best validation accuracy; checkpoint replaced only on strict "
    "improvement, so ties keep the earlier epoch"
)


def _write_config(path: Path, config: TrainConfig) -> None:
    payload = asdict(config)
    payload["optimizer"] = "SGD"
    payload["checkpoint_selection"] = CHECKPOINT_SELECTION_RULE
    path.write_text(json.dumps(payload, indent=2))


def _append_metrics(path: Path, metrics: EpochMetrics) -> None:
    record = metrics._asdict()
    record["timestamp"] = time.time()
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def train_run(run_name: str, config: TrainConfig, root: str = "runs") -> Path:
    out_dir = Path(root) / run_name
    config_path = out_dir / "config.json"
    if config_path.exists():
        raise FileExistsError(f"Run already exists: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_config(config_path, config)

    torch.manual_seed(config.training_seed)
    model = build_model(config.model_name).to(config.device)
    train_loader, val_loader = get_train_val_dataloaders(
        training_seed=config.training_seed, split_seed=config.split_seed, batch_size=config.batch_size
    )

    metrics_path = out_dir / "metrics.jsonl"
    result = fit(
        model,
        train_loader,
        val_loader,
        config,
        on_epoch_end=lambda m: _append_metrics(metrics_path, m),
    )

    torch.save(result.best_state_dict, out_dir / "best.pt")
    return out_dir
