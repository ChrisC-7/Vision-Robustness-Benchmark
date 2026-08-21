"""Training/evaluation algorithm for the Vision Robustness Benchmark.

Disk-free by design: this module never touches the filesystem, so the
training algorithm can be unit-tested on tiny synthetic tensors without a
`runs/` directory. Artifact persistence lives in `run.py`.
"""

import copy
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 0.03
    momentum: float = 0.9
    epochs: int = 5
    device: str = "cpu"


class EpochMetrics(NamedTuple):
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float


class FitResult(NamedTuple):
    epoch_metrics: list[EpochMetrics]
    best_epoch: int
    best_val_accuracy: float
    best_state_dict: dict


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    n = len(dataloader.dataset)
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X)
    return total_loss / n


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    n = len(dataloader.dataset)
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * len(X)
            total_correct += (pred.argmax(dim=1) == y).sum().item()
    return total_loss / n, 100 * total_correct / n


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    on_epoch_end: Optional[Callable[[EpochMetrics], None]] = None,
) -> FitResult:
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.learning_rate, momentum=config.momentum
    )
    loss_fn = nn.CrossEntropyLoss()

    epoch_metrics: list[EpochMetrics] = []
    best_epoch = 0
    best_val_accuracy = -1.0
    best_state_dict: dict = {}

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, config.device)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, config.device)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        metrics = EpochMetrics(epoch, train_loss, val_loss, val_accuracy)
        epoch_metrics.append(metrics)
        if on_epoch_end is not None:
            on_epoch_end(metrics)

    return FitResult(epoch_metrics, best_epoch, best_val_accuracy, best_state_dict)
