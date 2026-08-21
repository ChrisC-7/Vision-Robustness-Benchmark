"""Evaluate a trained run under clean / blur / noise conditions."""

import argparse
import json
import inspect
from pathlib import Path

from torch import nn
from .engine import evaluate
from .models import load_model
from .data import get_test_dataloader, _TRANSFORMS


def select_kwargs(func, candidates: dict) -> dict:
    valid_names = inspect.signature(func).parameters
    return {k: v for k, v in candidates.items() if k in valid_names and v is not None}


def get_eval_CLI():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["clean", "blur", "noise"],
        default=["clean", "blur", "noise"],
    )
    parser.add_argument("--kernel-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mean", type=float)
    parser.add_argument("--std", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    with open(run_dir / "config.json", "r", encoding="utf-8") as file:
        config_data = json.load(file)

    conditions = args.conditions
    device = args.device or config_data["device"]
    batch_size = config_data["batch_size"] if args.batch_size is None else args.batch_size
    model_name = config_data["model_name"]
    seed = config_data["seed"] if args.seed is None else args.seed
    args.seed = seed

    model = load_model(model_name, load_path=run_dir / "best.pt", device=device)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    con_results = {}
    for con in conditions:
        transform_fn = _TRANSFORMS[con]
        cleaned_args = select_kwargs(transform_fn, vars(args))
        transform = transform_fn(**cleaned_args)
        test_dataloader = get_test_dataloader(transform=transform, batch_size=batch_size)
        loss, acc = evaluate(model, test_dataloader, loss_fn, device)
        con_results[con] = {"loss": loss, "accuracy": acc}
    eval_result = {
        "run_dir": str(run_dir),
        "model": model_name,
        "conditions": con_results,
    }
    with open(run_dir / "eval.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(eval_result) + "\n")


if __name__ == "__main__":
    get_eval_CLI()
