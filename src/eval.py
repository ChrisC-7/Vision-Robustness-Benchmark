"""Evaluate a trained run under clean / blur / noise conditions."""

import argparse
import json
import inspect
from pathlib import Path

from torch import nn
from .engine import evaluate, EvalConfig
from .models import load_model
from .data import get_test_dataloader, _TRANSFORMS



def select_kwargs(func, candidates: dict) -> dict:
    valid_names = inspect.signature(func).parameters
    return {k: v for k, v in candidates.items() if k in valid_names and v is not None}

def eval_run(run_dir: str, config: EvalConfig):
    run_dir = Path(run_dir)
    with open(run_dir / "config.json", "r", encoding="utf-8") as file:
            config_data = json.load(file)
    model_name = config_data["model_name"]
    model = load_model(model_name, load_path=run_dir / "best.pt", device=config.device)
    model = model.to(config.device)
    loss_fn = nn.CrossEntropyLoss()

    con_results = {}
    for con in config.conditions:
        transform_fn = _TRANSFORMS[con]
        cleaned_args = select_kwargs(transform_fn, vars(config))
        transform = transform_fn(**cleaned_args)
        test_dataloader = get_test_dataloader(transform=transform, batch_size=config.batch_size)
        loss, acc = evaluate(model, test_dataloader, loss_fn, config.device)
        con_results[con] = {"loss": loss, "accuracy": acc}
    eval_result = {
        "run_dir": str(run_dir),
        "model": model_name,
        "conditions": con_results,
    }
    with open(run_dir / "eval.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(eval_result) + "\n")
    return eval_result
    

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
    parser.add_argument("--noise-seed", type=int)
    parser.add_argument("--mean", type=float)
    parser.add_argument("--std", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    with open(run_dir / "config.json", "r", encoding="utf-8") as file:
        config_data = json.load(file)


    args.device = args.device or config_data["device"]
    args.batch_size = args.batch_size or config_data["batch_size"]

    config = EvalConfig(**select_kwargs(EvalConfig, vars(args)))

    result = eval_run(args.run_dir, config)


if __name__ == "__main__":
    get_eval_CLI()
