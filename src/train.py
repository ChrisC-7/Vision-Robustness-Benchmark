import argparse
from dataclasses import fields
from .engine import TrainConfig
from .run import train_run

def get_train_CLI():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='simple')
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type = int)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--batch-size", type = int)
    parser.add_argument("--learning-rate", type = float)
    parser.add_argument("--momentum", type = float)
    parser.add_argument("--runs-dir")
    parser.add_argument("--device")
    args = parser.parse_args()

    cleaned_config = {
        "model_name": args.model,
    }

    for f in fields(TrainConfig):
        if f.name == "model_name":
            continue
        value = getattr(args, f.name, None)
        if value is not None:
            cleaned_config[f.name] = value
    config = TrainConfig(**cleaned_config)

    train_run(args.run_name, config, root=args.runs_dir or "runs")

if __name__ == "__main__":
    get_train_CLI()

