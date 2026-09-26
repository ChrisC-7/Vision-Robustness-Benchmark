from pathlib import Path
import json
from dataclasses import dataclass, field, replace, asdict

from .run import train_run
from .eval import eval_run
from .engine import TrainConfig, EvalConfig

@dataclass(frozen=True)
class MultiSeedConfig:
    models: tuple[str, ...]
    training_seeds: tuple[int, ...]
    root: str = 'runs/multiseed/'
    experiment_name: str = 'experiment1'

    base_train_config: TrainConfig = TrainConfig(
        model_name="simple",
        epochs=30,
    )
    eval_config: EvalConfig = field(default_factory=EvalConfig)

def load_multiseed_config(path = 'configs/multiseed.json'):
    with open(Path(path), "r", encoding="utf-8") as file:
        config_data = json.load(file)
    models = tuple(config_data['models'])
    seeds = tuple(config_data['training_seeds'])

    train_config = TrainConfig(model_name=config_data["models"][0],
                            **config_data['train'])

    eval_data = config_data["eval"].copy()
    if "conditions" in eval_data:
        eval_data["conditions"] = tuple(eval_data["conditions"])
    eval_config = EvalConfig(**eval_data)
    experiment_name=config_data.get(
        "experiment_name",
        "experiment1",
    )
    config = MultiSeedConfig(models= models, 
                            training_seeds=seeds,
                            root = config_data['root'], 
                            experiment_name= experiment_name,
                            base_train_config=train_config, 
                            eval_config=eval_config)
    return config

def run_multiseed(config: MultiSeedConfig):
    experiment_dir = Path(config.root) / config.experiment_name
    config_path = experiment_dir / "config.json"
    if config_path.exists():
        raise FileExistsError('Already exists such config')
    experiment_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(config_path), 'w', encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=4)

    models = config.models
    training_seeds = config.training_seeds
    eval_config = config.eval_config
    results=[]
    amount_expected = len(models) * len(training_seeds)
    amount_current = 0

    for model in models:
        for seed in training_seeds:
            train_config = replace(
                config.base_train_config,
                model_name = model,
                training_seed = seed
            )
            run_name = f'{model}_s{seed}'
            run_dir = train_run(run_name, train_config, Path(experiment_dir))
            eval_result = eval_run(run_dir, eval_config)
            results.append(eval_result)
            amount_current += 1
            print(f'[{amount_current}/{amount_expected}] Done {model} seed: {seed}')
    
    with open(Path(experiment_dir)/'result.jsonl', 'a') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    return results


