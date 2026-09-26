import json

import torch

from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from src import engine, run
from src.data import GaussianNoise
from src import multiseed


def test_gaussian_noise_reproducible_and_clamped():
    X = torch.randn(5, 5)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    n1 = GaussianNoise(0, 0.1, g1)
    n2 = GaussianNoise(0, 0.1, g2)

    y_1 = n1(X)
    y_2 = n2(X)

    assert torch.equal(y_1, y_2)
    assert torch.Tensor.max(y_1) <= 1
    assert torch.Tensor.min(y_1) >=0

def test_fit_keeps_earlier_epoch_on_val_accuracy_tie(monkeypatch):

    acc = iter([50, 90, 70, 90])

    def fake_train_one_epoch(model, dataloader, optimizer, loss_fn, device):
        return 0

    def fake_eval(model, dataloader, loss_fn, device):
        return 0, next(acc)

    monkeypatch.setattr(engine, 'train_one_epoch', fake_train_one_epoch)
    monkeypatch.setattr(engine, 'evaluate', fake_eval)

    config = engine.TrainConfig(model_name='simple', epochs=4)
    model = torch.nn.Linear(4, 1)

    result = engine.fit(model, None, None, config)
    assert result.best_epoch == 2
    assert result.best_val_accuracy == 90



def test_train_run_writes_config_metrics_and_checkpoint(tmp_path, monkeypatch):
    def fake_build_model(name):
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4,1))

    def fake_get_train_val_dataloaders(training_seed, split_seed, batch_size):
        t_data = torch.rand(4, 4)
        v_data = torch.rand(2, 4)
        y_t = torch.rand(4, 1)
        y_v = torch.rand(2, 1)

        t_loader = DataLoader(TensorDataset(t_data, y_t))
        v_loader = DataLoader(TensorDataset(v_data, y_v))
        return t_loader, v_loader

    monkeypatch.setattr(run, 'build_model', fake_build_model)
    monkeypatch.setattr(run, 'get_train_val_dataloaders', fake_get_train_val_dataloaders)

    config = run.TrainConfig(model_name='simple', epochs=1, batch_size=2)
    out_dir = run.train_run('smoke', config, root=tmp_path)
    assert (out_dir/"config.json").exists()
    assert (out_dir / "metrics.jsonl").exists()
    assert (out_dir/"best.pt").exists()

    config_data = json.loads((out_dir / "config.json").read_text())
    assert config_data["model_name"] == "simple"

    metrics_lines = (out_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(metrics_lines) == 1

def test_multiseed_run(tmp_path, monkeypatch):
    from dataclasses import replace

    train_calls = []
    eval_calls = []
    def fake_train(run_name, train_config, root):
        train_calls.append((run_name, train_config))
        out_dir = Path(root)/run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def fake_eval(run_dir, eval_config):
        eval_calls.append((run_dir, eval_config))
        return {"run_dir": str(run_dir)}

    monkeypatch.setattr(multiseed, 'train_run', fake_train)
    monkeypatch.setattr(multiseed, 'eval_run', fake_eval)
    config = multiseed.MultiSeedConfig(
        models=("simple", "stronger"),
        training_seeds=(42, 123, 456, 777),
        root=str(tmp_path),
        experiment_name="experiment1",
    )
    multiseed.run_multiseed(config)

    experiment_dir = Path(tmp_path) / "experiment1"
    run_dirs = [p for p in experiment_dir.iterdir() if p.is_dir()]
    assert len(run_dirs) == 8

    result_lines = (experiment_dir / "result.jsonl").read_text().strip().splitlines()
    assert len(result_lines) == 8

    assert len(train_calls) == 8
    assert len(eval_calls) == 8


    combos = [(cfg.model_name, cfg.training_seed) for _, cfg in train_calls]
    assert set(combos) == {(m, s) for m in ("simple", "stronger") for s in (42, 123, 456, 777)}
    assert len(set(combos)) == len(combos)

    run_names = [name for name, _ in train_calls]
    assert set(run_names) == {f"{m}_s{s}" for m, s in combos}
    assert len(set(run_names)) == len(run_names)

    normalized_train_configs = {
        replace(cfg, model_name="_", training_seed=0) for _, cfg in train_calls
    }
    assert len(normalized_train_configs) == 1

    eval_configs = {cfg for _, cfg in eval_calls}
    assert len(eval_configs) == 1

    train_out_dirs = [experiment_dir / name for name in run_names]
    eval_run_dirs = [run_dir for run_dir, _ in eval_calls]
    assert eval_run_dirs == train_out_dirs

    _, cfg = train_calls[0]

    assert cfg.split_seed == 42
    assert cfg.epochs == 30
    assert cfg.batch_size == 64
    assert cfg.learning_rate == 0.03
    assert cfg.momentum == 0.9
    assert cfg.device == "cpu"

    _, eval_cfg = eval_calls[0]
    assert eval_cfg.conditions == ("clean", "blur", "noise")
    assert eval_cfg.kernel_size == 5
    assert eval_cfg.noise_seed == 42
    assert eval_cfg.mean == 0.0
    assert eval_cfg.std == 0.10
    assert eval_cfg.batch_size == 64
    assert eval_cfg.device == "cpu"
