# Vision Robustness Benchmark

A reproducible PyTorch benchmark measuring how CNN architecture affects CIFAR-10 classification accuracy under image corruptions (Gaussian blur, Gaussian noise).

## Research Question

Does improving clean-image accuracy also improve robustness to common corruptions?

A baseline **SimpleCNN** and a deeper **StrongerCNN** are trained under an identical protocol and evaluated on the official CIFAR-10 test split under:

- **Clean**
- **Gaussian blur** — kernel size 5, sigma 1.0
- **Gaussian noise** — mean 0, std 0.10, seeded

Checkpoints are selected on validation accuracy only; the test split is held out for final evaluation.

## Results

Each model was trained with 4 training seeds (42, 123, 456, 777) on one fixed 90/10 train/validation split (`split_seed` 42) using SGD (learning rate 0.03, momentum 0.9), batch size 64, and 30 epochs, then evaluated on the official CIFAR-10 test split. Training and evaluation both ran on one NVIDIA GeForce RTX 2060 GPU (`cuda`). Accuracy is in %; Δ is the change from clean accuracy in percentage points (pp), so a negative Δ is accuracy lost. Cells are mean ± sample standard deviation over the 4 seeds.

| Model | Clean | Blur | Noise | Blur Δ | Noise Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| SimpleCNN | 38.01 ± 3.54 | 28.97 ± 1.62 | 29.62 ± 2.52 | -9.04 ± 4.68 | -8.40 ± 3.69 |
| StrongerCNN | 67.14 ± 0.75 | 49.91 ± 1.26 | 59.29 ± 1.52 | -17.23 ± 1.73 | -7.85 ± 1.47 |

Per-seed Δ (pp):

| Seed | SimpleCNN Blur Δ | StrongerCNN Blur Δ | SimpleCNN Noise Δ | StrongerCNN Noise Δ |
| ---: | ---: | ---: | ---: | ---: |
| 42 | -12.57 | -18.41 | -12.10 | -9.13 |
| 123 | -3.69 | -14.99 | -8.73 | -8.72 |
| 456 | -13.34 | -18.76 | -9.46 | -5.84 |
| 777 | -6.57 | -16.76 | -3.30 | -7.71 |

- StrongerCNN is more accurate than SimpleCNN in every condition and every seed (mean clean accuracy 67.14% vs. 38.01%).
- **Blur:** StrongerCNN's accuracy drop is larger in 4 of 4 seeds (mean -17.23 vs. -9.04 pp).
- **Noise:** StrongerCNN's accuracy drop is smaller in 3 of 4 seeds (mean -7.85 vs. -8.40 pp), but this comparison is not resolved: seed 123 is a tie (-8.72 vs. -8.73 pp), seed 777 goes the other way (-7.71 vs. -3.30 pp), and the 0.55 pp gap between the means is far smaller than the seed-to-seed standard deviations (3.69 and 1.47 pp).
- The single-seed observation in earlier versions of this README (a larger blur drop but a smaller noise drop for StrongerCNN) therefore replicated for blur; for noise the direction matches in 3 of 4 seeds, but the gap is within the seed-to-seed spread.

Δ is an absolute change, so it depends on the starting accuracy: StrongerCNN starts about 29 points higher and has more to lose. As mean corrupted accuracy relative to mean clean accuracy, StrongerCNN keeps 74.3% under blur (SimpleCNN: 76.2%) and 88.3% under noise (SimpleCNN: 77.9%).

## Project Structure

```text
.
├── src/
│   ├── models.py     # SimpleCNN / StrongerCNN
│   ├── data.py       # deterministic splits, device-resident train/val loader, clean/blur/noise transforms
│   ├── engine.py     # train/eval loops, TrainConfig / EvalConfig
│   ├── run.py        # writes config.json / metrics.jsonl / best.pt per run
│   ├── train.py      # training CLI
│   ├── eval.py       # evaluation CLI + reusable eval_run()
│   ├── multiseed.py  # run one config across several training seeds
│   ├── tracking.py   # SQLite tracking of runs and eval results
│   ├── inference.py  # load a checkpoint and classify one image
│   └── api.py        # FastAPI server exposing /health and /predict
├── scripts/
│   └── tracking_demo.py  # runnable walkthrough of tracking.py
├── configs/
│   └── multiseed.json    # example multiseed.run_multiseed() config
├── tests/
├── requirements.txt
├── .env.example      # env vars api.py needs at startup
└── README.md
```

## Setup

```bash
git clone https://github.com/ChrisC-7/Vision-Robustness-Benchmark.git
cd Vision-Robustness-Benchmark
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q tests
```

CIFAR-10 downloads automatically on first training or evaluation run.

## Usage

**Train**

```bash
python -m src.train \
  --model stronger \
  --run-name stronger_example \
  --epochs 30 --training-seed 42 --split-seed 42 --batch-size 64 \
  --learning-rate 0.03 --momentum 0.9 --device cpu
```

`--training-seed` controls shuffling/init, `--split-seed` controls the train/validation split, so you can vary one independently of the other.

The train/validation split is converted to tensors once and kept on `--device` for the whole run (about 0.6 GB of GPU memory with `cuda`), instead of being decoded from PIL images every epoch. Training uses no data augmentation, so the batches are the same as a standard `DataLoader` would produce. Evaluation still goes through a regular `DataLoader`, because the blur and noise corruptions are applied per image.

`--model` accepts `simple` or `stronger`. Each run writes `config.json`, `metrics.jsonl`, and `best.pt` to `runs/<run-name>/`. The checkpoint is replaced only on a strict validation-accuracy improvement, so ties keep the earlier epoch.

**Evaluate**

```bash
python -m src.eval \
  --run-dir runs/stronger_example \
  --conditions clean blur noise
```

Loads `best.pt`, evaluates on the official test split, and appends results to `eval.jsonl`. `--conditions` accepts any subset of `clean`, `blur`, `noise`.

## Additional Tooling

**Multi-seed experiments** — `multiseed.run_multiseed()` trains and evaluates one config across every model/seed combination in a JSON file (see `configs/multiseed.json`):

```bash
python -m src.multiseed --config configs/multiseed.json
```

or, from Python:

```python
from src.multiseed import load_multiseed_config, run_multiseed

run_multiseed(load_multiseed_config("configs/multiseed.json"))
```

For each `models` × `training_seeds` combination (8 with the provided file) it trains a run named `<model>_s<seed>` and evaluates it right away on every condition in the `eval` block, one run after another. Only the model and the training seed change between runs: the `train` block (split seed, epochs, batch size, learning rate, momentum, device) and the `eval` block (conditions, noise seed, noise std, device) are shared by all of them. Both blocks accept a `device` field that defaults to `cpu`.

Outputs go to `<root>/<experiment_name>/`:

```text
runs/multiseed/experiment1/
├── config.json       # the experiment config that was run
├── result.jsonl      # one line per run: loss and accuracy per condition
├── simple_s42/       # config.json, metrics.jsonl, best.pt, eval.jsonl
├── ...
└── stronger_s777/
```

`result.jsonl` is written once, after the last run finishes. A run directory that already exists is never overwritten (`FileExistsError`), so rerunning an experiment that already has runs, including after an interrupted run, fails: pick a new `experiment_name` or delete the directory first.

**Experiment tracking** — `tracking.py` records each run's config, checkpoint path, and best epoch, plus each evaluation's resolved per-condition parameters (e.g. actual `std`, `kernel_size`, `sigma`), into a local SQLite database. Writes are retry-safe: re-recording the same run or the same evaluation attempt updates that row instead of duplicating it, while re-running a config as a fresh attempt is stored as a new, independent row. See `scripts/tracking_demo.py` for a runnable walkthrough (train a couple of runs first, per the instructions in that script).

**Serving a trained checkpoint** — `api.py` exposes a FastAPI server for the checkpoints produced by `train.py`:

```bash
cp .env.example .env   # point VISION_SIMPLE_RUN_DIR / VISION_STRONGER_RUN_DIR at trained runs
uvicorn src.api:app --reload
```

`GET /health` lists the loaded models; `POST /predict` (multipart form: `file`, `model`) returns the predicted class and confidence. `inference.py` holds the underlying image preprocessing and prediction logic if you want to call a model directly instead of through the API.

## Tests

```bash
python -m pytest -q tests
```

Covers model output contracts, seeded noise reproducibility, output clamping to `[0, 1]`, validation-tie checkpoint selection, run-artifact creation, multi-seed orchestration, and the SQLite tracking layer (idempotent writes, filtering by model/seed/condition/param).

## Limitations & Next Step

Results cover 4 training seeds on a single train/validation split, one severity per corruption, and two small CNNs trained without data augmentation. With 4 seeds the per-seed comparisons are descriptive rather than statistical tests, and SimpleCNN's results vary a lot from seed to seed (clean accuracy 34.85–41.09%), so nothing here generalizes to other architectures or datasets.

The runs were trained on GPU, and `src/run.py` seeds PyTorch's RNG but does not enable deterministic cuDNN settings, so a rerun with the same seeds gives different numbers. A quick repeat (SimpleCNN, seed 123, identical code) already differed in training loss and validation accuracy from the first epoch on. A full earlier run of this same experiment, trained with the previous `DataLoader`-based pipeline, shows how large this is: that pipeline yields the same batches in the same order as the current device-resident loader (checked batch by batch for training seeds 42 and 123 over two epochs), yet its cells differ. StrongerCNN's accuracies moved by up to about 2.6 pp. SimpleCNN's moved by up to about 9 pp (seed-42 noise accuracy 37.81% there vs. 28.96% here; seed-123 noise accuracy 20.36% vs. 26.32%), and in that earlier run StrongerCNN's noise drop was larger in 3 of 4 seeds instead of 1 of 4. Differences smaller than this run-to-run spread, the noise comparison in particular, should not be read as findings. The single-seed table in even earlier versions of this README came from a separate CPU run; its seed-42 numbers differ from the seed-42 row here (for example SimpleCNN clean accuracy 39.71% vs. 41.06%), so the two are not comparable.

The next experiment is a severity sweep (several noise `std` and blur `kernel_size` values) to see whether the blur gap widens with severity and whether the noise comparison, unresolved here, separates with more seeds or deterministic cuDNN settings.
