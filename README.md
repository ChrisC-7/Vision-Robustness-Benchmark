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

| Model | Clean | Blur | Noise | Blur Δ | Noise Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| SimpleCNN | 39.71% | 30.02% | 29.23% | -9.69 pp | -10.48 pp |
| StrongerCNN | 67.44% | 53.55% | 59.14% | -13.89 pp | -8.30 pp |

StrongerCNN wins on absolute accuracy everywhere, but the *relative* degradation pattern is mixed — blur degradation gets worse, noise degradation gets better. These results reflect one seed and one severity per corruption, so they don't generalize on their own (see [Limitations](#limitations--next-step)).

## Project Structure

```text
.
├── src/
│   ├── models.py     # SimpleCNN / StrongerCNN
│   ├── data.py       # deterministic splits, clean/blur/noise transforms
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

```python
from src.multiseed import load_multiseed_config, run_multiseed

run_multiseed(load_multiseed_config("configs/multiseed.json"))
```

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

Results reflect a single training seed and a single severity per corruption — they're controlled observations, not general claims about CNN robustness. The next experiment repeats the StrongerCNN run under a second seed (architecture, optimizer, budget, and corruption severities held fixed) to test whether the blur/noise pattern persists across runs.
