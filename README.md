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
│   ├── models.py   # SimpleCNN / StrongerCNN
│   ├── data.py     # deterministic splits, clean/blur/noise transforms
│   ├── engine.py   # train/eval loops
│   ├── train.py    # training CLI
│   └── eval.py     # evaluation CLI
├── tests/
├── requirements.txt
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
  --epochs 30 --seed 42 --batch-size 64 \
  --learning-rate 0.03 --momentum 0.9 --device cpu
```

`--model` accepts `simple` or `stronger`. Each run writes `config.json`, `metrics.jsonl`, and `best.pt` to `runs/<run-name>/`. The checkpoint is replaced only on a strict validation-accuracy improvement, so ties keep the earlier epoch.

**Evaluate**

```bash
python -m src.eval \
  --run-dir runs/stronger_example \
  --conditions clean blur noise
```

Loads `best.pt`, evaluates on the official test split, and appends results to `eval.jsonl`. `--conditions` accepts any subset of `clean`, `blur`, `noise`.

## Tests

```bash
python -m pytest -q tests
```

Covers model output contracts, seeded noise reproducibility, output clamping to `[0, 1]`, validation-tie checkpoint selection, and run-artifact creation.

## Limitations & Next Step

Results reflect a single training seed and a single severity per corruption — they're controlled observations, not general claims about CNN robustness. The next experiment repeats the StrongerCNN run under a second seed (architecture, optimizer, budget, and corruption severities held fixed) to test whether the blur/noise pattern persists across runs.
