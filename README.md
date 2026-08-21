# Vision Robustness Benchmark

A reproducible PyTorch benchmark for studying how image-classification models respond to controlled input corruptions on CIFAR-10.

The project combines two goals:

1. **ML experimentation** — measure how architecture and training choices affect clean accuracy and corruption robustness under controlled conditions.
2. **Reproducible ML engineering** — provide stable training/evaluation interfaces, deterministic data pipelines, structured run artifacts, and focused automated tests.

The current completed comparison evaluates a SimpleCNN and a StrongerCNN under clean images, Gaussian blur, and additive Gaussian noise.

---

## Research Question

Does substantially improving clean-image classification accuracy also improve robustness to common input corruptions?

To study this question, the project compares two CNN architectures while keeping the evaluation protocol fixed.

Each selected checkpoint is evaluated on the official CIFAR-10 test split under:

- **Clean:** unmodified test images
- **Gaussian blur:** kernel size 5, sigma 1.0
- **Gaussian noise:** mean 0, standard deviation 0.10, with seeded corruption randomness

Checkpoint selection uses validation accuracy only. The official CIFAR-10 test split is reserved for final evaluation.

---

## Current Results

| Model | Clean Accuracy | Blur Accuracy | Noise Accuracy | Blur Δ vs. Clean | Noise Δ vs. Clean |
| --- | ---: | ---: | ---: | ---: | ---: |
| SimpleCNN | 39.71% | 30.02% | 29.23% | -9.69 pp | -10.48 pp |
| StrongerCNN | 67.44% | 53.55% | 59.14% | -13.89 pp | -8.30 pp |

The StrongerCNN substantially improved **absolute accuracy** under all three evaluation conditions.

However, higher clean accuracy did **not** uniformly reduce relative corruption degradation in this configuration:

- blur degradation increased from **9.69 pp** to **13.89 pp**;
- noise degradation decreased from **10.48 pp** to **8.30 pp**.

For the StrongerCNN run, Gaussian blur therefore caused the larger accuracy drop under the selected corruption settings.

These results are configuration-specific. They do **not** establish that stronger CNNs are generally more sensitive to blur or more robust to Gaussian noise.

---

## Experimental Protocol

The current experiments use:

- Dataset: CIFAR-10
- Optimizer: SGD
- Learning rate: 0.03
- Momentum: 0.9
- Batch size: 64
- Training seed: 42
- Training budget: 30 epochs
- Checkpoint selection: highest validation accuracy only

The train/validation split is deterministic for a fixed seed.

Across final clean / blur / noise evaluations, the following are kept fixed:

- model checkpoint
- CIFAR-10 test split
- labels
- batch size
- evaluation logic
- loss definition
- accuracy definition

Only the input corruption condition changes.

Gaussian noise is generated using a dedicated seeded `torch.Generator` and is clamped back to the `[0, 1]` image range.

Checkpoint replacement occurs only on a **strict validation-accuracy improvement**, so validation ties keep the earlier epoch.

---

## Models

### SimpleCNN

The baseline architecture contains one convolutional layer followed by ReLU, max pooling, flattening, and a linear classifier.

### StrongerCNN

The stronger architecture contains two convolutional blocks followed by a hidden fully connected layer and final classifier.

The architecture comparison intentionally avoids simultaneously introducing augmentation, dropout, BatchNorm, or other training changes.

---

## Stable Implementation

Reusable project code lives under:

```text
src/
├── __init__.py
├── models.py
├── data.py
├── engine.py
├── run.py
├── train.py
└── eval.py
```

---

## Setup

```bash
pip install -r requirements.txt
pytest -q
```
