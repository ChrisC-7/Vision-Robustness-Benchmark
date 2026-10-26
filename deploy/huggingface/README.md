---
title: Robustness Explorer API
emoji: 🧪
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: FastAPI backend of a CIFAR-10 robustness benchmark
---

# Robustness Explorer API

The FastAPI backend of [Vision Robustness Benchmark](https://github.com/ChrisC-7/Vision-Robustness-Benchmark): two small CIFAR-10 CNNs and their results under Gaussian blur and noise.

| Endpoint | Returns |
| --- | --- |
| `GET /health` | the loaded models |
| `POST /predict`, `POST /predict/compare` | one model's, or both models', prediction for an uploaded image |
| `GET /runs`, `GET /runs/{run_dir}` | the recorded training runs |
| `GET /results` | per-condition evaluation results |

Interactive docs are at `/docs`.

This Space is deployed by the GitHub repository's CI on every push to `main`, so change the code there, not here.
