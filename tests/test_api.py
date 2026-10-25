"""Tests for src.api.

The client is built without ``with TestClient(...)``, so the app's lifespan
never runs: no env vars or real checkpoints are needed. Each test instead
points ``VISION_DB_PATH`` at a temporary database filled by
:func:`backfill.backfill_experiment`, and installs two tiny stand-in models.
"""

import asyncio
import io

import httpx2
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch import nn

from src import api, tracking
from src.backfill import backfill_experiment


class _FixedPrediction(nn.Module):
    """Predicts the same class for every image."""

    def __init__(self, class_index: int) -> None:
        super().__init__()
        self.class_index = class_index
        # inference.predict reads the model's device from its parameters.
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(x.shape[0], 10)
        logits[:, self.class_index] = 5.0
        return logits + self.bias


@pytest.fixture
def client(experiment_dir, monkeypatch):
    conn = tracking.get_connection("experiments.db")
    backfill_experiment(conn, str(experiment_dir))
    conn.close()
    monkeypatch.setenv("VISION_DB_PATH", "experiments.db")

    monkeypatch.setattr(
        api.app.state,
        "models",
        {"simple": _FixedPrediction(3), "stronger": _FixedPrediction(5)},
        raising=False,
    )
    return TestClient(api.app)


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (48, 48), color=(200, 30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_list_runs_is_sorted_and_hides_server_paths(client):
    response = client.get("/runs")

    assert response.status_code == 200
    runs = response.json()
    assert [run["run_name"] for run in runs] == ["simple_s42", "simple_s123", "stronger_s42"]
    assert runs[0]["best_epoch"] == 2
    assert runs[0]["best_val_accuracy"] == 41.0
    assert "checkpoint_path" not in runs[0]
    assert "train_config_json" not in runs[0]


def test_get_run_accepts_run_dir_with_slashes(client, experiment_dir):
    run_dir = str(experiment_dir / "stronger_s42")

    response = client.get(f"/runs/{run_dir}")

    assert response.status_code == 200
    run = response.json()
    assert run["run_dir"] == run_dir
    assert run["train_config"]["model_name"] == "stronger"
    assert run["train_config"]["epochs"] == 2


def test_get_unknown_run_is_404(client):
    response = client.get("/runs/runs/multiseed/exp/missing_s1")

    assert response.status_code == 404


def test_results_filter_by_model_seed_and_condition(client):
    response = client.get(
        "/results", params={"model": "simple", "training_seed": 123, "condition": "noise"}
    )

    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 1
    assert rows[0]["run_name"] == "simple_s123"
    assert rows[0]["accuracy"] == 20.0
    assert rows[0]["params"] == {"noise_seed": 42, "mean": 0.0, "std": 0.1}


def test_results_without_filters_returns_every_condition(client):
    rows = client.get("/results").json()

    assert len(rows) == 9
    assert {row["condition"] for row in rows} == {"clean", "blur", "noise"}


@pytest.mark.usefixtures("client")
def test_concurrent_requests_each_get_a_working_connection():
    # Sequential requests happen to reuse one idle threadpool thread, so only
    # concurrent ones catch get_db opening a connection on one thread and the
    # endpoint using it on another (sqlite's same-thread check).
    async def get_runs_concurrently():
        transport = httpx2.ASGITransport(app=api.app)
        async with httpx2.AsyncClient(transport=transport, base_url="http://test") as async_client:
            return await asyncio.gather(*[async_client.get("/runs") for _ in range(50)])

    responses = asyncio.run(get_runs_concurrently())

    assert all(response.status_code == 200 for response in responses)


def test_results_reject_unknown_condition(client):
    response = client.get("/results", params={"condition": "fog"})

    assert response.status_code == 422


def test_compare_returns_one_prediction_per_model(client):
    response = client.post(
        "/predict/compare", files={"file": ("cat.png", _png_bytes(), "image/png")}
    )

    assert response.status_code == 200
    predictions = {p["model"]: p for p in response.json()}
    assert predictions["simple"]["predicted_class"] == "Cat"
    assert predictions["stronger"]["predicted_class"] == "Dog"
    assert all(0.0 < p["confidence"] <= 1.0 for p in predictions.values())


def test_compare_rejects_non_image(client):
    response = client.post(
        "/predict/compare", files={"file": ("notes.txt", b"not an image", "text/plain")}
    )

    assert response.status_code == 400


def test_predict_single_model_still_works(client):
    response = client.post(
        "/predict",
        files={"file": ("cat.png", _png_bytes(), "image/png")},
        data={"model": "stronger"},
    )

    assert response.status_code == 200
    assert response.json()["predicted_class"] == "Dog"


def test_cors_allows_local_vite_dev_server(client):
    response = client.get("/runs", headers={"Origin": "http://localhost:5173"})

    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_rejects_other_origins(client):
    response = client.get("/runs", headers={"Origin": "https://evil.example"})

    assert "access-control-allow-origin" not in response.headers
