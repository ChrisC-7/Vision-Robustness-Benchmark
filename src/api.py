from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import json
import os
import sqlite3
from . import tracking
from .inference import load_model_registry, predict, predict_img, preprocess_image
from pydantic import BaseModel
from typing import Annotated, Any, Iterator, Literal, Optional
from fastapi import UploadFile, File, Form
from PIL import UnidentifiedImageError

# Loaded at import, not in lifespan: the CORS origins below are read while
# the module is imported, before lifespan ever runs.
load_dotenv()

ModelName = Literal["simple", "stronger"]
Condition = Literal["clean", "blur", "noise"]


@asynccontextmanager
async def lifespan(app:FastAPI):
    simple_run_dir = os.environ["VISION_SIMPLE_RUN_DIR"]
    stronger_run_dir = os.environ['VISION_STRONGER_RUN_DIR']
    device = os.getenv("VISION_DEVICE", "cpu")
    app.state.models=load_model_registry(simple_run_dir, stronger_run_dir, device)

    yield

    app.state.models.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("VISION_CORS_ORIGINS", "http://localhost:5173").split(",")
        if origin.strip()
    ],
    allow_methods=["GET", "POST"],
)


def get_db() -> Iterator[sqlite3.Connection]:
    # FastAPI runs a sync dependency and the sync endpoint that uses it on
    # threadpool threads that are not guaranteed to be the same one, so the
    # same-thread check has to be off. Each request still opens its own
    # connection, so no two threads ever use one at the same time.
    conn = tracking.get_connection(
        os.getenv("VISION_DB_PATH", "experiments.db"), check_same_thread=False
    )
    try:
        yield conn
    finally:
        conn.close()

Db = Annotated[sqlite3.Connection, Depends(get_db)]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    models: list[ModelName]

@app.get('/health', response_model=HealthResponse)
def status():
    return {
        "status": "ok",
        "models": list(app.state.models.keys()),
    }

class PredictionResponse(BaseModel):
    model: str
    predicted_index: int
    predicted_class: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
def predict_image(
    file: Annotated[UploadFile, File()],
    model: Annotated[ModelName, Form()],
):

    selected_model = app.state.models[model]
    try:
        result = predict_img(selected_model, file.file)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Please upload an image file")

    return {
        "model": model,
        **result,
    }

@app.post("/predict/compare", response_model=list[PredictionResponse])
def compare_models(file: Annotated[UploadFile, File()]):
    # Decode and resize once; every model then sees the exact same tensor.
    try:
        img_tensor = preprocess_image(file.file)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Please upload an image file")

    return [
        {"model": name, **predict(model, img_tensor)}
        for name, model in app.state.models.items()
    ]


# Response models are the public contract, and the reason checkpoint_path,
# created_at and updated_at never leave the server: FastAPI drops every
# field a response model does not declare.
class RunSummary(BaseModel):
    run_dir: str
    run_name: str
    model_name: ModelName
    training_seed: Optional[int]
    split_seed: Optional[int]
    best_epoch: Optional[int]
    best_val_accuracy: Optional[float]

class RunDetail(RunSummary):
    # Kept as a free-form dict: tracking stores whatever config.json held,
    # and older TrainConfig shapes have different fields.
    train_config: dict[str, Any]

class EvalResult(BaseModel):
    run_dir: str
    run_name: str
    model_name: ModelName
    training_seed: Optional[int]
    eval_id: str
    noise_seed: Optional[int]
    condition: Condition
    loss: float
    accuracy: float
    params: dict[str, Any]

@app.get("/runs", response_model=list[RunSummary])
def list_runs(db: Db):
    return tracking.list_train_runs(db)

# ":path" lets run_dir keep its slashes, e.g. runs/multiseed/experiment1/simple_s42.
# The value is only ever compared against the database, never opened as a file.
@app.get("/runs/{run_dir:path}", response_model=RunDetail)
def get_run(run_dir: str, db: Db):
    run = tracking.get_train_run(db, run_dir)
    if run is None:
        raise HTTPException(status_code=404, detail=f"No run recorded for {run_dir!r}")
    run["train_config"] = json.loads(run.pop("train_config_json"))
    return run

@app.get("/results", response_model=list[EvalResult])
def list_results(
    db: Db,
    model: Optional[ModelName] = None,
    training_seed: Optional[int] = None,
    condition: Optional[Condition] = None,
):
    rows = tracking.query_eval_results(
        db, model_name=model, training_seed=training_seed, condition=condition
    )
    for row in rows:
        row["params"] = json.loads(row.pop("params_json"))
    return rows
