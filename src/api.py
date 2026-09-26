from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import os
from .inference import load_model_registry, predict_img
from pydantic import BaseModel
from typing import Annotated, Literal
from fastapi import UploadFile, File, Form
from PIL import UnidentifiedImageError


@asynccontextmanager
async def lifespan(app:FastAPI):
    load_dotenv()
    simple_run_dir = os.environ["VISION_SIMPLE_RUN_DIR"]
    stronger_run_dir = os.environ['VISION_STRONGER_RUN_DIR']
    device = os.getenv("VISION_DEVICE", "cpu")
    app.state.models=load_model_registry(simple_run_dir, stronger_run_dir, device)

    yield

    app.state.models.clear()

app = FastAPI(lifespan=lifespan)


@app.get('/health')
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
    model: Annotated[Literal["simple", "stronger"], Form()],
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