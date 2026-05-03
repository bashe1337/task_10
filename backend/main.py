from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMAGE_MODEL_PATH = Path("models/model_task5.keras")
DIGITS_MODEL_PATH = Path("models/digits_model.keras")

IMAGE_CLASSES = ["bike", "cars", "cats", "dogs", "flowers", "horses", "human"]
DIGIT_CLASSES = [str(i) for i in range(10)]

IMAGE_SIZE = (224, 224)
DIGIT_SIZE = (28, 28)
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

models: dict[str, tf.keras.Model | None] = {"images": None, "digits": None}


class PredictionResponse(BaseModel):
    task: str
    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: dict[str, float]
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
    image_classes: list[str]
    digit_classes: list[str]
    image_size: tuple[int, int]
    digit_size: tuple[int, int]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if IMAGE_MODEL_PATH.exists():
        models["images"] = tf.keras.models.load_model(IMAGE_MODEL_PATH)
    if DIGITS_MODEL_PATH.exists():
        models["digits"] = tf.keras.models.load_model(DIGITS_MODEL_PATH)
    yield
    models["images"] = None
    models["digits"] = None


app = FastAPI(
    title="Classification API",
    version="2.1.0",
    description="API для классификации изображений и рукописных цифр.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(img: Image.Image):
    """Предобработка цветных изображений (7 классов)"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img = img.resize((224, 224)) 
    arr = np.array(img, dtype=np.float32) 
    
    return np.expand_dims(arr, axis=0)


def preprocess_digit(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = ImageOps.invert(image)
    image = image.resize(DIGIT_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=-1)
    return np.expand_dims(array, axis=0)


def validate_output(predictions: np.ndarray, classes: list[str]) -> np.ndarray:
    probs = np.asarray(predictions[0], dtype=np.float32)
    if probs.shape[0] != len(classes):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Размер выхода модели ({probs.shape[0]}) не совпадает с числом классов "
                f"({len(classes)})."
            ),
        )
    return probs


@app.get("/", tags=["service"])
def root():
    return {
        "service": "classification-api",
        "predict_endpoint": "/predict",
        "health_endpoint": "/health",
        "supported_tasks": ["images", "digits"],
    }


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health():
    return HealthResponse(
        status="ok",
        models_loaded={
            "images": models["images"] is not None,
            "digits": models["digits"] is not None,
        },
        image_classes=IMAGE_CLASSES,
        digit_classes=DIGIT_CLASSES,
        image_size=IMAGE_SIZE,
        digit_size=DIGIT_SIZE,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(
    file: UploadFile = File(...),
    task: str = Form(...),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"Неподдерживаемый тип файла: {file.content_type}")

    if task not in {"images", "digits"}:
        raise HTTPException(status_code=400, detail="task должен быть 'images' или 'digits'.")

    if models[task] is None:
        expected = IMAGE_MODEL_PATH if task == "images" else DIGITS_MODEL_PATH
        raise HTTPException(status_code=500, detail=f"Модель для task='{task}' не загружена. Ожидается файл: {expected}")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Файл пустой.")

    try:
        if task == "images":
            batch = preprocess_image(image_bytes)
            classes = IMAGE_CLASSES
        else:
            batch = preprocess_digit(image_bytes)
            classes = DIGIT_CLASSES

        start = time.perf_counter()
        predictions = models[task].predict(batch, verbose=0)
        elapsed = (time.perf_counter() - start) * 1000
        probs = validate_output(predictions, classes)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Ошибка обработки изображения: {exc}")

    predicted_index = int(np.argmax(probs))
    predicted_class = classes[predicted_index]
    confidence = float(probs[predicted_index])

    return PredictionResponse(
        task=task,
        predicted_class=predicted_class,
        predicted_index=predicted_index,
        confidence=round(confidence, 6),
        probabilities={name: round(float(prob), 6) for name, prob in zip(classes, probs)},
        inference_time_ms=round(elapsed, 3),
    )