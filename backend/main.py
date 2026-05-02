import io
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────
# Настройка логгера
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Глобальные константы
# ─────────────────────────────────────────────
MODEL_PATH = Path("best_classification_model.h5")

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Входной размер изображения (должен совпадать с тем, что использовалось при обучении)
IMG_SIZE = (32, 32)

# ─────────────────────────────────────────────
# Загрузка модели при старте приложения
# ─────────────────────────────────────────────
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружаем модель один раз при запуске сервиса."""
    global model
    try:
        import tensorflow as tf  # noqa: F401 – импорт только при наличии TF
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"✅ Модель загружена из {MODEL_PATH}")
    except FileNotFoundError:
        logger.warning(
            f"⚠️  Файл модели '{MODEL_PATH}' не найден. "
            "API запущен в демо-режиме — возвращаются случайные предсказания."
        )
    except Exception as exc:
        logger.error(f"❌ Ошибка при загрузке модели: {exc}")
    yield
    # Shutdown
    model = None
    logger.info("Сервис остановлен.")


# ─────────────────────────────────────────────
# Инициализация FastAPI
# ─────────────────────────────────────────────
app = FastAPI(
    title="Image Classification API",
    description=(
        "API для классификации изображений. "
        "Практическая работа №10 — Нейронные сети."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Разрешаем запросы с любых источников (нужно для Streamlit-фронтенда)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Схемы ответов
# ─────────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_class_index: int
    confidence: float
    probabilities: dict[str, float]
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: list[str]


# ─────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Читает изображение из байтов, изменяет размер до IMG_SIZE,
    нормализует пиксели в диапазон [0, 1] и добавляет batch-измерение.

    Args:
        image_bytes: Сырые байты загруженного файла.

    Returns:
        np.ndarray формы (1, H, W, C) — готовый тензор для инференса.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)


def run_inference(tensor: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Запускает инференс (или демо-режим) и возвращает вероятности и время.

    Args:
        tensor: Предобработанный тензор формы (1, H, W, C).

    Returns:
        Кортеж (probabilities: np.ndarray формы (N,), inference_time_ms: float).
    """
    t0 = time.perf_counter()

    if model is not None:
        probs = model.predict(tensor, verbose=0)[0]
    else:
        # Демо-режим: случайные вероятности
        raw = np.random.dirichlet(np.ones(len(CLASS_NAMES)))
        probs = raw.astype(np.float32)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return probs, elapsed_ms


# ─────────────────────────────────────────────
# Эндпоинты
# ─────────────────────────────────────────────
@app.get("/", tags=["Root"])
def root():
    """Корневой маршрут — краткая справка."""
    return {
        "message": "Image Classification API работает. Отправьте POST на /predict для классификации изображения.",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Проверка состояния сервиса и наличия загруженной модели."""
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        classes=CLASS_NAMES,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(..., description="Изображение для классификации (JPG, PNG, BMP, WEBP)")):
    """
    Классифицирует загруженное изображение.

    - Принимает изображение в форматах JPG, PNG, BMP, WEBP.
    - Возвращает предсказанный класс, уверенность и вектор вероятностей.
    """
    # Проверка MIME-типа
    allowed_types = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Неподдерживаемый тип файла '{file.content_type}'. Допустимые: {allowed_types}",
        )

    # Чтение байтов
    try:
        image_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать файл: {exc}")

    # Предобработка
    try:
        tensor = preprocess_image(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Ошибка при предобработке изображения: {exc}")

    # Инференс
    probs, inference_ms = run_inference(tensor)

    # Формирование ответа
    class_idx = int(np.argmax(probs))
    return PredictionResponse(
        predicted_class=CLASS_NAMES[class_idx],
        predicted_class_index=class_idx,
        confidence=float(probs[class_idx]),
        probabilities={name: float(p) for name, p in zip(CLASS_NAMES, probs)},
        inference_time_ms=round(inference_ms, 3),
    )