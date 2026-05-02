# Практическая работа №10 — Сравнение моделей классификации изображений и развёртывание API

## Обзор проекта

Данный проект выполняет сравнительный анализ нескольких нейросетевых моделей классификации изображений (обученных в работах 2–5), выбирает лучшую по метрике F1, разворачивает её в виде REST API на FastAPI и предоставляет пользовательский интерфейс на Streamlit.

***

## Структура репозитория

```
.
├── backend/
│   ├── main.py                      # FastAPI-приложение
│   ├── requirements.txt             # Зависимости бэкенда
│   ├── Dockerfile                   # Docker-образ для деплоя
│   └── best_classification_model.h5 # Лучшая сохранённая модель
├── frontend/
│   ├── app.py                       # Streamlit-приложение
│   └── requirements.txt             # Зависимости фронтенда
└── README.md                        # Документация (этот файл)
```

***

## Датасет

В работе использован датасет **CIFAR-10** — стандартный бенчмарк для классификации изображений.

| Параметр | Значение |
|---|---|
| Количество классов | 10 |
| Обучающая выборка | 50 000 изображений |
| Тестовая выборка | 10 000 изображений |
| Размер изображений | 32×32 пикселей, RGB |
| Классы | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |

***

## Сравниваемые модели

В ходе работы сравнивались четыре модели, обученные в практических работах 2–5:

| # | Модель | Архитектура | Особенности |
|---|---|---|---|
| 1 | **Custom CNN** | Свёрточная сеть с нуля | 3 Conv-блока + MaxPooling + Dropout |
| 2 | **VGG-like** | Глубокая CNN в стиле VGG | BatchNorm, увеличенная глубина |
| 3 | **ResNet (Transfer)** | ResNet50 (предобученная) | Fine-tuning последних слоёв на CIFAR-10 |
| 4 | **MobileNet (Transfer)** | MobileNetV2 (предобученная) | Лёгкая модель, быстрый инференс |

***

## Результаты сравнения моделей

> **Примечание:** Приведённые значения являются примерными ориентирами. Замените их реальными числами из вашего обучения.

| Модель | Accuracy | Precision | Recall | F1-мера | Время инференса (мс) |
|---|---|---|---|---|---|
| Custom CNN | 0.812 | 0.815 | 0.812 | 0.811 | 3.2 |
| VGG-like | 0.871 | 0.873 | 0.871 | 0.870 | 8.7 |
| ResNet50 (Transfer) | **0.924** | **0.926** | **0.924** | **0.923** | 14.1 |
| MobileNetV2 (Transfer) | 0.901 | 0.903 | 0.901 | 0.900 | 5.3 |

**Лучшая модель по F1-мере: ResNet50 (Transfer Learning)** — сохранена как `best_classification_model.h5`.

***

## API — описание эндпоинтов

Бэкенд реализован на **FastAPI** и предоставляет следующие маршруты:

### `GET /`
Корневой маршрут. Возвращает краткую справку и ссылку на документацию.

### `GET /health`
Проверка состояния сервиса.

**Пример ответа:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "classes": ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]
}
```

### `POST /predict`
Классификация изображения.

**Параметры запроса:** `multipart/form-data`, поле `file` — файл изображения (JPG, PNG, BMP, WEBP).

**Пример ответа:**
```json
{
  "predicted_class": "cat",
  "predicted_class_index": 3,
  "confidence": 0.912,
  "probabilities": {
    "airplane": 0.003,
    "automobile": 0.005,
    "bird": 0.012,
    "cat": 0.912,
    "deer": 0.021,
    "dog": 0.031,
    "frog": 0.004,
    "horse": 0.006,
    "ship": 0.003,
    "truck": 0.003
  },
  "inference_time_ms": 12.5
}
```

### Интерактивная документация API
После запуска сервиса документация доступна по адресу:
```
http://localhost:8000/docs        ← Swagger UI
http://localhost:8000/redoc       ← ReDoc
```

***

## Примеры использования API

### curl
```bash
curl -X POST "https://YOUR_API_URL/predict" \
     -H "accept: application/json" \
     -F "file=@/path/to/image.jpg"
```

### Python (requests)
```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "https://YOUR_API_URL/predict",
        files={"file": ("image.jpg", f, "image/jpeg")},
    )

result = response.json()
print(f"Класс: {result['predicted_class']}")
print(f"Уверенность: {result['confidence']:.1%}")
print(f"Время: {result['inference_time_ms']} мс")
```

### JavaScript (fetch)
```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const response = await fetch("https://YOUR_API_URL/predict", {
  method: "POST",
  body: formData,
});
const result = await response.json();
console.log(result.predicted_class, result.confidence);
```

***

## Локальное развёртывание

### Требования
- Python 3.12+
- pip

### Бэкенд (FastAPI)

```bash
# 1. Перейдите в папку бэкенда
cd backend

# 2. Создайте виртуальное окружение
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 3. Установите зависимости
pip install -r requirements.txt

# 4. Поместите модель рядом с main.py
# best_classification_model.h5 → backend/best_classification_model.h5

# 5. Запустите сервер
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API будет доступен по адресу: `http://localhost:8000`

### Фронтенд (Streamlit)

```bash
# В отдельном терминале
cd frontend

pip install -r requirements.txt

streamlit run app.py
```

Приложение откроется по адресу: `http://localhost:8501`

В боковой панели укажите URL бэкенда: `http://localhost:8000`

***

## Развёртывание в облаке

### Бэкенд → Hugging Face Spaces (Docker)

```bash
# 1. Создайте Space на huggingface.co (SDK: Docker, Visibility: Public)
# 2. Добавьте удалённый репозиторий
git remote add hf https://huggingface.co/spaces/USERNAME/SPACE_NAME
# 3. Запушьте папку backend/
git subtree push --prefix backend hf main
```

После деплоя API доступен по адресу:
```
https://USERNAME-SPACE_NAME.hf.space
```

### Фронтенд → Streamlit Cloud

1. Загрузите папку `frontend/` в GitHub-репозиторий
2. Зайдите на [share.streamlit.io](https://share.streamlit.io) и войдите через GitHub
3. Нажмите **"New app"** → выберите репозиторий
4. Укажите **Main file path**: `frontend/app.py`
5. Нажмите **"Deploy"**
6. После деплоя введите URL бэкенда в боковой панели приложения

***

## Ссылки на развёрнутые сервисы

| Сервис | URL |
|---|---|
| GitHub-репозиторий | **ВАША ССЫЛКА** |
| Публичный API (POST /predict) | **ВАША ССЫЛКА** |
| Streamlit-интерфейс | **ВАША ССЫЛКА** |

***

## Зависимости

### Бэкенд

| Пакет | Версия | Назначение |
|---|---|---|
| fastapi | 0.115.6 | Web-фреймворк |
| uvicorn | 0.32.1 | ASGI-сервер |
| tensorflow | 2.17.0 | Загрузка и инференс модели |
| numpy | 1.26.4 | Работа с массивами |
| Pillow | 10.4.0 | Предобработка изображений |
| pydantic | 2.9.2 | Валидация данных |
| python-multipart | 0.0.12 | Загрузка файлов |

### Фронтенд

| Пакет | Версия | Назначение |
|---|---|---|
| streamlit | 1.41.1 | UI-фреймворк |
| streamlit-drawable-canvas | 0.9.3 | Холст для рисования |
| requests | 2.32.3 | HTTP-запросы к API |
| Pillow | 10.4.0 | Работа с изображениями |
| numpy | 1.26.4 | Работа с массивами |
| pandas | 2.2.3 | Формирование данных для графиков |