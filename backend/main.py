from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = FastAPI(title="Multi-Model Image Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Вы можете положить обе модели в папку models/
MODEL_IMAGES_PATH = "models/best_classification_model.keras"
MODEL_DIGITS_PATH = "models/digits_model.keras"  # Укажите реальное имя модели для цифр

CLASS_NAMES_IMAGES = ['bike', 'cars', 'cats', 'dogs', 'flowers', 'horses', 'human']
CLASS_NAMES_DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

models = {}

# Загрузка моделей (FastAPI запустится, даже если одной из них нет - просто выдаст Warning)
try:
    models["images"] = tf.keras.models.load_model(MODEL_IMAGES_PATH)
    print("Image classification model ('images') loaded successfully!")
except Exception as e:
    print(f"Warning: Image model not loaded. {e}")

try:
    models["digits"] = tf.keras.models.load_model(MODEL_DIGITS_PATH)
    print("Digit classification model ('digits') loaded successfully!")
except Exception as e:
    print(f"Warning: Digit model not loaded. {e}")


def preprocess_for_images(img: Image.Image):
    """Предобработка цветных изображений (7 классов)"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img = img.resize((224, 224)) 
    arr = np.array(img, dtype=np.float32) 
    
    return np.expand_dims(arr, axis=0)

def preprocess_for_digits(img: Image.Image):
    """Предобработка черно-белых цифр (MNIST)"""
    # MNIST обычно состоит из белых цифр на черном фоне 28x28 (Grayscale)
    if img.mode != "L":
        img = img.convert("L")
    img = img.resize((28, 28))
    arr = np.array(img) / 255.0
    
    # Поскольку canvas выдает нам черные рисунки на белом фоне (или наоборот),
    # может понадобиться инвертировать цвет пикселей для MNIST:
    # arr = 1.0 - arr # [Раскомментируйте, если предсказания цифр ошибаются]
    
    # Формат зависит от того как вы обучали: обычно (1, 28, 28) или (1, 28, 28, 1)
    arr = np.expand_dims(arr, axis=-1) # добавляем канал
    return np.expand_dims(arr, axis=0) # добавляем батч

@app.post("/predict/")
async def predict_image(
    file: UploadFile = File(...),
    model_type: str = Form(...) # Теперь требуется параметр из формы
):
    if model_type not in models or models[model_type] is None:
        raise HTTPException(status_code=500, detail=f"Модель '{model_type}' не найдена или не загружена на сервере.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if model_type == "images":
            processed_image = preprocess_for_images(image)
            classes = CLASS_NAMES_IMAGES
        elif model_type == "digits":
            processed_image = preprocess_for_digits(image)
            classes = CLASS_NAMES_DIGITS
        else:
            raise ValueError("Unknown model_type")
            
        # Получаем предсказания
        predictions = models[model_type].predict(processed_image)
        
        predicted_class_index = int(np.argmax(predictions[0]))
        predicted_class_name = classes[predicted_class_index]
        
        confidence = float(np.max(predictions[0]))
        all_probabilities = predictions[0].tolist()
        
        return {
            "model_used": model_type,
            "predicted_class_index": predicted_class_index,
            "predicted_class_name": predicted_class_name,
            "confidence": confidence,
            "all_probabilities": all_probabilities,
            "classes": classes
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)