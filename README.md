# Практическая работа №10. Классификация изображений и развертывание

## Структура проекта
* `models/` - Поместите сюда файл вашей обученной модели.
* `backend/` - FastAPI приложение, развертываемое на Render.com или Hugging Face Spaces.
* `frontend/` - Streamlit приложение с UI (загрузка файла / холст для рисования).

## Локальный запуск (тестирование)

### 1. Подготовка модели
Сохраните вашу Keras-модель в папку `models/` под именем `best_classification_model.keras`.

**Важно:** В скрипте `backend/main.py` в функции `preprocess_image` укажите правильную предобработку: размеры изображения (например, `28x28`), цвета (`Grayscale` или `RGB`) и нормализацию (по умолчанию `/ 255.0`), на которой обучалась модель!

### 2. Запуск Backend (API)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 7860
```
API будет доступен по адресу `http://localhost:7860`

### 3. Запуск Frontend (UI)
В новом терминале:
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```
Интерфейс откроется в браузере.

## Публикация
1. **Backend:** Загрузите репозиторий на GitHub и разверните на **Hugging Face Spaces** (через Dockerfile) или Render.com, как описано в инструкции к заданию.
2. **Frontend:** В файле `frontend/app.py` измените `API_URL` с `localhost:7860` на ссылку вашего развернутого бэкенда. Разверните `frontend` через **Streamlit Community Cloud** (подключив свой GitHub).