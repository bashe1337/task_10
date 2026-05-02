# -*- coding: utf-8 -*-
"""
Практическая работа №10 — Фронтенд (Streamlit)
Интерфейс для классификации изображений через API.
"""

import io
import requests
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

# ─────────────────────────────────────────────
# Конфигурация страницы
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Image Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Константы
# ─────────────────────────────────────────────
DEFAULT_API_URL = "https://YOUR_USERNAME-YOUR_SPACE.hf.space"

# ─────────────────────────────────────────────
# Стили
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #01696f;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #7a7974;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f9f8f5;
        border: 1px solid #dcd9d5;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .pred-label {
        font-size: 1.6rem;
        font-weight: 700;
        color: #01696f;
    }
    .confidence {
        font-size: 1rem;
        color: #7a7974;
    }
    .status-ok   { color: #437a22; font-weight: 600; }
    .status-fail { color: #a12c7b; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Боковая панель — настройки
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Настройки")
    api_url = st.text_input(
        "URL бэкенда",
        value=DEFAULT_API_URL,
        help="Адрес развёрнутого FastAPI-сервиса (без /predict)",
    ).rstrip("/")

    st.markdown("---")
    st.markdown("### Проверка подключения")
    if st.button("🔍 Проверить API"):
        try:
            r = requests.get(f"{api_url}/health", timeout=10)
            data = r.json()
            if data.get("status") == "ok":
                model_status = "✅ загружена" if data.get("model_loaded") else "⚠️ не найдена (демо-режим)"
                st.success(f"API доступен\nМодель: {model_status}")
                st.info(f"Классы: {', '.join(data.get('classes', []))}")
            else:
                st.error("API вернул неожиданный ответ")
        except Exception as e:
            st.error(f"Не удалось подключиться:\n{e}")

    st.markdown("---")
    st.markdown(
        "**Практическая работа №10**\n\nНейронные сети — Классификация изображений",
        unsafe_allow_html=False,
    )

# ─────────────────────────────────────────────
# Заголовок
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">🧠 Image Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Загрузите изображение или нарисуйте его, чтобы получить предсказание модели.</p>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Вкладки — способы ввода
# ─────────────────────────────────────────────
tab_upload, tab_draw = st.tabs(["📁 Загрузить файл", "✏️ Нарисовать"])

image_to_send: Image.Image | None = None  # итоговое изображение для отправки

# ── Вкладка 1: загрузка файла ─────────────────
with tab_upload:
    uploaded = st.file_uploader(
        "Выберите изображение (JPG, PNG, BMP, WEBP)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )
    if uploaded:
        image_to_send = Image.open(uploaded).convert("RGB")
        st.image(image_to_send, caption="Загруженное изображение", use_column_width=False, width=300)

# ── Вкладка 2: рисование на холсте ───────────
with tab_draw:
    st.markdown("Нарисуйте изображение на холсте ниже:")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=st.slider("Толщина кисти", 1, 30, 8, key="stroke"),
        stroke_color=st.color_picker("Цвет кисти", "#000000", key="color"),
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas_result.image_data is not None:
        arr = canvas_result.image_data.astype(np.uint8)
        # Если холст не пустой (не все пиксели белые)
        if not (arr[:, :, :3] == 255).all():
            image_to_send = Image.fromarray(arr).convert("RGB")
            st.image(image_to_send, caption="Нарисованное изображение", width=200)

# ─────────────────────────────────────────────
# Кнопка классификации
# ─────────────────────────────────────────────
st.markdown("---")
col_btn, col_hint = st.columns([1, 3])
with col_btn:
    classify_btn = st.button("🚀 Классифицировать", type="primary", use_container_width=True)
with col_hint:
    if image_to_send is None:
        st.info("⬆️ Сначала загрузите или нарисуйте изображение")

# ─────────────────────────────────────────────
# Отправка запроса и отображение результатов
# ─────────────────────────────────────────────
if classify_btn and image_to_send is not None:
    with st.spinner("Отправляем изображение на сервер..."):
        # Конвертируем PIL → bytes (PNG)
        buf = io.BytesIO()
        image_to_send.save(buf, format="PNG")
        buf.seek(0)

        try:
            response = requests.post(
                f"{api_url}/predict",
                files={"file": ("image.png", buf, "image/png")},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.ConnectionError:
            st.error("❌ Не удалось подключиться к API. Проверьте URL в боковой панели.")
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ Ошибка API ({response.status_code}): {response.text}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Неожиданная ошибка: {e}")
            st.stop()

    # ── Результаты ────────────────────────────
    st.markdown("## 📊 Результаты классификации")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="confidence">Предсказанный класс</div>'
            f'<div class="pred-label">{result["predicted_class"].upper()}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col2:
        conf_pct = result["confidence"] * 100
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="confidence">Уверенность</div>'
            f'<div class="pred-label">{conf_pct:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="confidence">Время инференса</div>'
            f'<div class="pred-label">{result["inference_time_ms"]:.1f} ms</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Распределение вероятностей по классам")

    # Сортируем по убыванию вероятности
    probs = result["probabilities"]
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_probs]
    values = [round(item[1] * 100, 2) for item in sorted_probs]

    # Отображаем прогресс-бары
    for label, val in zip(labels, values):
        is_top = label == result["predicted_class"]
        bar_color = "#01696f" if is_top else "#dcd9d5"
        label_bold = f"**{label}**" if is_top else label
        col_l, col_b, col_v = st.columns([2, 6, 1])
        with col_l:
            st.markdown(label_bold)
        with col_b:
            st.progress(val / 100)
        with col_v:
            st.markdown(f"`{val:.1f}%`")

    # Интерактивный bar chart через st.bar_chart
    st.markdown("### График вероятностей")
    import pandas as pd
    chart_df = pd.DataFrame({"Вероятность (%)": values}, index=labels)
    st.bar_chart(chart_df)

elif classify_btn and image_to_send is None:
    st.warning("⚠️ Нет изображения для классификации. Загрузите файл или нарисуйте на холсте.")