from __future__ import annotations

import io

import pandas as pd
import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Classifier", layout="wide")

DEFAULT_API_URL = "https://task-10-y0wh.onrender.com/predict"
TASK_OPTIONS = {
    "Images": {
        "task": "images",
        "classes": "cars, cats, dogs, flowers, horses, human",
        "canvas_bg": "#ffffff",
        "stroke": "#111111",
        "brush": 10,
        "hint": "Подходит для общей классификации изображений по 6 классам.",
    },
    "Digits": {
        "task": "digits",
        "classes": "0, 1, 2, 3, 4, 5, 6, 7, 8, 9",
        "canvas_bg": "#000000",
        "stroke": "#ffffff",
        "brush": 20,
        "hint": "Для рукописных цифр холст удобнее, чем загрузка фото.",
    },
}

st.markdown(
    """
    <style>
    :root {
        --bg: #0b0d10;
        --panel: #12161b;
        --panel-2: #171c22;
        --border: #242b34;
        --text: #e8edf2;
        --muted: #98a4b3;
        --accent: #dfe7ef;
        --accent-text: #0b0d10;
    }
    .stApp { background: var(--bg); color: var(--text); }
    [data-testid="stSidebar"] { background: #0f1318; border-right: 1px solid var(--border); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1180px; }
    h1, h2, h3, h4, h5, h6, p, label, div, span { color: var(--text); }
    .muted { color: var(--muted); font-size: 0.98rem; margin-bottom: 1.25rem; }
    .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.1rem 1.15rem;
    }
    .metric {
        background: var(--panel-2);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1rem;
        min-height: 116px;
    }
    .metric-label { color: var(--muted); font-size: 0.9rem; margin-bottom: 0.35rem; }
    .metric-value { font-size: 1.6rem; font-weight: 650; line-height: 1.2; }
    .divider { height: 1px; background: var(--border); margin: 1rem 0 1.25rem; }
    .small-note { color: var(--muted); font-size: 0.9rem; }
    button[kind="primary"] {
        background: var(--accent) !important;
        color: var(--accent-text) !important;
        border: none !important;
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Classification Interface")

with st.sidebar:
    st.subheader("Connection")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL).rstrip("/")
    if st.button("Check API", use_container_width=True):
        try:
            response = requests.get(f"{api_url}/health", timeout=10)
            response.raise_for_status()
            data = response.json()
            st.success("API is available")
            st.caption(f"Images model: {data['models_loaded'].get('images')}")
            st.caption(f"Digits model: {data['models_loaded'].get('digits')}")
        except Exception as exc:
            st.error(f"Connection error: {exc}")

    st.markdown("---")
    selected_label = st.selectbox("Task", list(TASK_OPTIONS.keys()))
    selected = TASK_OPTIONS[selected_label]
    st.caption(selected["hint"])
    st.caption(f"Classes: {selected['classes']}")

left, right = st.columns([1.08, 0.92], gap="large")
prepared_image = None

with left:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    mode = st.radio("Input", ["Upload file", "Draw on canvas"], horizontal=True)

    if mode == "Upload file":
        uploaded_file = st.file_uploader("Select image", type=["png", "jpg", "jpeg", "webp", "bmp"])
        if uploaded_file is not None:
            prepared_image = Image.open(uploaded_file).convert("RGB")
            st.image(prepared_image, use_container_width=True)
    else:
        canvas = st_canvas(
            fill_color="rgba(255,255,255,1)",
            stroke_width=selected["brush"],
            stroke_color=selected["stroke"],
            background_color=selected["canvas_bg"],
            width=420,
            height=420,
            drawing_mode="freedraw",
            key=f"canvas-{selected['task']}",
        )
        if canvas.image_data is not None:
            preview = Image.fromarray(canvas.image_data.astype("uint8"), mode="RGBA").convert("RGB")
            extrema = preview.convert("L").getextrema()
            if selected["task"] == "images" and extrema != (255, 255):
                prepared_image = preview
                st.image(prepared_image, width=300)
            if selected["task"] == "digits" and extrema != (0, 0):
                prepared_image = preview
                st.image(prepared_image, width=300)

    send = st.button("Run prediction", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Prediction")
    result_placeholder = st.empty()
    chart_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

if send:
    if prepared_image is None:
        right.warning("Load an image or draw on the canvas first.")
    else:
        buffer = io.BytesIO()
        prepared_image.save(buffer, format="PNG")
        buffer.seek(0)
        try:
            response = requests.post(
                f"{api_url}/predict",
                files={"file": ("input.png", buffer, "image/png")},
                data={"task": selected["task"]},
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.HTTPError:
            detail = response.text if 'response' in locals() else 'Unknown error'
            right.error(f"API error: {detail}")
        except Exception as exc:
            right.error(f"Request failed: {exc}")
        else:
            ordered = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)
            with result_placeholder.container():
                c1, c2, c3 = st.columns(3)
                c1.markdown(
                    f"<div class='metric'><div class='metric-label'>Task</div><div class='metric-value'>{result['task']}</div></div>",
                    unsafe_allow_html=True,
                )
                c2.markdown(
                    f"<div class='metric'><div class='metric-label'>Class</div><div class='metric-value'>{result['predicted_class']}</div></div>",
                    unsafe_allow_html=True,
                )
                c3.markdown(
                    f"<div class='metric'><div class='metric-label'>Confidence</div><div class='metric-value'>{result['confidence'] * 100:.2f}%</div></div>",
                    unsafe_allow_html=True,
                )
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.caption(f"Inference time: {result['inference_time_ms']:.2f} ms")
                for label, prob in ordered:
                    st.write(label)
                    st.progress(float(prob))
                    st.markdown(f"<div class='small-note'>{prob * 100:.2f}%</div>", unsafe_allow_html=True)

            with chart_placeholder.container():
                df = pd.DataFrame(ordered, columns=["class", "probability"]).set_index("class")
                st.bar_chart(df)