from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Skin Lesion Image Classifier",
    page_icon="🔬",
    layout="centered",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "isic_skin_lesion_model.keras"
IMG_SIZE = (64, 64)

st.title("Skin Lesion Image Classifier")
st.caption("Research/demo interface for the uploaded CNN model")

st.warning(
    "Research demonstration only — this model does NOT diagnose skin cancer. "
    "Its training script creates labels from image brightness (darker vs. lighter), "
    "not medically verified diagnoses. Do not use its output for medical decisions. "
    "For a concerning or changing lesion, consult a qualified healthcare professional."
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Place it in the same folder as app.py."
        )
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the CNN model: {exc}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG or PNG image for the model's darker/lighter image classification.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        resized = image.resize(IMG_SIZE)
        image_array = np.asarray(resized, dtype=np.float32) / 255.0
        batch = np.expand_dims(image_array, axis=0)

        if st.button("Run image classification", type="primary"):
            with st.spinner("Running the model..."):
                prediction = np.asarray(model.predict(batch, verbose=0)).reshape(-1)
                score = float(prediction[0])

            if not np.isfinite(score):
                st.error("The model returned an invalid score.")
                st.stop()

            # Training code: sigmoid > 0.5 maps to class 1 (lighter);
            # otherwise class 0 (darker).
            if score > 0.5:
                label = "Class 1 — lighter image"
                confidence = score
            else:
                label = "Class 0 — darker image"
                confidence = 1.0 - score

            st.subheader("Model output")
            st.write(f"**Predicted image class:** {label}")
            st.metric("Model score for Class 1 (lighter)", f"{score:.1%}")
            st.progress(float(np.clip(confidence, 0.0, 1.0)))
            st.caption(
                f"Confidence relative to the two output classes: {confidence:.1%}. "
                "This is not a probability of cancer or malignancy."
            )
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")

st.divider()
st.caption(
    "Model input preprocessing follows the training script: RGB image, resized "
    "to 64 × 64 pixels, pixel values scaled to 0–1."
)
