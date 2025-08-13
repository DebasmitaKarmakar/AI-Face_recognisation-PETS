import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import zipfile
import os

# -----------------------------
# Load trained face recognition model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenet_cats_dogs.h5")

model = load_model()

# Class name mapping (change these names to match your dataset)
class_names = {0: "Cat", 1: "Dog"}

# -----------------------------
# Page config & title
# -----------------------------
st.set_page_config(
    page_title="FaceVision: AI Recognition",
    page_icon="🧠",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>🧠 FaceVision: AI Recognition</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size: 16px;'>Upload an image and let our AI model identify it with high precision.</p>",
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar input
# -----------------------------
st.sidebar.header("📷 Upload Single Image")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_image(image, target_size=(224, 224)):
    img = np.array(image)
    if img.shape[-1] == 4:  # remove alpha channel if RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# Single image prediction
# -----------------------------
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict Identity"):
        with st.spinner("🤖 Analyzing image..."):
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_label = class_names.get(predicted_class_index, str(predicted_class_index))
            confidence = float(np.max(predictions) * 100)

        # Stylish Result Card
        st.markdown(
            f"""
            <div style="background-color: #2E7D32; padding: 15px; border-radius: 10px; margin-top: 10px;">
                <h3 style="color: white; text-align: center;">Predicted Class: {predicted_label}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="background-color: #1565C0; padding: 15px; border-radius: 10px; margin-top: 10px;">
                <h4 style="color: white; text-align: center;">Confidence: {confidence:.2f}%</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------
# Batch prediction section
# -----------------------------
st.markdown("---")
st.markdown("### 📂 Batch Prediction for Multiple Images")
batch_file = st.file_uploader(
    "Upload a ZIP file containing multiple images",
    type="zip"
)

if batch_file is not None:
    with zipfile.ZipFile(batch_file, "r") as zip_ref:
        zip_ref.extractall("batch_images")

    results = []
    for img_name in os.listdir("batch_images"):
        img_path = os.path.join("batch_images", img_name)
        image = Image.open(img_path)
        processed_img = preprocess_image(image)
        predictions = model.predict(processed_img)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_label = class_names.get(predicted_class_index, str(predicted_class_index))
        confidence = float(np.max(predictions) * 100)
        results.append([img_name, predicted_label, confidence])

    results_df = pd.DataFrame(results, columns=["Image Name", "Predicted Class", "Confidence (%)"])
    st.dataframe(results_df)

    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Batch Predictions CSV",
        csv,
        file_name='face_predictions.csv',
        mime='text/csv'
    )

st.markdown("---")
st.markdown("<p style='text-align: center;'>👩‍💻 <i>Built with ❤️ for AI and Computer Vision enthusiasts.</i></p>", unsafe_allow_html=True)

