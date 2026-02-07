import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

MODEL_PATH = 'rice_leaf_disease_model.h5'
IMG_HEIGHT = 224
IMG_WIDTH = 224

CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf scald",
    "Narrow Brown Leaf Spot",
    "Neck_Blast",
    "Rice Hispa",
    "Sheath Blight"
]

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = np.max(predictions[0]) * 100

    return predicted_class_name, confidence

st.set_page_config(page_title="Rice Leaf Disease Predictor", layout="centered")

st.title("Rice Leaf Disease Predictor")
st.write("Upload an image of a rice leaf to predict if it has a disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    with open(os.path.join("temp_image.png"), "wb") as f:
        f.write(uploaded_file.getbuffer())

    predicted_class, confidence = predict_image("temp_image.png")

    st.success(f"*Prediction:* {predicted_class}")
    st.info(f"*Confidence:* {confidence:.2f}%")

    st.markdown("---")
    st.subheader("All Probabilities:")
    img = image.load_img("temp_image.png", target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    all_predictions = model.predict(img_array)[0]

    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"- {class_name}: {all_predictions[i]*100:.2f}%")

    os.remove("temp_image.png")
