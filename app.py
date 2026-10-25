
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Dog Breed Classifier",
    page_icon="🐶",
    layout="centered"
)

MODEL_PATH = Path("best_dog_breed_model.keras")
CLASSES_PATH = Path("dog_breed_classes.json")
IMAGE_SIZE = (224, 224)

st.title("🐶 Dog Breed Classifier")
st.write("Upload a dog photo to discover its predicted breed.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH, compile=False
    )

@st.cache_data
def load_classes():
    with open(CLASSES_PATH, "r") as file:
        return json.load(file)

if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.error(
        "The model or breed labels are not uploaded yet. "
        "Please check the deployment files."
    )
    st.stop()

model = load_model()
classes = load_classes()

uploaded_file = st.file_uploader(
    "Choose a dog image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your uploaded image",
             use_container_width=True)

    if st.button("Predict Breed"):
        with st.spinner("Analyzing image..."):
            prepared = ImageOps.fit(image, IMAGE_SIZE)
            array = np.asarray(prepared, dtype=np.float32)
            array = np.expand_dims(array / 255.0, axis=0)

            predictions = model.predict(array, verbose=0)[0]
            top_indices = np.argsort(predictions)[-3:][::-1]

        st.subheader("Prediction results")

        for index in top_indices:
            st.write(
                f"**{classes[int(index)].replace('_', ' ').title()}** "
                f"— {predictions[index] * 100:.2f}%"
            )
