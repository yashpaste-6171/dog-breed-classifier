
import json
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Dog Breed Classifier",
    page_icon="🐶"
)

st.title("🐶 Dog Breed Classification")
st.write("Upload a dog photo to predict its breed.")

MODEL_PATH = "best_dog_breed_model.keras"
CLASSES_PATH = "dog_breed_classes.json"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

@st.cache_data
def load_classes():
    with open(CLASSES_PATH, "r") as f:
        return json.load(f)

try:
    model = load_model()
    class_names = load_classes()
except Exception as e:
    st.error(f"Could not load the model or breed labels: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a dog image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded dog image",
        use_container_width=True
    )

    if st.button("Predict Breed"):
        image = ImageOps.fit(image, (224, 224))
        image_array = np.asarray(image).astype("float32")
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)[0]
        top_indices = np.argsort(predictions)[-3:][::-1]

        st.subheader("Top 3 Predicted Breeds")

        for index in top_indices:
            breed = class_names[index]
            confidence = predictions[index] * 100

            st.write(
                f"**{breed.replace('_', ' ').title()}**"
                f" — {confidence:.2f}%"
            )

st.caption("Predictions are estimates, not guaranteed identifications.")
