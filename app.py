
import json
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Dog Breed Classifier",
    page_icon="🐶",
    layout="centered"
)

st.title("🐶 Dog Breed Classification")
st.write("Upload a dog photo to predict its breed.")

REPO_ID = "yashog19/dog-breed-classifier-model"
MODEL_NAME = "best_dog_breed_model.keras"
CLASSES_PATH = "dog_breed_classes.json"
IMG_SIZE = (224, 224)


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_NAME,
        repo_type="model"
    )
    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_classes():
    with open(CLASSES_PATH, "r") as f:
        return json.load(f)


try:
    model = load_model()
    class_names = load_classes()
except Exception as e:
    st.error("Could not load the model or breed labels.")
    st.exception(e)
    st.stop()


uploaded_file = st.file_uploader(
    "Choose a dog image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded dog", use_container_width=True)

    # Match the training pipeline: resize to 224x224,
    # convert to float32, and do not divide by 255.
    resized_image = image.resize(IMG_SIZE)
    image_array = np.asarray(resized_image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    if st.button("Predict breed"):
        with st.spinner("Analyzing image..."):
            try:
                predictions = model.predict(
                    image_array, verbose=0
                )[0]

                if len(predictions) != len(class_names):
                    st.error(
                        "Prediction count does not match breed labels."
                    )
                    st.stop()

                best_index = int(np.argmax(predictions))
                breed = class_names[best_index]
                confidence = float(predictions[best_index])

                st.success(
                    f"Predicted breed: {breed.replace('_', ' ').title()}"
                )
                st.write(
                    f"Model score: {confidence:.2%}"
                )

                st.subheader("Top 5 predictions")
                top_indices = np.argsort(predictions)[-5:][::-1]

                for index in top_indices:
                    st.write(
                        f"{class_names[index].replace('_', ' ').title()}: "
                        f"{float(predictions[index]):.2%}"
                    )

            except Exception as e:
                st.error("Prediction failed.")
                st.exception(e)
