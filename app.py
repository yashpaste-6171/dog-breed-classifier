
import json
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from huggingface_hub import hf_hub_download

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PawVision AI | Dog Breed Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --forest: #244B38;
    --forest-dark: #172B24;
    --cream: #F7F6F0;
    --muted: #6C786F;
    --lime: #D9E9B8;
}

.stApp {
    background: var(--cream);
    color: var(--forest-dark);
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--forest-dark) !important;
    letter-spacing: -0.8px;
}

.brand {
    font-family: 'Manrope', sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
    color: var(--forest);
}

.brand span { color: #809B55; }

.eyebrow {
    color: #647A56;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-title {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(2.5rem, 5vw, 4.4rem);
    font-weight: 800;
    letter-spacing: -3px;
    line-height: 1.08;
    color: var(--forest-dark);
    margin: 12px 0;
}

.hero-title span { color: #718C4C; }

.hero-copy {
    color: var(--muted);
    font-size: 1.08rem;
    line-height: 1.8;
    max-width: 620px;
}

.hero-card {
    background: var(--forest-dark);
    color: white;
    padding: 28px;
    border-radius: 24px;
    min-height: 205px;
}

.hero-card h3 {
    color: white !important;
    margin: 8px 0;
}

.hero-card p { color: #D0DCD2; }

.pill {
    display: inline-block;
    background: #E6EDD9;
    color: #35533C;
    padding: 7px 12px;
    border-radius: 30px;
    font-size: 0.78rem;
    font-weight: 700;
    margin: 4px 5px 4px 0;
}

.section-title {
    font-family: 'Manrope', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--forest-dark);
    margin-bottom: 5px;
}

.section-copy {
    color: var(--muted);
    margin-bottom: 20px;
}

div[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #BCCAB5;
    border-radius: 20px;
    padding: 20px;
}

div[data-testid="stFileUploader"] section {
    background: transparent;
}

div[data-testid="stButton"] > button {
    background: var(--forest);
    color: white;
    border: none;
    border-radius: 12px;
    min-height: 3.2rem;
    font-weight: 700;
    transition: all 0.2s ease;
}

div[data-testid="stButton"] > button:hover {
    background: #365E46;
    color: white;
    border: none;
    transform: translateY(-1px);
}

.result-card {
    background: white;
    border: 1px solid #E5E8DE;
    border-radius: 20px;
    padding: 24px;
    margin: 12px 0;
}

.result-label {
    color: #6C786F;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.result-breed {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(1.8rem, 3vw, 2.5rem);
    font-weight: 800;
    color: var(--forest);
    margin: 8px 0;
}

.score {
    font-family: 'Manrope', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #6F8A48;
}

.footer {
    text-align: center;
    color: #7A857C;
    font-size: 0.85rem;
    padding-top: 35px;
}

#MainMenu, footer {visibility: hidden;}

@media (max-width: 700px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .hero-title { letter-spacing: -1.5px; }
}
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL SETTINGS ----------------
REPO_ID = "yashog19/dog-breed-classifier-model"
MODEL_NAME = "best_dog_breed_model.keras"
CLASSES_PATH = "dog_breed_classes.json"
IMG_SIZE = (224, 224)


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_NAME,
        repo_type="model",
    )
    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_classes():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------- HEADER ----------------


left, right = st.columns([1.5, 1], gap="large")

with left:
    st.markdown(
        '<div class="eyebrow">YOUR SMART DOG BREED IDENTIFIER</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-title">Every dog has a story.<br>'
        'Discover their <span>breed.</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="hero-copy">
        Meet your AI-powered companion for discovering dog breeds.
        Upload a photo and explore the model's top breed predictions
        in seconds.
    </div>
    <div style="margin-top:18px">
        <span class="pill">🐕 120 breeds</span>
        <span class="pill">⚡ AI-powered</span>
        <span class="pill">📷 Simple photo upload</span>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="hero-card">
        <div style="font-size:2.5rem">🐕‍🦺</div>
        <div style="color:#B9D8A9;font-size:.75rem;
        font-weight:700;letter-spacing:2px">MEET YOUR AI COMPANION</div>
        <h3>One photo.<br>New discoveries.</h3>
        <p>Let your dog's photo reveal the model's best breed match.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# ---------------- UPLOAD SECTION ----------------
st.markdown(
    '<div class="section-title">Let’s meet your dog</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-copy">Upload a clear photo for the best results. '
    'JPG, JPEG or PNG supported.</div>',
    unsafe_allow_html=True,
)

upload_col, preview_col = st.columns([1, 1], gap="large")

with upload_col:
    uploaded_file = st.file_uploader(
        "Choose your dog's photo",
        type=["jpg", "jpeg", "png"],
        help="A clear photo with the dog visible works best.",
    )

    st.caption("Tip: a well-lit photo with the dog's face visible may help.")

with preview_col:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Your uploaded photo",
            use_container_width=True,
        )
    else:
        st.markdown("""
        <div style="background:#E9EDE2;border:1px solid #DDE4D5;
        border-radius:20px;padding:35px 20px;text-align:center;
        min-height:180px">
            <div style="font-size:3rem">📸</div>
            <div style="font-weight:700;color:#244B38;margin-top:10px">
                Your photo appears here
            </div>
            <div style="font-size:.9rem;color:#6C786F;margin-top:6px">
                Ready when you are.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
st.write("")

if uploaded_file is not None:
    if st.button("✨  Discover this dog's breed", use_container_width=True):
        try:
            with st.spinner("Our AI is examining the photo..."):
                model = load_model()
                class_names = load_classes()

                image = Image.open(uploaded_file).convert("RGB")
                resized_image = image.resize(IMG_SIZE)
                image_array = np.asarray(
                    resized_image, dtype=np.float32
                )
                image_array = np.expand_dims(image_array, axis=0)

                predictions = model.predict(image_array, verbose=0)[0]

                if len(predictions) != len(class_names):
                    st.error(
                        "The model output doesn't match the breed labels. "
                        "Please check the class-label file."
                    )
                    st.stop()

                best_index = int(np.argmax(predictions))
                best_breed = class_names[best_index]
                best_score = float(predictions[best_index])
                top_indices = np.argsort(predictions)[-5:][::-1]

            st.write("")
            st.markdown("---")
            st.markdown(
                '<div class="eyebrow">YOUR RESULTS ARE READY</div>',
                unsafe_allow_html=True,
            )

            result_left, result_right = st.columns([1.4, 1])

            with result_left:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">TOP BREED MATCH</div>
                    <div class="result-breed">
                        {best_breed.replace('_', ' ').title()}
                    </div>
                    <div style="color:#6C786F">
                        Highest-scoring breed from the model's predictions.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with result_right:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">MODEL SCORE</div>
                    <div class="score">{best_score:.2%}</div>
                    <div style="color:#6C786F;font-size:.85rem">
                        Prediction score, not guaranteed accuracy.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(
                '<div class="section-title">Top 5 breed matches</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-copy">Ranked by model output score.</div>',
                unsafe_allow_html=True,
            )

            for rank, index in enumerate(top_indices, start=1):
                breed_name = class_names[index].replace('_', ' ').title()
                score = float(predictions[index])

                label_col, bar_col, percent_col = st.columns([1.4, 2.5, 0.7])

                with label_col:
                    st.write(f"**{rank}. {breed_name}**")
                with bar_col:
                    st.progress(
                        max(0.0, min(1.0, score)),
                        text=None,
                    )
                with percent_col:
                    st.write(f"{score:.2%}")

        except Exception as e:
            st.error("We couldn't complete the prediction.")
            st.exception(e)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    🐾 PawVision AI · Built with TensorFlow & Streamlit<br>
    For educational use. Predictions can be incorrect.
</div>
""", unsafe_allow_html=True)
