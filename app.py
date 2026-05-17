import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Page settings
st.set_page_config(page_title="Skin Cancer Detection", layout="centered")

st.title("🩺 Skin Cancer Prediction App")
st.write("Upload a skin image to detect if it is Benign or Malignant")

# Load model
@st.cache_resource
def load_my_model():
    return load_model("skin_cancer_model.keras")

model = load_my_model()

# Upload image
uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=250)

    # Preprocess image
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict button
    if st.button("Predict"):

        prediction = model.predict(img)[0][0]

        if prediction > 0.5:
            st.error("⚠️ Result: Malignant (Cancer Detected)")
        else:
            st.success("✅ Result: Benign (Normal Skin)")

        st.write("Prediction Score:", round(float(prediction), 3))