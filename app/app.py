import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Setup model path
base_dir = os.path.dirname(os.path.abspath(__file__))
cnn_model_path = os.path.join(base_dir, '..', 'models', 'cnn_model.h5')

# Load CNN model
try:
    cnn_model = load_model(cnn_model_path)
except Exception as e:
    st.error(f"Error loading CNN model: {e}")
    st.stop()

# Streamlit UI
st.title("Pneumonia Detection Using Chest X-Ray")

st.markdown("""
    **Upload one or more chest X-ray images**, and this app will analyze each image to determine if it shows signs of **Pneumonia** or if it is **Normal**.  
    Results are based on a deep learning model trained using CNN.
""")

# Image preprocessing for CNN
def preprocess_image(image):
    try:
        # Convert to grayscale if needed
        if image.mode == "RGB":
            image = image.convert("L")

        # Resize and normalize
        image = image.resize((150, 150))
        img_array = np.array(image) / 255.0
        img_cnn = img_array.reshape(1, 150, 150, 1)
        return img_cnn
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None

# Upload and display images
uploaded_files = st.file_uploader("Upload X-ray Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    images = []
    for file in uploaded_files:
        try:
            img = Image.open(file)
            images.append(img)
        except Exception as e:
            st.error(f"Error loading image {file.name}: {e}")

    # Display preview
    st.subheader("Preview Images")
    cols = st.columns(min(len(images), 4))
    for i, img in enumerate(images):
        with cols[i % len(cols)]:
            st.image(img, caption=f"Image {i + 1}", use_column_width=True)

    # Predict
    if st.button("Classify Images"):
        st.subheader("Prediction Results")
        for i, img in enumerate(images):
            st.write(f"**Image {i + 1}**")
            img_cnn = preprocess_image(img)
            if img_cnn is not None:
                pred = cnn_model.predict(img_cnn)
                label = "Pneumonia" if pred[0][0] < 0.5 else "Normal"
                confidence = pred[0][0] if label == "Normal" else 1 - pred[0][0]
                st.write(f"- **Prediction:** {label}")
                st.write(f"- **Confidence:** {confidence:.2f}")
            else:
                st.write("Error in processing the image.")
else:
    st.info("Please upload images to get predictions.")

