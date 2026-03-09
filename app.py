import streamlit as st
from predict import predict, model
from disease_info import disease_info
from gradcam import generate_gradcam
from PIL import Image
import numpy as np
import torch
import cv2

st.title("🌿 AI Plant Disease Detector")

uploaded_file = st.file_uploader(
    "Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

# prediction
    result = predict(uploaded_file)

    st.success(f"Prediction: {result['prediction']} ({result['confidence']}%)")

    disease_name = result["prediction"]

    if disease_name in disease_info:

        st.subheader("Disease Information")

        st.write("Description:")
        st.write(disease_info[disease_name]["description"])

        st.write("Treatment Suggestion:")
        st.write(disease_info[disease_name]["treatment"])

# Grad-CAM
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).float()

    cam = generate_gradcam(model, img_tensor)

    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    overlay = heatmap * 0.4 + img_array * 255

    st.image(overlay.astype(np.uint8), caption="Disease Heatmap")
