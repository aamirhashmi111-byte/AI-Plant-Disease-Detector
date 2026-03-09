import streamlit as st
from predict import predict, model
from disease_info import disease_info
from gradcam import generate_gradcam
from PIL import Image
import numpy as np
import torch
import cv2

st.title("🌿 AI Plant Disease Detector")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")

    # Resize original image for display
    display_image = image.resize((300, 300))

    # Prediction
    result = predict(uploaded_file)
    disease_name = result["prediction"]

    # Prepare image for model
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).float()

    # Generate Grad-CAM
    cam = generate_gradcam(model, img_tensor)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = heatmap * 0.4 + img_array * 255

    # Resize heatmap image to match original display size
    overlay = cv2.resize(overlay, (300, 300))
    overlay = overlay.astype(np.uint8)

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.image(display_image, caption="Uploaded Leaf")

    with col2:
        st.image(overlay, caption="Disease Heatmap")

    # Prediction result
    st.success(f"Prediction: {result['prediction']} ({result['confidence']}%)")

    # Disease information
    if disease_name in disease_info:

        st.subheader("Disease Information")

        st.write("Description:")
        st.write(disease_info[disease_name]["description"])

        st.write("Treatment Suggestion:")
        st.write(disease_info[disease_name]["treatment"])
