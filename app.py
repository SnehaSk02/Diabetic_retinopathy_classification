import streamlit as st
import numpy as np
import cv2

from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
from src.grad_cam import (make_gradcam_heatmap,overlay_heatmap)

from src.preprocessing import crop_black
from src.preprocessing import apply_clahe


# ====================================================
# CONFIG
# ====================================================

IMG_SIZE = 224

CLASS_NAMES = {

    0: "Mild",

    1: "Moderate",

    2: "No DR",

    3: "Proliferative DR",

    4: "Severe"

}


# ====================================================
# LOAD MODEL
# ====================================================

@st.cache_resource
def load_dr_model():

    model = load_model("src/models/best_model.h5")

    return model


model = load_dr_model()


# ====================================================
# PREPROCESS
# ====================================================

def preprocess_image(image):

    image = np.array(image)

    image = crop_black(image)

    image = apply_clahe(image)

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = preprocess_input(image)

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ====================================================
# UI
# ====================================================

st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    layout="centered"
)

st.title("👁️ Diabetic Retinopathy Severity Detection")

st.write(
    """
    Upload a retinal fundus image.
    
    """
)


uploaded_file = st.file_uploader(

    "Upload Fundus Image",

    type=["jpg", "jpeg", "png"]

)


# ====================================================
# PREDICTION
# ====================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    image = image.convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )
    descriptions = {

    0: "Early signs of diabetic retinopathy detected.",

    1: "Moderate diabetic retinopathy detected.",

    2: "No diabetic retinopathy detected.",

    3: "Advanced proliferative diabetic retinopathy detected.",

    4: "Severe diabetic retinopathy detected."
}
    
    if st.button("Predict"):

        with st.spinner("Analyzing Image..."):

            processed = preprocess_image(image)
            print("Processed Shape:", processed.shape)
            print("Processed Type:", type(processed))

            prediction = model.predict(
                processed,
                verbose=0
            )

            heatmap = make_gradcam_heatmap(processed,model)

            original_img = np.array(image)

            gradcam_img = overlay_heatmap(
                heatmap,
                original_img
            )

            pred_class = np.argmax(
                prediction
            )

            confidence = np.max(
                prediction
            ) * 100

            top2 = np.argsort(prediction[0])[-2:][::-1]

        if confidence < 60:
            st.warning(
        "Low confidence prediction. "
        "Please consult a specialist.")

        st.write(f"Most likely: {CLASS_NAMES[top2[0]]}")

        st.write(f"Second most likely: {CLASS_NAMES[top2[1]]}")

        # st.success(
        #     f"Prediction: {CLASS_NAMES[pred_class]}"
        # )

        # st.info(
        #     f"Confidence: {confidence:.2f}%"
        # )
        st.subheader("Report")

        st.write(
        descriptions[pred_class]
    )

        st.subheader(
            "Class Probabilities"
        )

        for i in range(5):

            st.write(
                f"{CLASS_NAMES[i]} : "
                f"{prediction[0][i]*100:.2f}%"
            )

        st.subheader("Grad-CAM Visualization")

        st.image(
            gradcam_img,
            caption="Areas influencing prediction",
            use_container_width=True
        )



