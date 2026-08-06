import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from database import (
    save_retinopathy_prediction,
    get_retinopathy_predictions
)
@st.cache_resource
def load_dr_model():
    return load_model("models/Diabetic_Retinopathy_99.keras")
model = load_dr_model()
CLASS_NAMES = [
    "No_DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative_DR"
]
def show_retinopathy_page():
    st.markdown("## 👁️ Diabetic Retinopathy Detection")
    st.caption("Upload a retinal image to detect diabetic retinopathy.")
    st.divider()
    st.markdown("### 📤 Upload retinal Image")
    st.markdown("""
<style>
/* Hide the automatic image preview shown by file_uploader */
[data-testid="stFileUploader"] img {
    display: none !important;
}

/* Hide any image displayed inside the uploader */
[data-testid="stFileUploader"] [data-testid="stImage"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        help="Upload a Retinal Image in JPG, JPEG, or PNG format."
    )

    if uploaded_file is None:
        st.info("Please upload a retinal image.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.success(f"📷 Selected Image: {uploaded_file.name}")
    st.divider()

    if st.button(
        "🔍 Analyze Retina",
        use_container_width=True,
        type="primary"
    ):
        with st.spinner("Analyzing Retinal Image..."):
            img = image.resize((300, 300))
            img = img_to_array(img)
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img, verbose=0)[0]

        pred = np.argmax(prediction)
        stage = CLASS_NAMES[pred]
        confidence = float(prediction[pred] * 100)
        sorted_idx = np.argsort(prediction)[::-1]
        second_stage = CLASS_NAMES[sorted_idx[1]]
        second_confidence = float(prediction[sorted_idx[1]] * 100)
        status = "Non-Diabetic" if pred == 0 else "Diabetic"
        result = {
    "status": status,
    "stage": stage,
    "confidence": confidence,
    "second_stage": second_stage,
    "second_confidence": second_confidence,
    "prediction": prediction
}
        st.subheader("Prediction Result")
        col1, col2, col3 = st.columns(3)
        with col1:
            if result["status"] == "Non-Diabetic":
                st.success(result["status"])
            else:
                st.error(result["status"])

        with col2:
            st.info(result["stage"])

        with col3:
            st.warning(f'{result["confidence"]:.2f}%')

        st.info(
            f'Second Prediction: {result["second_stage"]} ({result["second_confidence"]:.2f}%)'
        )

        st.divider()

        st.subheader("📊 Class Probabilities")

        for cls, prob in zip(CLASS_NAMES, result["prediction"]):
            st.write(f"**{cls}** : {prob*100:.2f}%")
            st.progress(float(prob))
        save_retinopathy_prediction(
            uploaded_file.name,
            result["status"],
            result["stage"],
            round(result["confidence"], 2),
            float(result["prediction"][0]),
            float(result["prediction"][1]),
            float(result["prediction"][2]),
            float(result["prediction"][3]),
            float(result["prediction"][4])
        )
        st.divider()