import streamlit as st
import pandas as pd
from retinopathy import show_retinopathy_page
from database import (
    get_retinopathy_predictions,
    save_prediction,
    get_predictions,
    clear_history,
    total_predictions,
    diabetic_count,
    non_diabetic_count
)
st.sidebar.markdown("### 🌐 Connect with Me")
st.sidebar.link_button(
    "🐙 GitHub",
    "https://github.com/FuturisticAtul-70/TRCS-102_IntelliGluco"
)
st.sidebar.link_button(
    "💼 LinkedIn",
    "https://www.linkedin.com/in/atul-pandey-832820318?utm_source=share_via&utm_content=profile&utm_medium=member_android"
)
st.sidebar.markdown("### 📘 Colab Notebook")
st.sidebar.link_button(
    "📓 Open Colab Notebook",
    "https://colab.research.google.com/drive/1aVsjJwNS-HLNQwkHWE4PePvxjCo6jjn3?usp=sharing"
)
import joblib
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
h1, h2, h3, h4, h5, h6,
p,
label,
li,
span {
    font-family: 'Poppins', sans-serif;
}
.toast {
    position: fixed;
    top: 30%;
    left: 50%;
    transform: translate(-50%, -50%);
    min-width: 200px;
    padding: 18px 24px;
    border-radius: 12px;
    color: white;
    text-align: center;
    font-size: 18px;
    font-weight: 600;
    z-index: 9999;
    animation: fadeMove 1.5s ease-in-out forwards;
    box-shadow: 0 8px 25px rgba(0,0,0,0.35);
}
.toast.success{
    background: linear-gradient(135deg,#16a34a,#22c55e);
}
.toast.warning{
    background: linear-gradient(135deg,#dc2626,#ef4444);
}
@keyframes fadeMove{
    0%{
        opacity:0;
        transform:translate(-50%,-50%);
    }
    20%{
        opacity:0.75;
        transform:translate(-50%,-50%);
    }
    80%{
        opacity:1;
        transform:translate(-50%,-170%);
    }
    100%{
        opacity:0;
        transform:translate(-50%,-220%);
    }
</style>
""", unsafe_allow_html=True)
def show_popup(message, popup_type="success"):
    st.markdown(
        f"""
        <div class="toast {popup_type}">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )
smoking_labels = {
    0: "No Info",
    1: "Current",
    2: "Ever",
    3: "Former",
    4: "Never",
    5: "Not Current"
}
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)
st.markdown("""
<style>
.stApp{
background:linear-gradient(135deg,#071426,#0b1f3a,#12376a);
color:#f8fafc;
}
.stButton>button{
width:100%;
height:54px;
background:linear-gradient(90deg,#0ea5e9,#2563eb);
color:#fff;
font-size:18px;
font-weight:700;
border:2px solid #38bdf8;
border-radius:16px;
transition:.3s;
box-shadow:0 0 18px rgba(56,189,248,.35);
}
.stButton>button:hover{
background:linear-gradient(90deg,#2563eb,#38bdf8);
border:2px solid #7dd3fc;
transform:translateY(-2px);
box-shadow:0 0 25px rgba(56,189,248,.6);
}
.stExpander{
border:2px solid #3b82f6!important;
border-radius:14px!important;
background:#102544!important;
box-shadow:0 0 12px rgba(59,130,246,.2);
}
[data-testid="stExpander"]{
border:2px solid #3b82f6!important;
border-radius:14px!important;
background:#102544!important;
}
[data-testid="stExpander"] details{
border-radius:14px;
}
[data-testid="stExpander"] summary{
font-size:18px;
font-weight:700;
color:#f8fafc!important;
}
.stNumberInput input,.stSelectbox div[data-baseweb="select"]{
background:#17345f!important;
border:2px solid #38bdf8!important;
border-radius:12px!important;
color:#fff!important;
font-weight:600;
}
.stNumberInput input:focus,.stSelectbox div[data-baseweb="select"]:focus-within{
border:2px solid #7dd3fc!important;
box-shadow:0 0 15px rgba(56,189,248,.5)!important;
}
label{
color:#f8fafc!important;
font-size:17px!important;
font-weight:700!important;
}
.title{
text-align:center;
font-size:42px;
font-weight:800;
color:#7dd3fc;
text-shadow:0 0 18px rgba(125,211,252,.4);
}
.subtitle{
text-align:center;
font-size:18px;
font-weight:500;
color:#cbd5e1;
}
.card{
background:rgba(16,37,68,.95);
border:2px solid #2563eb;
border-radius:18px;
padding:25px;
box-shadow:0 0 25px rgba(37,99,235,.25);
}
hr{
border:1px solid #2563eb;
}
</style>
""",unsafe_allow_html=True)
model = joblib.load("models/diabetes_pr1lk_model.joblib")
scaler = joblib.load("models/dia1lk_scaler(1).joblib")
st.markdown("""
<div class="card">
<h1 style="text-align:center;">
🧪 IntelliGluco
</h1>
<p style="text-align:center;
color:#cbd5e1;
font-size:20px;">
ML based diabetes prediction system
</p>
</div>""", unsafe_allow_html=True)
if "reset" not in st.session_state:
    st.session_state.reset = False
def get_diabetic_risk(prob):
    if prob >= 0.90:
        return (
            "🔴 Very High Diabetes Risk",
            [
                "Seek medical evaluation by an endocrinologist or physician immediately.",
                "Undergo confirmatory tests such as HbA1c, Fasting Blood Glucose (FBG), and Oral Glucose Tolerance Test (OGTT) if advised.",
                "Begin strict dietary modifications by reducing sugar and refined carbohydrates.",
                "Monitor blood glucose regularly as recommended by your healthcare provider.",
                "Maintain a healthy body weight and follow a structured physical activity plan.",
                "Control blood pressure and other cardiovascular risk factors.",
                "Avoid smoking and excessive alcohol consumption."
            ]
        )
    elif prob >= 0.75:
        return (
            "🟠 High Diabetes Risk",
            [
                "Consult a physician for further evaluation.",
                "Get HbA1c and fasting blood glucose tests performed.",
                "Adopt a balanced diet rich in vegetables, whole grains, and lean proteins.",
                "Exercise for at least 150 minutes per week unless medically contraindicated.",
                "Monitor body weight and blood pressure regularly.",
                "Reduce intake of sugary beverages and processed foods."
            ]
        )
    elif prob >= 0.60:
        return (
            "🟡 Moderate Diabetes Risk",
            [
                "Maintain a healthy lifestyle to reduce future diabetes risk.",
                "Increase daily physical activity.",
                "Limit foods high in added sugars and refined carbohydrates.",
                "Maintain a healthy BMI.",
                "Consider routine blood glucose screening during regular health checkups."
            ]
        )
    else:
        return (
            "🟢 Mild Diabetes Risk",
            [
                "Continue following healthy eating habits.",
                "Exercise regularly.",
                "Maintain a healthy body weight.",
                "Schedule routine preventive health checkups.",
                "Seek medical advice if symptoms such as excessive thirst, frequent urination, or unexplained weight loss occur."
            ]
        )
def get_nondiabetic_message():
    return (
        "✅ No Diabetes Predicted",
        [
            "Continue maintaining a healthy lifestyle.",
            "Exercise regularly and maintain a healthy body weight.",
            "Follow a balanced diet with limited added sugars.",
            "Have routine health checkups and diabetes screening as recommended for your age and risk factors."
        ]
    )
def load_sample_data():
    st.session_state.update({
        "gender": 1,                
        "age": 52,
        "hypertension": 1,          
        "heart_disease": 0,          
        "smoking_history": 3,        
        "bmi": 31.5,
        "hba1c": 7.2,
        "blood_glucose": 185,
        "obese": 1,               
        "high_glucose": 1,          
        "bmi_glucose": 5827.5       
    }) 
def calculate_bmi_glucose():
    st.session_state.bmi_glucose = (
        st.session_state.bmi *
        st.session_state.blood_glucose
    )
if "page" not in st.session_state:
    st.session_state.page = "🧪 Diabetes Prediction"
page = st.segmented_control(
    "",
    ["🧪 Diabetes Prediction", "ℹ️ About",  "📊 Prediction History",  "👁️ Diabetic Retinopathy"],
    key="page_selector"
)
if page != st.session_state.page:
    st.session_state.page = page
if page == "🧪 Diabetes Prediction":
    with st.expander("📝 Enter the Patient Details below", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            with st.expander("Gender"):
                gender = st.selectbox(
                    "Gender",
                    [0, 1],
                    format_func=lambda x: "Female" if x == 0 else "Male",
                    key="gender",
                    label_visibility="collapsed"
                )
            with st.expander("Age"):
                age = st.number_input(
                    "Age",
                    value=30,
                    key="age",
                    label_visibility="collapsed"
                )
            with st.expander("Hypertension"):
                hypertension = st.selectbox(
                    "Hypertension",
                    [0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes",
                    key="hypertension",
                    label_visibility="collapsed"
                )
        with col2:
            with st.expander("Heart Disease"):
                heart_disease = st.selectbox(
                    "Heart Disease",
                    [0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes",
                    key="heart_disease",
                    label_visibility="collapsed"
                )
            with st.expander("Smoking History"):
                smoking_history = st.selectbox(
                    "Smoking History",
                    options=list(smoking_labels.keys()),
                    format_func=lambda x: f"{x}  - {smoking_labels[x]}",
                    key="smoking_history",
                    label_visibility="collapsed"
                )
            with st.expander("BMI"):
                bmi = st.number_input(
                    "BMI",
                    value=25.0,
                    key="bmi",
                    label_visibility="collapsed"
                )
        with col3:
            with st.expander("HbA1c Level"):
                hba1c = st.number_input(
                    "HbA1c Level",
                    value=5.5,
                    key="hba1c",
                    label_visibility="collapsed"
                )
            with st.expander("Blood Glucose Level"):
                blood_glucose = st.number_input(
                    "Blood Glucose Level",
                    value=100,
                    key="blood_glucose",
                    label_visibility="collapsed"
                )
            with st.expander("Obese"):
                obese = st.selectbox(
                    "Obese",
                    [0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes",
                    key="obese",
                    label_visibility="collapsed"
                )
        with col4:
            with st.expander("High Glucose"):
                high_glucose = st.selectbox(
                    "High Glucose",
                    options=[0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes",
                    key="high_glucose",
                    label_visibility="collapsed"
        )
            with st.expander("BMI × Glucose"):
                bmi_glucose = st.number_input(
                    "BMI_Glucose",
                    value=0.0,
                    key="bmi_glucose",
                    label_visibility="collapsed"
                )
                st.caption("Click below to calculate BMI × Blood Glucose")
                st.button(
                    "🧮 Calculate",
                    on_click=calculate_bmi_glucose
                )
        col1,col2 = st.columns(2)
        with col1:
            sample = st.button(
            "🧪 Load Sample Data",
            on_click=load_sample_data,
            use_container_width=True
        )
        if sample:
            show_popup("✅ Sample patient data loaded successfully!", "success")

        with col2:
            predict = st.button(
                "🔍 Predict Result",
                use_container_width=True
            )
        if predict:
                errors = []
                if age < 1 or age > 120:
                    errors.append("❌ Age must be between 1 and 120 years.")
                if bmi < 10 or bmi > 100:
                    errors.append("❌ BMI must be between 10 and 60 kg/m².")
                if hba1c < 3.0 or hba1c > 15.0:
                    errors.append("❌ HbA1c Level must be between 3.0% and 15.0%.")
                if blood_glucose < 50 or blood_glucose > 800:
                    errors.append("❌ Blood Glucose Level must be between 50 and 300 mg/dL.")
                if bmi_glucose < 500 or bmi_glucose > 18000:
                    errors.append("❌ BMI × Glucose value is outside the valid range.")
                if errors:
                    for error in errors:
                        st.error(error)
                    st.stop()
                fields = {
                    "Age": age,
                    "BMI": bmi,
                    "HbA1c Level": hba1c,
                    "Blood Glucose Level": blood_glucose,
                    "BMI_Glucose": bmi_glucose
                }
                missing_fields = []
                for field_name, value in fields.items():
                    if value <= 0:
                        missing_fields.append(field_name)
                if missing_fields:
                    st.error(
                        "⚠ Please fill the following fields before prediction:\n\n- " +
                        "\n- ".join(missing_fields)
                    )
                    st.stop()
                input_data = pd.DataFrame({
                    "gender":[gender],
                    "age":[age],
                    "hypertension":[hypertension],
                    "heart_disease":[heart_disease],
                    "smoking_history":[smoking_history],
                    "bmi":[bmi],
                    "hbA1c_level":[hba1c],
                    "blood_glucose_level":[blood_glucose],
                    "obese":[obese],
                    "high_glucose":[high_glucose],
                    "BMI_Glucose":[bmi_glucose]
                })
                
                st.markdown(f"""
                <div style="
                    background-color:#1f2937;
                    padding:18px 22px;
                    border-radius:12px;
                    border:1px solid #3b82f6;
                    margin-top:20px;
                    width:420px;
                ">
                    <h5 style="margin-top:0; color:white;">📋 Patient Input Summary</h5>
                    <p style="margin:4px 0;"><b>Age:</b> {age}</p>
                    <p style="margin:1px 0;"><b>Gender:</b> {"Male" if gender == 1 else "Female"}</p>
                    <p style="margin:1px 0;"><b>Hypertension:</b> {"Yes" if hypertension == 1 else "No"}</p>
                    <p style="margin:1px 0;"><b>Heart Disease:</b> {"Yes" if heart_disease == 1 else "No"}</p>
                    <p style="margin:1px 0;"><b>Smoking History:</b> {smoking_history} ({smoking_labels[smoking_history]})</p>
                    <p style="margin:1px 0;"><b>BMI:</b> {bmi:.2f}</p>
                    <p style="margin:1px 0;"><b>HbA1c Level:</b> {hba1c:.1f}</p>
                    <p style="margin:1px 0;"><b>Blood Glucose Level:</b> {blood_glucose}</p>
                    <p style="margin:1px 0;"><b>Obese:</b> {"Yes" if obese == 1 else "No"}</p>
                    <p style="margin:1px 0;"><b>High Glucose:</b> {"Yes" if high_glucose == 1 else "No"}</p>
                    <p style="margin:4px 0;"><b>BMI × Glucose:</b> {bmi_glucose:.2f}</p>
                </div>""", unsafe_allow_html=True)
                input_scaled = scaler.transform(input_data)
                prediction_n = int(model.predict(input_scaled)[0])
                probability = model.predict_proba(input_scaled)[0]
                prediction = "Diabetic" if (prediction_n) == 1 else "Non-Diabetic"
                diabetic = probability[1]
                cal_risk=diabetic*100
                st.markdown("---")
                st.header("🩺 Prediction Result")
                save_prediction(
    gender,                
    age,                    
    hypertension,            
    heart_disease,          
    smoking_history,         
    bmi,                    
    hba1c,                  
    blood_glucose,          
    high_glucose,           
    obese,                  
    bmi_glucose,           
    prediction,
    diabetic * 100
)
                if prediction_n == 1:
                    risk_level, precautions = get_diabetic_risk(diabetic)
                    st.subheader("Patient is Diabetic")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Estimated Diabetes Risk",
                            f"{diabetic*100:.2f}%"
                        )
                    with col2:
                        st.metric(
                            "Risk Level",
                            risk_level
                        )
                    st.progress(float(diabetic))
                    st.markdown("### 🩸 Recommended Tests")
                    st.info("""
                • HbA1c Test
                • Fasting Blood Sugar
                • Post-Prandial Blood Sugar
                """)
                    st.markdown("### 💡 Precautions")

                    for item in precautions:
                        st.write(f"✅ {item}")
                    st.warning("""
                **Medical Disclaimer**

                This result is generated using a Machine Learning model.

                It is **not a medical diagnosis**.
                Please consult a qualified healthcare professional for confirmatory testing.
                """)
                else:
                    risk_level, precautions = get_nondiabetic_message()
                    st.subheader("✅ Patient is Non-Diabetic")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Estimated Diabetic Risk",
                            f"{diabetic*100:.2f}%"
                        )
                    with col2:
                        st.metric(
                            "Risk Level",
                            risk_level
                        )
                    st.progress(float(diabetic))
                    st.markdown("### 🌿 Healthy Lifestyle Recommendations")
                    for item in precautions:
                        st.write(f"✅ {item}")
                    st.info("""
                Maintain a balanced diet, regular physical activity and periodic health checkups to reduce future diabetes risk.
                """)           
if page == "ℹ️ About":
    st.markdown("""
<p style='font-size:16px; text-align:justify; line-height:1.8; margin-bottom:25px;'>
<b>IntelliGluco</b> is a web-based healthcare application that integrates two Machine Learning models to assist in diabetes screening. The first model predicts the likelihood of diabetes using patient health parameters, while the second analyzes uploaded retinal fundus images to detect and classify the severity of diabetic retinopathy. Together, these modules provide a comprehensive platform for diabetes risk assessment and retinal disease detection.
</p>
""", unsafe_allow_html=True)
    st.markdown("""
This application consists of two Machine Learning modules:<br><br>
<b>🧪 Diabetes Prediction</b> – Predicts the likelihood of diabetes using patient demographic information, medical history, laboratory test values, and engineered features.<br><br>
<b>👁️ Diabetic Retinopathy Detection</b> – Classifies retinal fundus images into different stages of diabetic retinopathy using a deep learning model based on the EfficientNet architecture.<br><br>
""", unsafe_allow_html=True)
    with st.expander("📋 Input Parameters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""### 🩺 Diabetes Prediction
            
**Gender:** Patient's biological gender (Male/Female).
**Age:** Age of the patient in completed years.
**Hypertension:** Indicates whether the patient has high blood pressure.
- **0:** No
- **1:** Yes

**Heart Disease:** Indicates whether the patient has a diagnosed heart disease.
- **0:** No
- **1:** Yes

**Smoking History**
- **0:** No Information
- **1:** Current Smoker
- **2:** Ever Smoked
- **3:** Former Smoker
- **4:** Never Smoked
- **5:** Not Currently Smoking

**BMI (Body Mass Index):** Calculated using height and weight.
**HbA1c Level:** Average blood glucose over the previous 2–3 months.
**Blood Glucose Level:** Current blood glucose concentration measured in mg/dL.
**Obese:** Engineered feature.
- BMI ≥ 30 → 1
- BMI < 30 → 0

**High Glucose:** Engineered feature.
- Blood Glucose ≥ 140 mg/dL → 1
- Otherwise → 0

**BMI × Glucose:** Feature created by multiplying BMI and Blood Glucose to improve model performance.
""")
        with col2:
            st.markdown("### 👁️ Diabetic Retinopathy Detection")
            st.markdown("""
**Fundus Image:** A color retinal photograph captured using a fundus camera.

**Accepted Formats**
- JPG
- JPEG
- PNG

**Image Preprocessing**
- Converted to RGB
- Resized to **300 × 300 pixels**
- Pixel normalization using **EfficientNet preprocessing**

**Prediction Classes**
- No Diabetic Retinopathy
- Mild
- Moderate
- Severe
- Proliferative Diabetic Retinopathy (PDR)

**Prediction Confidence**
The model provides a confidence score for every class and reports the class with the highest probability.

**Second Prediction**
The second highest probability class is also displayed to assist in borderline cases where two stages have similar confidence.

**Prediction History**
Every analyzed retinal image is automatically stored in MongoDB together with:

- Image Name
- Predicted Stage
- Disease Status
- Confidence Score
- Probability of every DR class
""")
    with st.expander("📊 How the Prediction is Generated", expanded=False):
            col1, col2= st.columns(2)
            with col1:
                st.markdown(""" ### 🧪 Diabetes Prediction

The model analyzes all the above health parameters using a trained **Random Forest Classifier**.

It provides:
- **Prediction:** Diabetic or Non-Diabetic
- **Estimated Probability of Diabetes**
- **Risk Level:** Low, Moderate, High or Very High
""")
            with col2:
                st.markdown("""
### 👁️ Retinopathy Detection

The uploaded retinal image undergoes:
• Image resizing
• EfficientNet preprocessing

The deep learning model extracts retinal features automatically and classifies the image into one of five diabetic retinopathy stages.
The application also displays:
• Predicted Stage
• Confidence Score
• Second Most Probable Stage
• Probability for every class
""")
    with st.expander("🚀 Features", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""### 🧪 Diabetes Prediction
- 🤖 Random Forest Machine Learning Model
- 📊 Estimated Diabetes Probability
- ⚡ Real-Time Prediction
- 🎯 Automatic Risk Classification
- 🧠 Automatic Feature Engineering
- 📝 Interactive & User-Friendly Dashboard
- ✅ Input Validation
- 👤 Sample Patient Data Loader
- 📈 Instant Prediction Results
""")
            with col2:
                st.markdown("""
### 👁️ Retinopathy Detection

- EfficientNetB3 Deep Learning Model
- Five-Class Classification
- Retinal Image Upload
- Class-wise Probability
- MongoDB Prediction History
- Real-time Retina Analysis with Confidence Score
""")
    with st.expander("🤖 Machine Learning Model", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""### 🧪 Diabetes Prediction

- **Algorithm:** Random Forest Classifier
- **Training Dataset:** ~100,000 patient healthcare records
- **Preprocessing:** Data Cleaning, Feature Engineering, Feature Scaling & Hyperparameter Tuning
- **Outputs:**
    - Diabetes Prediction
    - Estimated Probability
    - Risk Category
""")
            with col2:

                st.markdown("""
### 👁️ Diabetic Retinopathy Detection

**Architecture**
EfficientNetB3

**Transfer Learning**
ImageNet Pretrained Weights

**Input Size**
300 × 300 RGB

**Output Classes**
- No_DR
- Mild
- Moderate
- Severe
- Proliferative_DR

**Output**
Disease Stage
Prediction Confidence
Class Probabilities
""")
    with st.expander("🛠 Technologies Used", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""### 🧪 Diabetes Prediction
- 🐍 Python
- 🎈 Streamlit
- 🤖 Scikit-learn
- 🐼 Pandas
- 🔢 NumPy
- 💾 Joblib
""")
            with col2:

                st.markdown("""
### 👁️ Retinopathy Detection

- Python
- TensorFlow
- Keras
- EfficientNet
- Pillow
- NumPy
- MongoDB
- Streamlit
""")
    with st.expander("📌 Interpretation of Results", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""### 🧪 Diabetes Prediction

- **Prediction:** Indicates whether the patient is classified as **Diabetic** or **Non-Diabetic**.
- **Estimated Probability:** Represents the model's confidence that the patient's profile belongs to the diabetic class.
- **Risk Level:** Categorizes the probability into **Low, Moderate, High,** or **Very High** for easier understanding.
""")
            with col2:
                st.markdown("""
### 👁️ Retinopathy Detection :

- **Predicted Stage:** One of the five diabetic retinopathy stages.
- **Confidence:** Likelihood assigned to the predicted class.
- **Class Probabilities:** Displays probabilities for every disease stage.
The model assists screening and should always be verified by an ophthalmologist.
""")
    st.markdown("""
    ### Disclaimer :
    This application is intended **for educational and research purposes only**. The predictions generated by this machine learning model **are not medical diagnoses** and should **not** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for any medical concerns or healthcare decisions.

    ### 👨‍💻 Author :
     **Atul Pandey**

     **GNE**
    
     **B.Tech CSE, U.R.N : 2434941**
    """)
    st.divider()
    def go_back():
        st.session_state.page = "🧪 Diabetes Prediction"
        st.session_state.page_selector = "🧪 Diabetes Prediction"
    st.button("⬅ Back", on_click=go_back)
if page=="📊 Prediction History":
    st.subheader("Prediction History for text input")
    history = get_predictions()
    if not history.empty:
        history["Diabetic Probability"] = history["Diabetic Probability"].map(
            lambda x: f"{x:.2f}%"
        )
        st.dataframe(history, use_container_width=True)
    else:
        st.info("No prediction history available.")
    st.divider()
    st.subheader("Prediction History for image input.")
    history = get_retinopathy_predictions()
    if isinstance(history, pd.DataFrame) and not history.empty:
            st.dataframe(
                history,
                use_container_width=True,
                hide_index=True
            )
    else:
            st.info("No prediction history available.")
if page == "👁️ Diabetic Retinopathy":
    show_retinopathy_page()
