# 📦 Pre-trained Models : The trained machine learning and deep learning models used in **IntelliGluco** are **not included** in this GitHub repository because they exceed GitHub's file size limit.

## 📥 Download Models : Download all required models from the link below:
🔗 **Google Drive:**  
https://drive.google.com/drive/folders/14akkZIKXshv1O8Rg26TaEbkaEilLKLep?usp=sharing

## 📁 Required Files : After downloading, place the following files inside the `models/` folder.

```
models/
├── dia1lk_scaler(1).joblib
├── diabetes_pr1lk_model.joblib
└── Diabetic_Retinopathy_99.keras
```

## 📂 Project Structure :
Your project should look like this after downloading the models:

```
TRCS-102_IntelliGluco/
│
├── models/
   ├── dia1lk_scaler(1).joblib
   ├── diabetes_pr1lk_model.joblib
   └── Diabetic_Retinopathy_99.kera
```

---
## ⚠️ Why are the models not included? 
GitHub limits the size of files that can be stored in a repository. Since the trained model files are large, they are hosted separately to:

- Keep the repository lightweight and faster to clone.
- Avoid GitHub file size restrictions.
- Allow users to download the latest trained models independently.

---

## 🚀 Setup :
1. Clone this repository.
2. Download the model files using the link above.
3. Extract the downloaded archive (if applicable).
4. Copy all model files into the `models/` directory.
5. Install the project dependencies.
6. Run the application.

```
pip install -r requirements.txt
streamlit run main.py
```

---
If you encounter any issues while downloading or placing the models, please open an issue in this repository.
