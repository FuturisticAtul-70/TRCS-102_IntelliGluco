from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["DiabetesPrediction"]

collection = db["Predictions"]
retinopathy_collection = db["RetinopathyPredictions"]

try:
    client.admin.command("ping")
    print("✅ Connected to MongoDB Atlas")
except Exception as e:
    print("❌ MongoDB Connection Error")
    print(e)


def save_prediction(
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
    diabetic_probability
):

    document = {
        "Date": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "Gender": gender,
        "Age": int(age),
        "Hypertension": int(hypertension),
        "Heart Disease": int(heart_disease),
        "Smoking History": smoking_history,
        "BMI": float(bmi),
        "HbA1c": float(hba1c),
        "Blood Glucose": int(blood_glucose),
        "High Glucose": int(high_glucose),
        "Obese": int(obese),
        "BMI × Glucose": float(bmi_glucose),
        "Prediction": prediction,
        "Diabetic Probability": round(float(diabetic_probability), 2)
    }

    result = collection.insert_one(document)
    print(f"✅ Prediction Saved : {result.inserted_id}")


def get_predictions():

    records = list(collection.find().sort("_id", -1))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df.drop(columns=["_id"], errors="ignore", inplace=True)

    return df


def clear_history():
    collection.delete_many({})


def total_predictions():
    return collection.count_documents({})
def diabetic_count():
    return collection.count_documents({"Prediction": "Diabetic"})


def non_diabetic_count():
    return collection.count_documents({"Prediction": "Non-Diabetic"})


def save_retinopathy_prediction(
    image_name,
    prediction,
    stage,
    confidence,
    no_dr,
    mild,
    moderate,
    severe,
    proliferative
):

    document = {
        "Date": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "Image": image_name,
        "Prediction": prediction,
        "Stage": stage,
        "Confidence (%)": round(float(confidence), 2),
        "No_DR (%)": round(float(no_dr) * 100, 2),
        "Mild (%)": round(float(mild) * 100, 2),
        "Moderate (%)": round(float(moderate) * 100, 2),
        "Severe (%)": round(float(severe) * 100, 2),
        "Proliferative_DR (%)": round(float(proliferative) * 100, 2)
    }

    result = retinopathy_collection.insert_one(document)
    print(f"✅ Retinopathy Prediction Saved : {result.inserted_id}")


def get_retinopathy_predictions():

    records = list(retinopathy_collection.find().sort("_id", -1))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df.drop(columns=["_id"], errors="ignore", inplace=True)

    return df


def clear_retinopathy_history():
    retinopathy_collection.delete_many({})

def total_retinopathy_predictions():
    return retinopathy_collection.count_documents({})


def diabetic_retinopathy_count():
    return retinopathy_collection.count_documents(
        {"Prediction": "Diabetic"}
    )


def non_retinopathy_count():
    return retinopathy_collection.count_documents(
        {"Prediction": "Non-Diabetic"}
    )