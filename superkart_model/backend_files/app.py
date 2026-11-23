import joblib
import pandas as pd
from flask import Flask, request, jsonify

superkart_api = Flask("SuperKart Sales Predictor")
model = joblib.load(saved_model_path)

FEATURES = ["Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
            "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Id_char",
            "Store_Age_Years", "Product_Type_Category"]

@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart Sales Prediction API!"

@superkart_api.post("/v1/predict")
def predict():
    data = request.get_json()
    df = pd.DataFrame([{c: data[c] for c in FEATURES}])
    return jsonify({"Predicted_Sales": round(float(model.predict(df)[0]), 2)})

@superkart_api.post("/v1/predictbatch")
def predict_batch():
    df = pd.read_csv(request.files["file"])[FEATURES]
    preds = [round(float(p), 2) for p in model.predict(df)]
    return jsonify(dict(zip(df.index.astype(str), preds)))

if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860)
