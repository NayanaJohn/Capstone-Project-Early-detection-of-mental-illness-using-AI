from flask import Flask, request, jsonify, render_template,make_response, send_from_directory
import pickle
import pandas as pd
import sys
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Ensure the model path is accessible
sys.path.append(os.path.abspath("C:/Anju/Academic/Sem 4/Capstone/Project/Mental_Health-Web/Mental_Health-Web/model"))
from model.ml_pipeline import MentalHealthModel  # Ensure correct import

# Load trained model
with open("C:/Anju/Academic/Sem 4/Capstone/Project/Mental_Health-Web/Mental_Health-Web/model/mental_health_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index1.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        input_df = pd.DataFrame([data])

        # Ensure model has selected_feature_names
        if not hasattr(model, "selected_feature_names") or model.selected_feature_names is None:
            return jsonify({"error": "Model feature names are missing. Retrain the model."}), 500

        # Preprocess input using the model's preprocessing function
        processed_input = model.preprocess_mental_health_data(input_df)

        # Align input features to match model training
        missing_cols = set(model.selected_feature_names) - set(processed_input.columns)
        for col in missing_cols:
            processed_input[col] = 0  # Fill missing columns with 0

        # Ensure correct column order
        processed_input = processed_input[model.selected_feature_names]

        # Make predictions
        prediction = model.predict(processed_input)[0]
        probability = model.predict_proba(processed_input)[0]

        # Determine risk level
        if probability >= 0.75:
            risk_level = "High risk"
        elif probability >= 0.45:
            risk_level = "Medium risk"
        else:
            risk_level = "Low risk"

        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability),
            "risk_level": risk_level
        })

        # return jsonify({"prediction": int(prediction), "probability": float(probability)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
