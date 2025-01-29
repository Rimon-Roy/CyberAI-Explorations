from flask import Flask, render_template, request
import pickle
import pandas as pd
import requests
import os

app = Flask(__name__)

# Define model paths on GitHub
GITHUB_BASE_URL = "https://raw.githubusercontent.com/Rimon-Roy/CyberAI-Explorations/main/Phising_website_detection/Models/"
MODELS = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
    "CatBoost": "catboost.pkl",
    "TabNet": "tabnet.pkl"
}

# Model Evaluation Metrics (Best model recommendation)
model_scores = {
    "Logistic Regression": 0.9510,
    "Random Forest": 0.9855,
    "Decision Tree": 0.9640,
    "Gradient Boosting": 0.9790,
    "CatBoost": 0.9865,
    "TabNet": 0.9820
}
recommended_model = max(model_scores, key=model_scores.get)

# Function to download and load models
def load_model(model_name):
    model_path = f"models/{MODELS[model_name]}"
    if not os.path.exists(model_path):
        url = GITHUB_BASE_URL + MODELS[model_name]
        response = requests.get(url)
        os.makedirs("models", exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(response.content)
    return pickle.load(open(model_path, "rb"))

# Load models on startup
loaded_models = {name: load_model(name) for name in MODELS}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    selected_model = None

    if request.method == "POST":
        selected_model = request.form["model"]
        features = [float(request.form[f"feature_{i}"]) for i in range(1, 50)]
        model = loaded_models[selected_model]
        prediction = model.predict([features])[0]

    return render_template("index.html", models=MODELS.keys(), prediction=prediction,
                           selected_model=selected_model, recommended_model=recommended_model)

if __name__ == "__main__":
    app.run(debug=True)

