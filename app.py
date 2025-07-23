# app.py
from flask import Flask, request, jsonify, render_template # <-- Import render_template
from model import SuicideModel
import os

app = Flask(__name__)

# Define paths to artifacts
MODEL_PATH = os.path.join("artifacts", "xgb_tuned_model.joblib")
SCALER_PATH = os.path.join("artifacts", "ss_model.joblib")
VOCAB_PATH = os.path.join("artifacts", "bert-base-uncased-vocab.txt")
GLOVE_PATH = os.path.join("artifacts", "glove-wiki-gigaword-100.kv")

# Load the model once when the app starts
model = SuicideModel(MODEL_PATH, SCALER_PATH, VOCAB_PATH, GLOVE_PATH)

@app.route("/")
def index():
    # Serve the HTML page
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        text = data.get('text', '') # Use .get for safety
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
            
        result = model.predict(text)
        return jsonify(result)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred during prediction."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))