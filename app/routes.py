from flask import render_template, request, jsonify
from app import app
from models.tfidf_model import predict_news

@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/analyze")
def analyze():
    return render_template("analyze.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    label, confidence = predict_news(text)
    return jsonify({
        "label": "Real" if label == 1 else "Fake",
        "confidence": confidence
    })
