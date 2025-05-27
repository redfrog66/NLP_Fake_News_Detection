from flask import render_template, request, jsonify
from app import app
from models.tfidf_model import predict_news
import openai
from openai import OpenAI

# Create a client using the API key
client = OpenAI(api_key=open("api_key.txt").read().strip())


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

@app.route("/explain", methods=["POST"])
def explain():
    user_text = request.json.get("text", "")
    prompt = f"Explain in 3 short sentences why the following news is likely real or fake:\n\n\"{user_text}\""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{ "role": "user", "content": prompt }],
        max_tokens=100
    )
    explanation = response.choices[0].message.content.strip()
    return jsonify({"explanation": explanation})

@app.route("/generate_fake")
def generate_fake():
    prompt = "Generate a one-sentence fake news headline."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{ "role": "user", "content": prompt }],
        max_tokens=60
    )
    return jsonify({"text": response.choices[0].message.content.strip()})


@app.route("/generate_true")
def generate_true():
    prompt = "Generate a one-sentence real and verifiable news headline."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{ "role": "user", "content": prompt }],
        max_tokens=60
    )
    return jsonify({"text": response.choices[0].message.content.strip()})

