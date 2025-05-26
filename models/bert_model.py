from sentence_transformers import SentenceTransformer
from joblib import load

# Load models once
clf = load("models/bert_lr/classifier.joblib")
embedder = SentenceTransformer("models/bert_lr/embedding_model")

def predict_news(text):
    embedding = embedder.encode([text])  # Shape: (1, 384)
    prediction = clf.predict(embedding)[0]
    confidence = clf.predict_proba(embedding)[0][prediction]
    return prediction, round(confidence, 3)
