from joblib import load

# Load the TF-IDF vectorizer and classifier
vectorizer = load("models/tfidf_lr/vectorizer.joblib")
classifier = load("models/tfidf_lr/classifier.joblib")

def predict_news(text):
    X = vectorizer.transform([text])
    prediction = classifier.predict(X)[0]
    confidence = classifier.predict_proba(X)[0][prediction]
    return prediction, round(confidence, 3)
