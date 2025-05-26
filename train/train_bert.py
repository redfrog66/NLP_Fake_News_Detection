import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample, class_weight
from joblib import dump
import os

# 1. Load latest cleaned data
DATA_PATH = "data/cleaned"
file_list = os.listdir(DATA_PATH)
file_list.sort(reverse=True)
latest_file = [f for f in file_list if f.startswith("cleaned_data_")][0]
df = pd.read_csv(os.path.join(DATA_PATH, latest_file))

# 2. Balance the dataset
majority = df[df['label'] == 0]
minority = df[df['label'] == 1]

if len(minority) < len(majority):
    majority_downsampled = resample(majority, replace=False, n_samples=len(minority), random_state=42)
    df = pd.concat([majority_downsampled, minority]).sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Prepare data
texts = df["text"].astype(str).tolist()
labels = df["label"].tolist()

# 4. TF-IDF Vectorization
print("🔤 Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, stratify=labels, random_state=42
)

# 6. Train Logistic Regression classifier
print("🔧 Training Logistic Regression model...")
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weight_dict = dict(zip(np.unique(labels), class_weights))
clf = LogisticRegression(max_iter=1000, class_weight=class_weight_dict)
clf.fit(X_train, y_train)

# 7. Evaluate
print("📊 Evaluation results:")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("y_pred counts:", np.bincount(y_pred))

# 8. Manual test predictions
print("\n🧪 Manual prediction samples:")
samples = [
    "The president was impeached for covering up a UFO.",
    "NASA discovers water on the moon.",
    "Aliens have landed in California, witnesses say.",
    "The stock market reached an all-time high today.",
    "Breaking: actor fakes own death to promote movie."
]
sample_vectors = vectorizer.transform(samples)
sample_preds = clf.predict(sample_vectors)

for text, pred in zip(samples, sample_preds):
    print(f"[{'Fake' if pred == 0 else 'Real'}] {text}")

# 9. Save model
os.makedirs("models/tfidf_lr", exist_ok=True)
dump(clf, "models/tfidf_lr/classifier.joblib")
dump(vectorizer, "models/tfidf_lr/vectorizer.joblib")
print("✅ TF-IDF model and vectorizer saved to models/tfidf_lr")
