import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
import os

# 1. Load latest cleaned data
DATA_PATH = "data/cleaned"
file_list = os.listdir(DATA_PATH)
file_list.sort(reverse=True)
latest_file = [f for f in file_list if f.startswith("cleaned_data_")][0]
df = pd.read_csv(os.path.join(DATA_PATH, latest_file))

# 2. Prepare data
texts = df["text"].astype(str).tolist()
labels = df["label"].tolist()

# 3. Load sentence-transformers model
print("🔍 Loading sentence-transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Generate embeddings
print("🧠 Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# 6. Train Logistic Regression classifier
print("🔧 Training Logistic Regression model...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 7. Evaluate
print("📊 Evaluation results:")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# 8. Save model
os.makedirs("models/bert_lr", exist_ok=True)
dump(clf, "models/bert_lr/classifier.joblib")
model.save("models/bert_lr/embedding_model")
print("✅ Classifier and embedding model saved to models/bert_lr")
