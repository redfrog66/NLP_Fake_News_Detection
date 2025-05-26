import pandas as pd
import re
from datetime import datetime
import os

# Load datasets
fake_df = pd.read_csv('data/raw/Fake.csv')
true_df = pd.read_csv('data/raw/True.csv')

# Add labels
fake_df['label'] = 0
true_df['label'] = 1

# Combine datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Define enhanced text cleaning function
custom_stopwords = set([
    'the', 'and', 'is', 'in', 'to', 'of', 'for', 'on', 'with', 'a', 'an', 'by',
    'at', 'from', 'or', 'as', 'that', 'this', 'it', 'be', 'are', 'was', 'were'
])

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\\S+|www\\.\\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\\s]", "", text)            # Remove punctuation and numbers
    text = text.lower()                                     # Lowercase
    text = re.sub(r"\\s+", " ", text).strip()            # Remove extra whitespace
    words = text.split()
    words = [word for word in words if word not in custom_stopwords and len(word) > 2]  # Filter
    return " ".join(words)

# Apply cleaning to relevant columns
if 'text' in df.columns:
    df['text'] = df['text'].apply(clean_text)
elif 'content' in df.columns:
    df['content'] = df['content'].apply(clean_text)

# Create timestamped filename
now = datetime.now()
timestamp = now.strftime("%H%M")
output_path = f"data/cleaned/cleaned_data_{timestamp}.csv"

# Ensure output directory exists
os.makedirs("data/cleaned", exist_ok=True)

# Save cleaned dataset
df.to_csv(output_path, index=False)

print(f"✅ Cleaned data saved to {output_path}")