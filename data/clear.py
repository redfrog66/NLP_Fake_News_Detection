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

# Define a basic text cleaning function
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\\S+|www\\.\\S+", "", text)         # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\\s]", "", text)             # Remove punctuation
    text = text.lower()                                     # Lowercase
    text = re.sub(r"\\s+", " ", text).strip()               # Remove extra whitespace
    return text

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
