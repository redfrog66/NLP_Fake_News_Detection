from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer once (global for reuse)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("your_trained_model_path")
model.eval()

def predict_news(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][label].item()
    return label, confidence
