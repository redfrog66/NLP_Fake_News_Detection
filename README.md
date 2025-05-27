# 📰 NLP Fake News Detector

## 🔍 Project Overview

This project explores several NLP approaches for detecting fake news, with a blend of classical machine learning and ChatGPT-assisted features.

---

## 🧪 Workflow Summary

1. **Notebook Prototyping**  
   We initially tested out ideas and model setups in Jupyter notebooks.

2. **BERT Training**  
   On a cleaned dataset, we trained a BERT model after preprocessing and visualizations.  
   Unfortunately, it produced high-confidence predictions that were often incorrect.  
   > This can happen when fine-tuning large models on small or imbalanced datasets, causing overfitting or shallow generalization.

3. **TF-IDF + Logistic Regression**  
   We transitioned to a simpler, interpretable approach using TF-IDF vectors and Logistic Regression.  
   - ✅ This reduced false confidence and improved transparency.  
   - ⚠️ But the model sometimes underperforms due to its linear structure and inability to handle complex context.  
   > Logistic Regression is often better at reflecting uncertainty, which is valuable when classifying ambiguous or clickbait content.

---

## 🤖 ChatGPT Integration

To add explainability and engagement, we integrated the ChatGPT API:

- 🧠 `Explain Why`: Briefly explains why the submitted news might be real or fake
- 📝 `Generate Fake`: Produces a fake news headline
- ✅ `Generate True`: Produces a real, verifiable news headline

> ChatGPT serves as an intelligent assistant to double-check and reflect on model output.

---

## 🔬 Considering LoRA?

We are considering applying **LoRA (Low-Rank Adaptation)** to fine-tune transformer models in a lightweight and efficient way.  
> It could allow us to gain BERT-level reasoning with less risk of overfitting and faster training — making it an ideal upgrade path.

