import re
import joblib
import numpy as np

model = joblib.load('app/model.pkl')
vectorizer = joblib.load('app/vectorizer.pkl')

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict(text: str) -> dict:
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    confidence = float(np.max(model.predict_proba(vec)))
    
    return {
        "text": text,
        "sentiment": "Positive" if prediction == 1 else "Negative",
        "confidence": round(confidence, 4)
    }