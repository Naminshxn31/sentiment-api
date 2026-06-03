import re
import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ===== Load real dataset (IMDB) =====
print("Loading dataset...")
dataset = load_dataset("stanfordnlp/imdb")
df_train = dataset['train'].to_pandas().sample(2000, random_state=42)
df_test = dataset['test'].to_pandas().sample(500, random_state=42)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)        # ลบ HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Cleaning text...")
df_train['clean_text'] = df_train['text'].apply(clean_text)
df_test['clean_text'] = df_test['text'].apply(clean_text)

# ===== Train =====
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(df_train['clean_text'])
X_test_vec = vectorizer.transform(df_test['clean_text'])

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, df_train['label'])

# ===== Evaluate =====
y_pred = model.predict(X_test_vec)
print(f"\nAccuracy: {accuracy_score(df_test['label'], y_pred):.4f}")
print(classification_report(df_test['label'], y_pred))

# ===== Save =====
joblib.dump(model, 'app/model.pkl')
joblib.dump(vectorizer, 'app/vectorizer.pkl')
print("Model saved!")