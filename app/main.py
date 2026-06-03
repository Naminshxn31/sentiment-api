from fastapi import FastAPI
from app.schemas import PredictRequest, PredictResponse
from app.model import predict

app = FastAPI(
    title="Sentiment Analysis API",
    description="Predict sentiment of text using ML",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running!"}

@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(request: PredictRequest):
    result = predict(request.text)
    return result