
from fastapi import FastAPI
from pydantic import BaseModel
from src.models.emotion_evaluators import DistilBert, MultiBert

app = FastAPI(title="Sentiment Analysis API")

# Load model once at startup
model = DistilBert()

class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    sentiment: str

@app.post("/predict", response_model=PredictionOutput)
def predict_sentiment(input_data: TextInput):
    label = model.predict_single(input_data.text)
    if label == 1:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'
    return PredictionOutput(
        sentiment = sentiment
    )

@app.get("/health")
def health_check():
    return {"status": "healthy"}


