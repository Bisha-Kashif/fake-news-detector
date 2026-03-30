from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import re

app = FastAPI(title="Fake News Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


class Request(BaseModel):
    text: str


def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@app.get("/")
def root():
    return {"message": "Fake News Detector API is running!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: Request):
    if not req.text or len(req.text.strip()) < 5:
        raise HTTPException(status_code=422, detail="Text too short.")

    cleaned    = clean(req.text)
    vec        = vectorizer.transform([cleaned])
    pred       = int(model.predict(vec)[0])
    proba      = model.predict_proba(vec)[0]
    confidence = float(proba[pred])

    THRESHOLD = 0.70
    if confidence < THRESHOLD:
        label = "Uncertain"
        emoji = "⚠️"
    else:
        label = "Real" if pred == 1 else "Fake"
        emoji = "✅" if label == "Real" else "🚨"

    pct = round(confidence * 100)

    if label == "Uncertain":
        verdict = f"Model is not confident enough ({pct}%). Please verify manually."
    elif label == "Real":
        verdict = f"This looks like legitimate news ({pct}% confidence)."
    else:
        verdict = f"High probability of fake news ({pct}% confidence). Do not share."

    return {
        "label":      label,
        "confidence": round(confidence, 4),
        "verdict":    verdict,
        "emoji":      emoji,
    }
