# Fake News Detector 🇵🇰

Classifies news headlines as **Real ✅**, **Fake 🚨**, or **Uncertain ⚠️**

---

## Live Links
- **Frontend:** https://Bisha-Kashif.github.io/fake-news-detector/frontend/
- **Backend API:** https://huggingface.co/spaces/Bisha-Kashif/fake-news-detector

---

## Folder Structure

```
fake-news-detector/
│
├── frontend/              ← Hosted on GitHub Pages
│   └── index.html         (the web app users interact with)
│
├── backend/               ← Hosted on HuggingFace Spaces
│   ├── app.py             (FastAPI server)
│   ├── Dockerfile         (for containerization)
│   ├── requirements.txt
│   ├── fake_news_model.pkl
│   └── vectorizer.pkl
│
└── model-training/        ← Training code (on GitHub)
    ├── train.py           (trains the ML model)
    ├── requirements.txt
    ├── confusion_matrix.png
    ├── model_comparison.png
    └── data/
        └── sample_data.csv
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Logistic Regression (Scikit-learn) |
| Features | TF-IDF Vectorizer |
| API | FastAPI (Python) |
| Containerization | Docker |
| Backend Hosting | HuggingFace Spaces |
| Frontend Hosting | GitHub Pages |

---

## How It Works

```
User types headline
        ↓
 frontend/index.html
  (GitHub Pages)
        ↓  POST /predict
 backend/app.py
  (HuggingFace)
        ↓
 TF-IDF + Logistic Regression
        ↓
 Real ✅ / Fake 🚨 / Uncertain ⚠️
```

---

*Pre-Mid ML Project · PUCIT 2025 · Bisha Kashif*

## Model Accuracy
Test Accuracy: 98%. F1-Score: 0.97
