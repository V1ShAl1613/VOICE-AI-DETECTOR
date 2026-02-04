# 🎙️ Voice AI Detection REST API

**AI-Generated vs Human Voice Classification (5 Languages)**

---

## 📌 Overview

This project implements a production-grade REST API that detects whether a given voice recording is **AI-generated** or spoken by a **real human**.

The system supports five languages:
- Tamil
- English
- Hindi
- Malayalam
- Telugu

The solution uses **acoustic feature analysis** and a **classical machine learning model (XGBoost)** to ensure accuracy, explainability, and compliance, without relying on external AI detection APIs.

---

## 🚀 Key Features

- ✅ AI vs Human voice classification
- 🌍 Multilingual support (5 languages)
- 🔐 API-key protected REST endpoint
- 🎧 Accepts Base64-encoded audio input
- 🧠 Explainable predictions (feature-based)
- 📦 Dockerized for deployment
- ⚖️ Ethical and transparent ML pipeline

---

## 🧠 Technical Approach

### 1️⃣ Audio Feature Extraction

Audio is analyzed using `librosa`, extracting:
- **MFCC** (mean & variance)
- **Spectral centroid, rolloff, flatness**
- **Pitch (F0 mean & variance)**
- **Jitter & shimmer approximations**
- **Temporal and spectral consistency metrics**

These features capture natural human irregularities versus AI voice synthesis artifacts.

### 2️⃣ Machine Learning Model

| Component | Details |
|-----------|---------|
| **Primary Model** | XGBoost Classifier |
| **Pipeline** | StandardScaler → XGBoost |
| **Training** | Offline only (no runtime training) |
| **Inference** | Deterministic, probability-based |

**Outputs:**
- `HUMAN`
- `AI_GENERATED`

...with a calibrated confidence score.

---

## 📊 Dataset

### Human Speech
- **Source:** Google FLEURS (via HuggingFace Datasets)
- Real human speech across all supported languages

### AI-Generated Speech
Generated using:
- Microsoft Edge TTS
- Google Text-to-Speech (gTTS)

Multiple voices and sentences are used to reduce bias.

> ⚠️ All datasets are used strictly for research and evaluation purposes.

---

## 📂 Project Structure

```
GUVI HACKTHON/
├── app/
│   ├── api/           # API routes
│   ├── audio/         # Audio decoding & feature extraction
│   ├── ml/            # Model loader & explanation logic
│   └── main.py        # FastAPI entry point
├── training/
│   ├── data_generator.py
│   └── train_model.py
├── dataset/
│   ├── human/
│   └── ai_generated/
├── tests/
├── Dockerfile
└── README.md
```

---

## 🔌 API Specification

### Endpoint
```
POST /api/voice-detection
```

### Headers
```
Content-Type: application/json
x-api-key: YOUR_API_KEY
```

### Request Body
```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<BASE64_ENCODED_AUDIO>"
}
```

### Success Response
```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.93,
  "explanation": "Low pitch variance and overly smooth spectral transitions detected"
}
```

### Error Response
```json
{
  "status": "error",
  "message": "Invalid API key or malformed request"
}
```

---

## 🔐 Security

- API key authentication enforced
- Requests without a valid key are rejected
- Strong input validation using Pydantic schemas

---

## 🧪 Validation & Testing Summary

- ✅ Trained model artifact verified
- ✅ Offline predictions show clear class separation
- ✅ API functional and security tests passed
- ⚠️ WAV files fully supported across platforms
- ⚠️ MP3 support improved; some rare Windows MP3 encodings may fail due to codec limitations

---

## 🐳 Docker Support

### Build
```bash
docker build -t voice-ai-detector .
```

### Run
```bash
docker run -p 8000:8000 -e API_KEY=secret123 voice-ai-detector
```

> The trained model is baked into the Docker image for reliable cold starts.

---

## 🧾 Run Locally

```bash
cd GUVI HACKTHON
python -m uvicorn app.main:app --port 8000 --reload
```

Run tests:
```bash
python tests/test_api.py
```

---

## ⚠️ Compliance & Ethics

- ❌ No hard-coded outputs
- ❌ No external AI detection APIs
- ❌ No runtime training
- ✅ Explainable ML
- ✅ Transparent dataset usage
- ✅ Competition-safe design

---

## 🏁 Project Status


- Model trained and validated
- API stable and secure
- All problem constraints satisfied
- Production-grade architecture

---

## 🔮 Future Improvements

- Full MP3 normalization via ffmpeg
- Larger AI-generated dataset for balance
- Automatic threshold calibration
- Async processing for higher throughput

---

## 🔧 Designed & Engineered By

**Madeline Prathana V**  
**Marlene Saraniya**  
**Vishal V**
