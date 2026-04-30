# 📄 Smart Resume Analyzer

> **Computer Vision + NLP + Machine Learning** powered resume analysis system built with FastAPI, Streamlit, spaCy, HuggingFace Transformers, XGBoost, and OpenCV.

[![CI](https://github.com/AntonySusaivictor/smart-resume-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/AntonySusaivictor/smart-resume-analyzer/actions)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🏗 Architecture

```
smart-resume-analyzer/
├── src/
│   ├── cv/
│   │   └── document_processor.py   # OpenCV pipeline: deskew, denoise, OCR
│   ├── nlp/
│   │   └── extractor.py            # spaCy NER + HuggingFace embeddings
│   ├── ml/
│   │   └── scorer.py               # XGBoost scorer + SHAP explanations
│   ├── api/
│   │   └── main.py                 # FastAPI REST API
│   └── config.py                   # Pydantic settings
├── frontend/
│   └── app.py                      # Streamlit UI
├── tests/
│   └── test_all.py                 # pytest test suite
├── .github/workflows/ci.yml        # GitHub Actions CI/CD
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🧠 How It Works

### Stage 1 — Computer Vision (OpenCV + Tesseract)
- Detects whether the uploaded PDF is **digital** (text layer) or **scanned** (image-only)
- For scanned documents: converts PDF pages to high-DPI images via `pdf2image`
- **Preprocessing pipeline**: grayscale → deskew (Hough lines) → denoise → adaptive threshold → morphological cleanup
- Runs **Tesseract OCR** (OEM 3, PSM 6) on preprocessed images
- Computes an image **quality score** via Laplacian variance

### Stage 2 — NLP (spaCy + HuggingFace)
- **Named Entity Recognition** (spaCy `en_core_web_lg`) for names, orgs, locations
- **Regex extraction** for email, phone, LinkedIn, GitHub, dates
- **Skills taxonomy matching** across 6 categories: Programming, ML/AI, Data, Cloud/DevOps, Analytics, Web
- **Experience parsing**: job titles, companies, date ranges, duration estimation
- **Sentence-transformer embeddings** (`all-MiniLM-L6-v2`) for semantic similarity between resume and JD

### Stage 3 — Machine Learning (XGBoost + SHAP)
- 20 engineered features from the parsed resume + job description
- **XGBoost binary classifier** predicts fit probability (0–1)
- **SHAP explanations** show which features drove the prediction
- **Rule-based sub-scores**: skill match %, experience score, education score, completeness
- **Weighted overall score** (0–100) with letter grade A/B/C/D

---

## 🚀 Quick Start

### Option 1: Local (recommended for development)

```bash
# 1. Clone the repo
git clone https://github.com/AntonySusaivictor/smart-resume-analyzer.git
cd smart-resume-analyzer

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLP models
python -m spacy download en_core_web_lg
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 5. Copy environment config
cp .env.example .env

# 6. Start the API
uvicorn src.api.main:app --reload --port 8000

# 7. In a new terminal, start the UI
streamlit run frontend/app.py
```

Open **http://localhost:8501** for the UI, **http://localhost:8000/docs** for the API.

### Option 2: Docker Compose

```bash
docker-compose up --build
```

---

## 🔌 API Reference

### `POST /analyze`
Analyze a single resume against a job description.

**Form fields:**
| Field | Type | Description |
|-------|------|-------------|
| `file` | File | Resume PDF/PNG/JPG |
| `job_description` | string | Full job description text |
| `job_skills` | string | Comma-separated required skills |

**Response:**
```json
{
  "success": true,
  "processing_time_ms": 1234.5,
  "file_info": { "pages": 2, "is_scanned": false, "quality_score": 0.94 },
  "entities": {
    "name": "John Doe",
    "email": "john@example.com",
    "skills": ["python", "machine learning", "sql"],
    "total_experience_years": 5.0
  },
  "skills_by_category": {
    "programming": ["python"],
    "ml_ai": ["machine learning"]
  },
  "score": {
    "overall_score": 82.5,
    "grade": "A",
    "fit_probability": 0.87,
    "matched_skills": ["python", "sql"],
    "missing_skills": ["kubernetes"],
    "strengths": ["Strong skill alignment"],
    "recommendations": ["Add your LinkedIn URL"]
  }
}
```

### `POST /batch-analyze`
Rank multiple resumes against one job description. Returns candidates sorted by score.

### `GET /health`
Health check with model load status.

---

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 📊 Streamlit UI Features

- **Single Resume mode**: Upload → Analyze → See full breakdown
- **Batch Ranking mode**: Upload up to 20 resumes → ranked leaderboard
- **Radar chart**: Visual score breakdown across 5 dimensions
- **SHAP feature importance**: Understand what drove the ML score
- **Skills gap analysis**: Matched vs missing skills with color chips
- **Recommendations**: Actionable suggestions to improve the resume

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Computer Vision | OpenCV, pdf2image, Pillow, Tesseract, PyMuPDF |
| NLP | spaCy, HuggingFace Transformers, sentence-transformers, NLTK |
| Machine Learning | XGBoost, scikit-learn, SHAP, NumPy, Pandas |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit, Plotly |
| Testing | pytest, pytest-asyncio, httpx |
| DevOps | Docker, GitHub Actions |

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built by [AntonySusaivictor](https://github.com/AntonySusaivictor) — Data Scientist | MSc Data Science (Coventry University)*
