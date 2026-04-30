"""
Test Suite — Smart Resume Analyzer
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# ──────────────────────────────────────────────────────────────────────────────
# CV Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDocumentProcessor:

    def test_clean_ocr_text(self):
        from src.cv.document_processor import DocumentProcessor
        dp = DocumentProcessor()
        dirty = "Hello   World\n\n\n\nFoo  Bar\x93"
        clean = dp._clean_ocr_text(dirty)
        assert "  " not in clean
        assert "\n\n\n" not in clean

    def test_unsupported_format_raises(self):
        from src.cv.document_processor import DocumentProcessor
        dp = DocumentProcessor()
        with pytest.raises(ValueError, match="Unsupported file type"):
            dp.process(b"fake", "resume.docx")

    def test_deskew_no_lines(self):
        import cv2
        import numpy as np
        from src.cv.document_processor import DocumentProcessor
        dp = DocumentProcessor()
        blank = np.ones((100, 100), dtype=np.uint8) * 255
        result = dp._deskew(blank)
        assert result.shape == blank.shape

    def test_quality_score_range(self):
        import cv2
        import numpy as np
        from src.cv.document_processor import DocumentProcessor
        dp = DocumentProcessor()
        img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        score = dp._estimate_quality(img)
        assert 0.0 <= score <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# NLP Tests
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_RESUME = """
John Doe
john.doe@email.com
+1 (555) 123-4567
linkedin.com/in/johndoe
github.com/johndoe
San Francisco, CA

Summary:
Experienced Data Scientist with 5 years building ML pipelines and NLP systems.

Experience:
Senior Data Scientist — Acme Corp
Jan 2020 — Present
Built Python-based ML models using scikit-learn and PyTorch.

Junior ML Engineer — TechStartup Inc
Jun 2018 — Dec 2019
Developed NLP pipelines using spaCy and transformers.

Education:
M.Tech in Computer Science — Stanford University 2018
B.Tech in Electronics — IIT Madras 2016

Skills:
Python, Machine Learning, Deep Learning, NLP, PyTorch, scikit-learn,
SQL, AWS, Docker, FastAPI, Pandas, NumPy

Certifications:
AWS Certified Machine Learning Specialty
"""


class TestResumeNLPExtractor:

    @pytest.fixture(scope="class")
    def extractor(self):
        from src.nlp.extractor import ResumeNLPExtractor
        return ResumeNLPExtractor()

    def test_email_extraction(self, extractor):
        entities = extractor.extract(SAMPLE_RESUME)
        assert entities.email == "john.doe@email.com"

    def test_phone_extraction(self, extractor):
        entities = extractor.extract(SAMPLE_RESUME)
        assert entities.phone is not None

    def test_skills_extraction(self, extractor):
        entities = extractor.extract(SAMPLE_RESUME)
        skills_lower = [s.lower() for s in entities.skills]
        assert "python" in skills_lower
        assert "machine learning" in skills_lower

    def test_github_extraction(self, extractor):
        entities = extractor.extract(SAMPLE_RESUME)
        assert entities.github is not None
        assert "github" in entities.github.lower()

    def test_semantic_similarity_same(self, extractor):
        sim = extractor.semantic_similarity("machine learning engineer", "machine learning engineer")
        assert sim > 0.95

    def test_semantic_similarity_different(self, extractor):
        sim = extractor.semantic_similarity("machine learning engineer", "chef de cuisine")
        assert sim < 0.6

    def test_skills_by_category(self, extractor):
        entities = extractor.extract(SAMPLE_RESUME)
        by_cat = extractor.get_skills_by_category(entities.skills)
        assert "programming" in by_cat or "ml_ai" in by_cat


# ──────────────────────────────────────────────────────────────────────────────
# ML Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestResumeScorer:

    @pytest.fixture(scope="class")
    def setup(self):
        from src.nlp.extractor import ResumeNLPExtractor
        from src.ml.scorer import ResumeScorer
        nlp = ResumeNLPExtractor()
        scorer = ResumeScorer()
        entities = nlp.extract(SAMPLE_RESUME)
        return nlp, scorer, entities

    def test_score_range(self, setup):
        nlp, scorer, entities = setup
        jd = "Looking for a Data Scientist with Python, ML, NLP, AWS experience."
        sim = nlp.semantic_similarity(SAMPLE_RESUME, jd)
        score = scorer.score(entities, jd, ["python", "machine learning", "nlp", "aws"], sim)
        assert 0 <= score.overall_score <= 100

    def test_grade_assigned(self, setup):
        nlp, scorer, entities = setup
        jd = "Data Scientist role"
        sim = nlp.semantic_similarity(SAMPLE_RESUME, jd)
        score = scorer.score(entities, jd, ["python"], sim)
        assert score.grade in {"A", "B", "C", "D"}

    def test_matched_skills_subset(self, setup):
        nlp, scorer, entities = setup
        job_skills = ["python", "machine learning", "java"]
        jd = "We need Python and ML skills."
        sim = nlp.semantic_similarity(SAMPLE_RESUME, jd)
        score = scorer.score(entities, jd, job_skills, sim)
        assert "java" in score.missing_skills
        assert all(s in entities.skills for s in score.matched_skills)

    def test_feature_engineer_shape(self):
        from src.ml.scorer import FeatureEngineer
        from src.nlp.extractor import ResumeNLPExtractor
        nlp = ResumeNLPExtractor()
        fe = FeatureEngineer()
        entities = nlp.extract(SAMPLE_RESUME)
        feat = fe.build(entities, ["python", "sql"], 0.7)
        assert feat.shape == (len(FeatureEngineer.FEATURE_NAMES),)

    def test_train_predict(self):
        from src.ml.scorer import ResumeScorer, FeatureEngineer
        scorer = ResumeScorer()
        X = np.random.rand(50, len(FeatureEngineer.FEATURE_NAMES)).astype(np.float32)
        y = np.random.randint(0, 2, 50)
        scorer.train(X, y)
        assert scorer.model is not None
        proba = scorer.model.predict_proba(X[:1])
        assert proba.shape == (1, 2)


# ──────────────────────────────────────────────────────────────────────────────
# API Tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAPI:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        return TestClient(app)

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_analyze_no_file(self, client):
        r = client.post("/analyze", data={"job_description": "test"})
        assert r.status_code == 422  # unprocessable — missing file

    def test_analyze_bad_extension(self, client):
        r = client.post(
            "/analyze",
            files={"file": ("resume.exe", b"bad content", "application/octet-stream")},
            data={"job_description": "test jd"},
        )
        assert r.status_code in {422, 500}
