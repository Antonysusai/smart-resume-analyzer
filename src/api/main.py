"""
FastAPI Backend — Smart Resume Analyzer API
Endpoints for upload, analysis, batch processing, and health check.
"""

import io
import time
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger
import uvicorn

from src.config import settings
from src.cv.document_processor import DocumentProcessor
from src.nlp.extractor import ResumeNLPExtractor
from src.ml.scorer import ResumeScorer


# ──────────────────────────────────────────────────────────────────────────────
# App init
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="CV + NLP + ML powered Resume Analysis API",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded singletons
_doc_processor: Optional[DocumentProcessor] = None
_nlp_extractor: Optional[ResumeNLPExtractor] = None
_scorer: Optional[ResumeScorer] = None


def get_doc_processor() -> DocumentProcessor:
    global _doc_processor
    if _doc_processor is None:
        _doc_processor = DocumentProcessor(dpi=settings.DPI)
    return _doc_processor


def get_nlp_extractor() -> ResumeNLPExtractor:
    global _nlp_extractor
    if _nlp_extractor is None:
        _nlp_extractor = ResumeNLPExtractor(
            spacy_model=settings.SPACY_MODEL,
            embedding_model=settings.EMBEDDING_MODEL,
        )
    return _nlp_extractor


def get_scorer() -> ResumeScorer:
    global _scorer
    if _scorer is None:
        _scorer = ResumeScorer(model_path=settings.ML_MODEL_PATH)
    return _scorer


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    job_description: str
    job_skills: list[str] = []


class AnalyzeResponse(BaseModel):
    success: bool
    processing_time_ms: float
    file_info: dict
    entities: dict
    skills_by_category: dict
    score: dict
    message: str = ""


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: dict


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {"message": f"Welcome to {settings.APP_NAME} API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        models_loaded={
            "document_processor": _doc_processor is not None,
            "nlp_extractor": _nlp_extractor is not None,
            "ml_scorer": _scorer is not None,
        },
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_resume(
    file: UploadFile = File(..., description="Resume file — PDF, PNG, JPG"),
    job_description: str = Form(..., description="Target job description"),
    job_skills: str = Form("", description="Comma-separated required skills"),
):
    """
    Full resume analysis pipeline:
    1. CV: extract text from uploaded document
    2. NLP: parse entities (name, email, skills, experience, education)
    3. ML: score the resume against the job description
    """
    start = time.perf_counter()

    # Validate file size
    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {settings.MAX_FILE_SIZE_MB} MB",
        )

    # Parse job_skills from form field
    parsed_job_skills = [s.strip() for s in job_skills.split(",") if s.strip()]

    try:
        # ── Stage 1: CV processing ────────────────────────────────────────────
        logger.info(f"Processing file: {file.filename} ({size_mb:.2f} MB)")
        doc_processor = get_doc_processor()
        cv_result = doc_processor.process(file_bytes, file.filename)

        raw_text = cv_result["raw_text"]
        if not raw_text.strip():
            raise HTTPException(status_code=422, detail="Could not extract text from document.")

        # ── Stage 2: NLP extraction ───────────────────────────────────────────
        nlp = get_nlp_extractor()
        entities = nlp.extract(raw_text)
        skills_by_category = nlp.get_skills_by_category(entities.skills)
        semantic_sim = nlp.semantic_similarity(raw_text, job_description)

        # ── Stage 3: ML scoring ───────────────────────────────────────────────
        scorer = get_scorer()
        score = scorer.score(
            entities=entities,
            job_description=job_description,
            job_skills=parsed_job_skills,
            semantic_sim=semantic_sim,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return AnalyzeResponse(
            success=True,
            processing_time_ms=round(elapsed_ms, 1),
            file_info={
                "filename": file.filename,
                "size_mb": round(size_mb, 3),
                "pages": cv_result["pages"],
                "is_scanned": cv_result["is_scanned"],
                "quality_score": cv_result["quality_score"],
            },
            entities={
                "name": entities.name,
                "email": entities.email,
                "phone": entities.phone,
                "linkedin": entities.linkedin,
                "github": entities.github,
                "location": entities.location,
                "skills": entities.skills,
                "education": [
                    {
                        "degree": e.degree,
                        "institution": e.institution,
                        "year": e.year,
                    }
                    for e in entities.education
                ],
                "experience": [
                    {
                        "title": e.title,
                        "company": e.company,
                        "start_date": e.start_date,
                        "end_date": e.end_date,
                    }
                    for e in entities.experience
                ],
                "certifications": entities.certifications,
                "summary": entities.summary,
                "total_experience_years": entities.total_experience_years,
            },
            skills_by_category=skills_by_category,
            score={
                "overall_score": score.overall_score,
                "grade": score.grade,
                "fit_probability": score.fit_probability,
                "skill_match_score": score.skill_match_score,
                "experience_score": score.experience_score,
                "education_score": score.education_score,
                "completeness_score": score.completeness_score,
                "semantic_similarity": score.semantic_similarity,
                "matched_skills": score.matched_skills,
                "missing_skills": score.missing_skills,
                "strengths": score.strengths,
                "gaps": score.gaps,
                "recommendations": score.recommendations,
                "shap_explanation": score.shap_explanation,
            },
            message="Analysis complete",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/batch-analyze", tags=["Analysis"])
async def batch_analyze(
    files: list[UploadFile] = File(...),
    job_description: str = Form(...),
    job_skills: str = Form(""),
):
    """
    Analyze multiple resumes at once and return ranked results.
    """
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per batch.")

    results = []
    for f in files:
        try:
            file_bytes = await f.read()
            doc_processor = get_doc_processor()
            nlp = get_nlp_extractor()
            scorer = get_scorer()

            cv_result = doc_processor.process(file_bytes, f.filename)
            entities = nlp.extract(cv_result["raw_text"])
            parsed_skills = [s.strip() for s in job_skills.split(",") if s.strip()]
            semantic_sim = nlp.semantic_similarity(cv_result["raw_text"], job_description)
            score = scorer.score(entities, job_description, parsed_skills, semantic_sim)

            results.append({
                "filename": f.filename,
                "name": entities.name,
                "overall_score": score.overall_score,
                "grade": score.grade,
                "fit_probability": score.fit_probability,
                "matched_skills_count": len(score.matched_skills),
                "total_experience_years": entities.total_experience_years,
            })
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})

    # Rank by overall score
    results.sort(key=lambda r: r.get("overall_score", 0), reverse=True)
    return {"total": len(results), "ranked_candidates": results}


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
