"""
FastAPI Backend — Smart Resume Analyzer API
Endpoints for upload, analysis, batch processing, and health check.
"""

import io
import re
import time
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger
import uvicorn

from src.config import settings
from src.cv.document_processor import DocumentProcessor
from src.nlp.extractor import ResumeNLPExtractor
from src.ml.scorer import ResumeScorer
from src.ml.confidence import ConfidenceEngine
from src.ml.hard_filter import HardFilterEngine, ResumeEntities as HardFilterResumeEntities
from src.ml.job_analyzer import JobDescriptionAnalyzer


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
_job_analyzer: Optional[JobDescriptionAnalyzer] = None
_hard_filter: Optional[HardFilterEngine] = None
_confidence_engine: Optional[ConfidenceEngine] = None


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


def get_job_analyzer() -> JobDescriptionAnalyzer:
    global _job_analyzer
    if _job_analyzer is None:
        _job_analyzer = JobDescriptionAnalyzer()
    return _job_analyzer


def get_hard_filter() -> HardFilterEngine:
    global _hard_filter
    if _hard_filter is None:
        _hard_filter = HardFilterEngine()
    return _hard_filter


def get_confidence_engine() -> ConfidenceEngine:
    global _confidence_engine
    if _confidence_engine is None:
        _confidence_engine = ConfidenceEngine()
    return _confidence_engine


def _to_hard_filter_resume(entities) -> HardFilterResumeEntities:
    """Map NLP extraction output to the lean hard-filter DTO."""
    return HardFilterResumeEntities(
        skills=entities.skills,
        total_experience_years=entities.total_experience_years,
        languages=_languages_to_dict(entities.languages),
        job_titles=[exp.title for exp in entities.experience if exp.title],
        education=_education_to_string(entities.education),
    )


def _languages_to_dict(languages: list[str]) -> dict[str, str]:
    parsed = {}
    for language in languages or []:
        lower = language.lower()
        level_match = re.search(r"\b(A1|A2|B1|B2|C1|C2)\b", language, re.IGNORECASE)
        if level_match:
            level = level_match.group(1).upper()
        elif re.search(r"\b(native|muttersprache|muttersprachlich)\b", lower):
            level = "NATIVE"
        elif re.search(r"\b(fluent|verhandlungssicher)\b", lower):
            level = "C1"
        elif re.search(r"\b(business|working|good)\b", lower):
            level = "B2"
        elif re.search(r"\b(intermediate|advanced)\b", lower):
            level = "B1"
        elif re.search(r"\b(basic|beginner|elementary)\b", lower):
            level = "A2"
        else:
            level = ""
        if "german" in lower or "deutsch" in lower:
            parsed["german"] = level
        elif "english" in lower or "englisch" in lower:
            parsed["english"] = level
    return parsed


def _education_to_string(education: list) -> str | None:
    if not education:
        return None
    return " ".join(edu.degree for edu in education if edu.degree) or None


def _hiring_decision(overall_score: float, confidence_score: float, score) -> dict:
    if overall_score > 80 and confidence_score > 0.75:
        decision = "shortlist"
    elif overall_score > 60:
        decision = "consider"
    else:
        decision = "reject"

    if score.strengths:
        reason = score.strengths[0]
    elif score.gaps:
        reason = score.gaps[0]
    elif decision == "shortlist":
        reason = "Strong skill alignment and sufficient German language evidence"
    elif decision == "consider":
        reason = "Relevant profile with gaps requiring recruiter review"
    else:
        reason = "Score is below the interview threshold"

    return {
        "decision": decision,
        "confidence": "high" if confidence_score >= 0.8 else "medium" if confidence_score >= 0.6 else "low",
        "reason": reason,
    }


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
    status: str = "accepted"
    overall_score: Optional[float] = None
    grade: Optional[str] = None
    confidence: Optional[dict] = None
    score_breakdown: Optional[dict] = None
    critical_missing_skills: list[str] = Field(default_factory=list)
    optional_missing_skills: list[str] = Field(default_factory=list)
    recruiter_insights: Optional[dict] = None
    recruiter_decision: Optional[dict] = None
    hiring_decision: Optional[dict] = None
    summary: str = ""
    issues: list[dict] = Field(default_factory=list)
    achievement_examples: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
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
            "job_analyzer": _job_analyzer is not None,
            "hard_filter": _hard_filter is not None,
            "confidence_engine": _confidence_engine is not None,
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
        requirements = get_job_analyzer().analyze(job_description, parsed_job_skills)
        hard_filter_resume = _to_hard_filter_resume(entities)
        hard_filter = get_hard_filter().evaluate(hard_filter_resume, requirements)

        if not hard_filter.pass_to_next_stage:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "rejected",
                    "rejection_reason": hard_filter.rejection_reason,
                    "failed_criteria": hard_filter.failed_criteria,
                    "critical_missing_skills": hard_filter.critical_missing_skills,
                    "diagnostics": hard_filter.diagnostics,
                    "risk_flags": hard_filter.risk_flags,
                },
            )

        # ── Stage 3+: hard filter + German ATS scoring ───────────────────────
        scorer = get_scorer()
        score = scorer.score(
            entities=entities,
            job_description=job_description,
            job_skills=parsed_job_skills,
            semantic_sim=semantic_sim,
            raw_text=raw_text,
            quality_score=cv_result["quality_score"],
            job_requirements=requirements,
            hard_filter_result=hard_filter,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        confidence = get_confidence_engine().evaluate(
            parsing_quality=cv_result["quality_score"],
            entities=entities,
            raw_text=raw_text,
        )
        confidence_payload = {
            "score": confidence.score,
            "level": confidence.level,
            "issues": confidence.issues,
            "components": confidence.components,
        }
        hiring_decision = _hiring_decision(score.overall_score, confidence.score, score)

        return AnalyzeResponse(
            success=True,
            processing_time_ms=round(elapsed_ms, 1),
            status="accepted",
            overall_score=score.overall_score,
            grade=score.grade,
            confidence=confidence_payload,
            score_breakdown=score.score_breakdown,
            critical_missing_skills=score.critical_missing_skills,
            optional_missing_skills=score.optional_missing_skills,
            recruiter_insights=score.recruiter_insights,
            recruiter_decision=score.recruiter_decision,
            hiring_decision=hiring_decision,
            summary=score.summary,
            issues=score.issues,
            achievement_examples=score.achievement_analysis.get("examples", []),
            recommendations=score.recommendations,
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
                "languages": entities.languages,
                "summary": entities.summary,
                "total_experience_years": entities.total_experience_years,
            },
            skills_by_category=skills_by_category,
            score={
                "status": score.status,
                "rejection_reason": score.rejection_reason,
                "pass_to_next_stage": score.pass_to_next_stage,
                "overall_score": score.overall_score,
                "grade": score.grade,
                "fit_probability": score.fit_probability,
                "score_breakdown": score.score_breakdown,
                "skill_match_score": score.skill_match_score,
                "experience_score": score.experience_score,
                "education_score": score.education_score,
                "completeness_score": score.completeness_score,
                "semantic_similarity": score.semantic_similarity,
                "language_score": score.language_score,
                "achievement_score": score.achievement_score,
                "formatting_score": score.formatting_score,
                "recruiter_alignment_score": score.recruiter_alignment_score,
                "matched_skills": score.matched_skills,
                "missing_skills": score.missing_skills,
                "critical_missing_skills": score.critical_missing_skills,
                "optional_missing_skills": score.optional_missing_skills,
                "skill_gap_analysis": score.skill_gap_analysis,
                "issues": score.issues,
                "recruiter_insights": score.recruiter_insights,
                "recruiter_decision": score.recruiter_decision,
                "hiring_decision": hiring_decision,
                "summary": score.summary,
                "achievement_analysis": score.achievement_analysis,
                "confidence": confidence_payload,
                "job_requirements": {
                    "mandatory_skills": requirements.mandatory_skills,
                    "optional_skills": requirements.optional_skills,
                    "min_years_experience": requirements.min_years_experience,
                    "german_min_level": requirements.german_min_level,
                    "german_required": requirements.german_required,
                    "preferred_titles": requirements.preferred_titles,
                    "education_required": requirements.education_required,
                    "role_type": requirements.role_type,
                },
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
            requirements = get_job_analyzer().analyze(job_description, parsed_skills)
            hard_filter_resume = _to_hard_filter_resume(entities)
            hard_filter = get_hard_filter().evaluate(hard_filter_resume, requirements)

            if not hard_filter.pass_to_next_stage:
                results.append({
                    "filename": f.filename,
                    "name": entities.name,
                    "status": "rejected",
                    "rejection_reason": hard_filter.rejection_reason,
                    "failed_criteria": hard_filter.failed_criteria,
                    "critical_missing_skills": hard_filter.critical_missing_skills,
                    "diagnostics": hard_filter.diagnostics,
                    "risk_flags": hard_filter.risk_flags,
                    "overall_score": 0,
                    "grade": "D",
                    "fit_probability": 0,
                    "matched_skills_count": 0,
                    "total_experience_years": entities.total_experience_years,
                })
                continue

            score = scorer.score(
                entities,
                job_description,
                parsed_skills,
                semantic_sim,
                raw_text=cv_result["raw_text"],
                quality_score=cv_result["quality_score"],
                job_requirements=requirements,
                hard_filter_result=hard_filter,
            )
            confidence = get_confidence_engine().evaluate(
                parsing_quality=cv_result["quality_score"],
                entities=entities,
                raw_text=cv_result["raw_text"],
            )

            results.append({
                "filename": f.filename,
                "name": entities.name,
                "status": score.status,
                "rejection_reason": score.rejection_reason,
                "overall_score": score.overall_score,
                "grade": score.grade,
                "fit_probability": score.fit_probability,
                "matched_skills_count": len(score.matched_skills),
                "critical_missing_skills": score.critical_missing_skills,
                "hiring_decision": _hiring_decision(score.overall_score, confidence.score, score),
                "recruiter_decision": score.recruiter_decision,
                "summary": score.summary,
                "confidence": {
                    "score": confidence.score,
                    "level": confidence.level,
                    "issues": confidence.issues,
                    "components": confidence.components,
                },
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
