"""
ML Module — Resume Scoring & Job-Fit Prediction
Combines rule-based scoring with XGBoost classifier and SHAP explanations.
"""

import numpy as np
import pandas as pd
import shap
import joblib
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from src.nlp.extractor import ResumeEntities, SKILLS_TAXONOMY


# ──────────────────────────────────────────────────────────────────────────────
# Score output
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ResumeScore:
    overall_score: float            # 0-100
    fit_probability: float          # 0-1  (ML model output)
    skill_match_score: float        # 0-100
    experience_score: float         # 0-100
    education_score: float          # 0-100
    completeness_score: float       # 0-100
    semantic_similarity: float      # 0-1
    grade: str                      # A / B / C / D
    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    shap_explanation: dict = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """Convert ResumeEntities + job description into a numeric feature vector."""

    FEATURE_NAMES = [
        "num_skills", "skill_match_ratio", "has_email", "has_phone",
        "has_linkedin", "has_github", "num_education", "has_degree",
        "num_experience", "total_exp_years", "has_summary", "num_certs",
        "semantic_sim", "completeness",
        # Per-category skill counts
        "skills_programming", "skills_ml_ai", "skills_data",
        "skills_cloud_devops", "skills_analytics", "skills_web",
    ]

    def build(
        self,
        entities: ResumeEntities,
        job_skills: list[str],
        semantic_sim: float,
    ) -> np.ndarray:
        job_skills_lower = {s.lower() for s in job_skills}
        candidate_skills_lower = {s.lower() for s in entities.skills}

        matched = candidate_skills_lower & job_skills_lower
        skill_match_ratio = len(matched) / max(len(job_skills_lower), 1)

        has_degree = any(
            any(deg in e.degree.lower() for deg in ["bachelor", "master", "phd", "b.tech", "m.tech", "be", "ms"])
            for e in entities.education
        )

        completeness = self._completeness(entities)

        # Per-category skill counts
        cat_counts = {}
        for cat, cat_skills in SKILLS_TAXONOMY.items():
            cat_counts[f"skills_{cat}"] = len(
                candidate_skills_lower & {s.lower() for s in cat_skills}
            )

        features = [
            len(entities.skills),
            skill_match_ratio,
            int(entities.email is not None),
            int(entities.phone is not None),
            int(entities.linkedin is not None),
            int(entities.github is not None),
            len(entities.education),
            int(has_degree),
            len(entities.experience),
            entities.total_experience_years,
            int(entities.summary is not None),
            len(entities.certifications),
            semantic_sim,
            completeness,
            cat_counts.get("skills_programming", 0),
            cat_counts.get("skills_ml_ai", 0),
            cat_counts.get("skills_data", 0),
            cat_counts.get("skills_cloud_devops", 0),
            cat_counts.get("skills_analytics", 0),
            cat_counts.get("skills_web", 0),
        ]
        return np.array(features, dtype=np.float32)

    def _completeness(self, entities: ResumeEntities) -> float:
        """What fraction of key resume fields are present."""
        fields = [
            entities.name, entities.email, entities.phone,
            entities.linkedin, entities.summary,
            bool(entities.skills), bool(entities.education), bool(entities.experience),
        ]
        return sum(1 for f in fields if f) / len(fields)


# ──────────────────────────────────────────────────────────────────────────────
# Resume scorer
# ──────────────────────────────────────────────────────────────────────────────

class ResumeScorer:
    """
    Hybrid scorer:
      - Rule-based sub-scores (skill match, completeness, experience, education)
      - XGBoost binary classifier (fit / not-fit) with SHAP explanations
      - Semantic similarity via sentence-transformers
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.fe = FeatureEngineer()
        self.model: Optional[Pipeline] = None
        self.explainer = None
        if model_path and Path(model_path).exists():
            self.load(model_path)

    # ──────────────────────────────────────────────────────────────────────────
    # Scoring
    # ──────────────────────────────────────────────────────────────────────────

    def score(
        self,
        entities: ResumeEntities,
        job_description: str,
        job_skills: list[str],
        semantic_sim: float,
    ) -> ResumeScore:
        """Produce a full ResumeScore for one candidate."""

        job_skills_lower = {s.lower() for s in job_skills}
        candidate_skills_lower = {s.lower() for s in entities.skills}
        matched_skills = sorted(candidate_skills_lower & job_skills_lower)
        missing_skills = sorted(job_skills_lower - candidate_skills_lower)

        # Sub-scores
        skill_score = self._skill_score(matched_skills, job_skills_lower)
        exp_score = self._experience_score(entities.total_experience_years)
        edu_score = self._education_score(entities.education)
        completeness = self.fe._completeness(entities) * 100

        # Feature vector
        feat = self.fe.build(entities, job_skills, semantic_sim).reshape(1, -1)

        # ML prediction
        if self.model:
            fit_prob = float(self.model.predict_proba(feat)[0][1])
            shap_exp = self._explain(feat)
        else:
            fit_prob = self._rule_based_fit(skill_score, exp_score, edu_score, semantic_sim)
            shap_exp = {}

        # Weighted overall score
        overall = (
            skill_score * 0.35
            + exp_score * 0.25
            + edu_score * 0.15
            + completeness * 0.10
            + semantic_sim * 100 * 0.15
        )
        overall = round(min(overall, 100), 1)

        grade = self._grade(overall)
        strengths, gaps, recs = self._insights(
            entities, matched_skills, missing_skills, exp_score, edu_score
        )

        return ResumeScore(
            overall_score=overall,
            fit_probability=round(fit_prob, 4),
            skill_match_score=round(skill_score, 1),
            experience_score=round(exp_score, 1),
            education_score=round(edu_score, 1),
            completeness_score=round(completeness, 1),
            semantic_similarity=round(semantic_sim, 4),
            grade=grade,
            strengths=strengths,
            gaps=gaps,
            matched_skills=matched_skills,
            missing_skills=missing_skills[:10],
            shap_explanation=shap_exp,
            recommendations=recs,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Sub-scores
    # ──────────────────────────────────────────────────────────────────────────

    def _skill_score(self, matched: list, required: set) -> float:
        if not required:
            return 50.0
        return min(len(matched) / len(required) * 100, 100)

    def _experience_score(self, years: float) -> float:
        # Score peaks at 5-8 years
        if years <= 0:
            return 10.0
        elif years < 2:
            return 40.0
        elif years < 5:
            return 65.0 + (years - 2) * 5
        elif years < 8:
            return 80.0 + (years - 5) * 3
        else:
            return 100.0

    def _education_score(self, education: list) -> float:
        if not education:
            return 30.0
        text = " ".join(e.degree.lower() for e in education)
        if any(d in text for d in ["phd", "doctorate", "ph.d"]):
            return 100.0
        elif any(d in text for d in ["master", "m.tech", "ms ", "msc"]):
            return 85.0
        elif any(d in text for d in ["bachelor", "b.tech", "be ", "bsc", "b.e"]):
            return 70.0
        return 50.0

    def _rule_based_fit(self, skill: float, exp: float, edu: float, sim: float) -> float:
        score = skill * 0.4 + exp * 0.3 + edu * 0.15 + sim * 100 * 0.15
        return round(min(score / 100, 1.0), 4)

    # ──────────────────────────────────────────────────────────────────────────
    # SHAP explanations
    # ──────────────────────────────────────────────────────────────────────────

    def _explain(self, feat: np.ndarray) -> dict:
        if self.explainer is None:
            return {}
        try:
            shap_values = self.explainer(feat)
            values = shap_values.values[0].tolist()
            return dict(zip(FeatureEngineer.FEATURE_NAMES, values))
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return {}

    # ──────────────────────────────────────────────────────────────────────────
    # Insights
    # ──────────────────────────────────────────────────────────────────────────

    def _insights(
        self,
        entities: ResumeEntities,
        matched: list,
        missing: list,
        exp_score: float,
        edu_score: float,
    ) -> tuple[list, list, list]:
        strengths, gaps, recs = [], [], []

        if len(matched) >= 5:
            strengths.append(f"Strong skill alignment — {len(matched)} required skills matched")
        if entities.total_experience_years >= 3:
            strengths.append(f"{entities.total_experience_years} years of relevant experience")
        if entities.github:
            strengths.append("GitHub profile present — demonstrates active coding")
        if entities.certifications:
            strengths.append(f"{len(entities.certifications)} certification(s) found")

        if missing:
            gaps.append(f"Missing key skills: {', '.join(missing[:5])}")
        if entities.total_experience_years < 2:
            gaps.append("Limited professional experience detected")
        if not entities.summary:
            gaps.append("No professional summary found")
        if not entities.linkedin:
            gaps.append("LinkedIn profile not listed")

        if missing:
            recs.append(f"Add projects or coursework covering: {', '.join(missing[:3])}")
        if not entities.summary:
            recs.append("Add a 3-4 sentence professional summary at the top of your resume")
        if not entities.github:
            recs.append("Include a link to your GitHub profile to showcase your work")
        if not entities.linkedin:
            recs.append("Add your LinkedIn URL to improve recruiter reachability")

        return strengths, gaps, recs

    # ──────────────────────────────────────────────────────────────────────────
    # Grade
    # ──────────────────────────────────────────────────────────────────────────

    def _grade(self, score: float) -> str:
        if score >= 80:
            return "A"
        elif score >= 65:
            return "B"
        elif score >= 50:
            return "C"
        else:
            return "D"

    # ──────────────────────────────────────────────────────────────────────────
    # Model persistence
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray, save_path: Optional[Path] = None):
        """Train the XGBoost scorer on labeled resume data."""
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
            )),
        ])
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring="roc_auc")
        logger.info(f"CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        self.model.fit(X, y)
        self.explainer = shap.Explainer(
            self.model.named_steps["clf"],
            feature_names=FeatureEngineer.FEATURE_NAMES,
        )
        if save_path:
            joblib.dump({"model": self.model, "explainer": self.explainer}, save_path)
            logger.info(f"Model saved to {save_path}")

    def load(self, path: Path):
        """Load a previously trained model."""
        data = joblib.load(path)
        self.model = data["model"]
        self.explainer = data.get("explainer")
        logger.info(f"Model loaded from {path}")


# Type hint fix
from typing import Optional
