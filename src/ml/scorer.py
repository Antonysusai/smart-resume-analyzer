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
from typing import Optional
from loguru import logger
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from src.nlp.extractor import ResumeEntities, SKILLS_TAXONOMY
from src.ml.achievement_detector import AchievementAnalysis, AchievementDetector
from src.ml.hard_filter import (
    GERMAN_LEVELS,
    HardFilterEngine,
    HardFilterResult,
    JobRequirements,
    ResumeEntities as HardFilterResumeEntities,
)
from src.ml.job_analyzer import JobDescriptionAnalyzer
from src.ml.recruiter_simulator import RecruiterSimulator


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
    status: str = "accepted"
    rejection_reason: str = ""
    pass_to_next_stage: bool = True
    score_breakdown: dict = field(default_factory=dict)
    language_score: float = 0.0
    achievement_score: float = 0.0
    formatting_score: float = 0.0
    recruiter_alignment_score: float = 0.0
    critical_missing_skills: list[str] = field(default_factory=list)
    optional_missing_skills: list[str] = field(default_factory=list)
    skill_gap_analysis: list[dict] = field(default_factory=list)
    recruiter_insights: dict = field(default_factory=dict)
    achievement_analysis: dict = field(default_factory=dict)
    issues: list[dict] = field(default_factory=list)
    summary: str = ""
    recruiter_decision: dict = field(default_factory=dict)


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
        self.job_analyzer = JobDescriptionAnalyzer()
        self.achievement_detector = AchievementDetector()
        self.hard_filter = HardFilterEngine()
        self.recruiter_simulator = RecruiterSimulator()
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
        raw_text: str = "",
        quality_score: float | None = None,
        job_requirements: JobRequirements | None = None,
        hard_filter_result: HardFilterResult | None = None,
    ) -> ResumeScore:
        """Produce a full ResumeScore for one candidate."""

        requirements = job_requirements or self.job_analyzer.analyze(job_description, job_skills)
        hard_filter_result = hard_filter_result or self.hard_filter.evaluate(
            self._to_hard_filter_resume(entities),
            requirements,
        )

        required_skills = requirements.mandatory_skills or job_skills
        job_skills_lower = {s.lower() for s in required_skills}
        optional_skills_lower = {s.lower() for s in requirements.optional_skills}
        candidate_skills_lower = {s.lower() for s in entities.skills}
        matched_skills = sorted(candidate_skills_lower & job_skills_lower)
        missing_skills = sorted(job_skills_lower - candidate_skills_lower)
        optional_missing = sorted(optional_skills_lower - candidate_skills_lower)

        # Sub-scores
        keyword_skill_score = self._skill_score(matched_skills, job_skills_lower)
        skill_score = min(100.0, keyword_skill_score * 0.6 + semantic_sim * 100 * 0.4)
        exp_score = self._experience_score(
            entities.total_experience_years,
            requirements.min_years_experience,
        )
        language_score = self._language_score(entities.languages, requirements)
        achievement_analysis = self.achievement_detector.analyze(raw_text or self._entities_text(entities))
        formatting_score = self._formatting_score(raw_text, entities, quality_score)
        edu_score = self._education_score(entities.education)
        completeness = self.fe._completeness(entities) * 100

        # Feature vector
        feat = self.fe.build(entities, required_skills, semantic_sim).reshape(1, -1)

        # ML prediction
        if self.model:
            fit_prob = float(self.model.predict_proba(feat)[0][1])
            shap_exp = self._explain(feat)
        else:
            fit_prob = self._rule_based_fit(skill_score, exp_score, language_score, semantic_sim)
            shap_exp = {}

        recruiter_insights = self.recruiter_simulator.scan(
            entities=entities,
            requirements=requirements,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            achievement_analysis=achievement_analysis,
            hard_filter_result=hard_filter_result,
        )
        recruiter_alignment_score = float(recruiter_insights.get("shortlist_probability", 0.0)) * 100

        weights = self._weights_for_role(requirements.role_type)
        # Weighted overall score. Skill score already includes semantic similarity.
        overall = (
            skill_score * weights["skills"]
            + exp_score * weights["experience"]
            + language_score * weights["language"]
            + achievement_analysis.score * weights["achievements"]
            + formatting_score * weights["formatting"]
            + recruiter_alignment_score * weights["recruiter_alignment"]
        )
        overall = round(min(overall, 100), 1)
        if not hard_filter_result.pass_to_next_stage:
            overall = min(overall, 49.0)
            fit_prob = min(fit_prob, 0.35)

        grade = self._grade(overall)
        strengths, gaps, recs = self._insights(
            entities, matched_skills, missing_skills, exp_score, edu_score,
            language_score, achievement_analysis, optional_missing, requirements.role_type
        )
        issues = self._issues(missing_skills, optional_missing, exp_score, language_score, achievement_analysis)
        summary = self._job_fit_summary(requirements.role_type, strengths, gaps)
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
            status="accepted" if hard_filter_result.pass_to_next_stage else "rejected",
            rejection_reason=hard_filter_result.rejection_reason,
            pass_to_next_stage=hard_filter_result.pass_to_next_stage,
            score_breakdown={
                "skills": round(skill_score, 1),
                "experience": round(exp_score, 1),
                "language": round(language_score, 1),
                "achievements": round(achievement_analysis.score, 1),
                "formatting": round(formatting_score, 1),
                "recruiter_alignment": round(recruiter_alignment_score, 1),
            },
            language_score=round(language_score, 1),
            achievement_score=round(achievement_analysis.score, 1),
            formatting_score=round(formatting_score, 1),
            recruiter_alignment_score=round(recruiter_alignment_score, 1),
            critical_missing_skills=hard_filter_result.critical_missing_skills[:10],
            optional_missing_skills=optional_missing[:10],
            skill_gap_analysis=self._skill_gap_analysis(missing_skills, optional_missing),
            recruiter_insights=recruiter_insights,
            recruiter_decision=recruiter_insights.get("recruiter_decision", {}),
            achievement_analysis={
                "score": achievement_analysis.score,
                "quantified_count": achievement_analysis.quantified_count,
                "action_verb_count": achievement_analysis.action_verb_count,
                "has_metrics": achievement_analysis.has_metrics,
                "examples": achievement_analysis.examples,
            },
            issues=issues,
            summary=summary,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Sub-scores
    # ──────────────────────────────────────────────────────────────────────────

    def _skill_score(self, matched: list, required: set) -> float:
        if not required:
            return 50.0
        return min(len(matched) / len(required) * 100, 100)

    def _experience_score(self, years: float, required_years: float = 0.0) -> float:
        if required_years > 0:
            if years <= 0:
                return 5.0
            ratio = years / required_years
            if ratio < 1:
                return max(10.0, ratio * 70)
            return min(100.0, 75 + min(ratio - 1, 1.0) * 25)

        # Score peaks at 5-8 years when the JD has no explicit threshold.
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
        # Normalize: lowercase and strip dots so "M.S." → "ms", "Ph.D." → "phd"
        text = " ".join(e.degree.lower().replace(".", "") for e in education)
        if any(d in text for d in ["phd", "doctorate", "ph d"]):
            return 100.0
        elif any(d in text for d in ["master", "mtech", "ms ", "ms\n", "msc", " ms"]):
            return 85.0
        elif any(d in text for d in ["bachelor", "btech", "bsc", "be ", "be\n", " be", "bs "]):
            return 70.0
        return 50.0

    def _language_score(self, languages: list[str], requirements: JobRequirements) -> float:
        candidate_level = HardFilterEngine.best_german_level(languages)
        if not requirements.german_required:
            return 100.0 if candidate_level else 75.0
        if not candidate_level:
            return 0.0
        required_level = requirements.german_min_level or "B1"
        candidate_rank = GERMAN_LEVELS.get(candidate_level, 0)
        required_rank = GERMAN_LEVELS.get(required_level, 1)
        return min(100.0, max(20.0, candidate_rank / required_rank * 100))

    def _formatting_score(
        self,
        raw_text: str,
        entities: ResumeEntities,
        quality_score: float | None = None,
    ) -> float:
        score = 35.0
        text = raw_text or ""
        lower = text.lower()
        section_groups = [
            ("experience", "work experience", "professional experience", "berufserfahrung"),
            ("education", "ausbildung", "studium"),
            ("skills", "technical skills", "kenntnisse", "fähigkeiten", "faehigkeiten"),
        ]
        for aliases in section_groups:
            if any(alias in lower for alias in aliases):
                score += 15
        if entities.email and entities.phone:
            score += 10
        if self._looks_reverse_chronological(entities):
            score += 10
        if quality_score is not None:
            score = score * 0.85 + quality_score * 100 * 0.15
        return min(100.0, score)

    def _looks_reverse_chronological(self, entities: ResumeEntities) -> bool:
        years = []
        for exp in entities.experience:
            date = exp.start_date or exp.end_date
            if not date:
                continue
            import re
            match = re.search(r"\d{4}", date)
            if match:
                years.append(int(match.group(0)))
        return len(years) < 2 or years == sorted(years, reverse=True)

    def _rule_based_fit(self, skill: float, exp: float, language: float, sim: float) -> float:
        score = skill * 0.35 + exp * 0.25 + language * 0.15 + sim * 100 * 0.15
        return round(min(score / 100, 1.0), 4)

    def _skill_gap_analysis(self, critical_missing: list[str], optional_missing: list[str]) -> list[dict]:
        gaps = [
            {"skill": skill, "type": "critical", "priority": "HIGH"}
            for skill in critical_missing
        ]
        gaps.extend(
            {"skill": skill, "type": "optional", "priority": "MEDIUM" if idx < 5 else "LOW"}
            for idx, skill in enumerate(optional_missing)
        )
        return gaps[:20]

    def _weights_for_role(self, role_type: str) -> dict[str, float]:
        base = {
            "skills": 0.30,
            "experience": 0.25,
            "language": 0.15,
            "achievements": 0.15,
            "formatting": 0.10,
            "recruiter_alignment": 0.05,
        }
        if role_type == "data":
            base.update({"skills": 0.32, "experience": 0.24, "achievements": 0.16, "formatting": 0.08})
        elif role_type == "backend":
            base.update({"skills": 0.34, "experience": 0.26, "formatting": 0.07, "achievements": 0.13})
        elif role_type == "frontend":
            base.update({"skills": 0.33, "experience": 0.23, "achievements": 0.12, "formatting": 0.12})
        return base

    def _issues(
        self,
        critical_missing: list[str],
        optional_missing: list[str],
        exp_score: float,
        language_score: float,
        achievement_analysis: AchievementAnalysis,
    ) -> list[dict]:
        issues = [
            {"type": "missing_skill", "skill": skill, "severity": "HIGH"}
            for skill in critical_missing
        ]
        issues.extend(
            {"type": "optional_missing_skill", "skill": skill, "severity": "MEDIUM"}
            for skill in optional_missing[:5]
        )
        if exp_score < 60:
            issues.append({"type": "experience", "severity": "HIGH"})
        if language_score < 60:
            issues.append({"type": "language", "severity": "HIGH"})
        if not achievement_analysis.has_metrics:
            issues.append({"type": "achievement", "severity": "MEDIUM"})
        return issues[:12]

    def _job_fit_summary(self, role_type: str, strengths: list[str], gaps: list[str]) -> str:
        role_label = {
            "data": "data role",
            "backend": "backend engineering role",
            "frontend": "frontend engineering role",
        }.get(role_type, "target role")
        strength = strengths[0] if strengths else "Candidate has some baseline alignment"
        gap = gaps[0].lower() if gaps else "no major blocker is visible"
        return f"Candidate shows fit for the {role_label}: {strength}. Main concern: {gap}."

    def _entities_text(self, entities: ResumeEntities) -> str:
        parts = [entities.summary or ""]
        parts.extend(exp.description or exp.title for exp in entities.experience)
        parts.extend(entities.skills)
        return "\n".join(parts)

    def _to_hard_filter_resume(self, entities: ResumeEntities) -> HardFilterResumeEntities:
        return HardFilterResumeEntities(
            skills=entities.skills,
            total_experience_years=entities.total_experience_years,
            languages=self._languages_to_dict(entities.languages),
            job_titles=[exp.title for exp in entities.experience if exp.title],
            education=" ".join(edu.degree for edu in entities.education if edu.degree) or None,
        )

    def _languages_to_dict(self, languages: list[str]) -> dict[str, str]:
        import re

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
        language_score: float = 75.0,
        achievement_analysis: AchievementAnalysis | None = None,
        optional_missing: list[str] | None = None,
        role_type: str = "general",
    ) -> tuple[list, list, list]:
        strengths, gaps, recs = [], [], []
        optional_missing = optional_missing or []

        if len(matched) >= 5:
            strengths.append(f"Strong skill alignment — {len(matched)} required skills matched")
        if entities.total_experience_years >= 3:
            strengths.append(f"{entities.total_experience_years} years of relevant experience")
        if entities.github:
            strengths.append("GitHub profile present — demonstrates active coding")
        if entities.certifications:
            strengths.append(f"{len(entities.certifications)} certification(s) found")
        if language_score >= 85:
            strengths.append("German language requirement is covered")
        if achievement_analysis and achievement_analysis.quantified_count:
            strengths.append(f"{achievement_analysis.quantified_count} quantified impact statement(s) found")

        if missing:
            gaps.append(f"Missing key skills: {', '.join(missing[:5])}")
        if entities.total_experience_years < 2:
            gaps.append("Limited professional experience detected")
        if not entities.summary:
            gaps.append("No professional summary found")
        if not entities.linkedin:
            gaps.append("LinkedIn profile not listed")
        if language_score < 60:
            gaps.append("German language evidence is weak or missing")
        if achievement_analysis and achievement_analysis.quantified_count == 0:
            gaps.append("No measurable achievements detected")

        if missing:
            recs.append(f"Add evidence for required skill(s): {', '.join(missing[:3])}")
        if optional_missing:
            recs.append(f"Add {optional_missing[0]} to improve alignment with the job requirements")
        if role_type == "data" and "power bi" in optional_missing + missing:
            recs.append("Add Power BI dashboards or reporting examples to match data role expectations")
        if role_type == "backend" and not entities.github:
            recs.append("Add a backend project repository showing APIs, databases, or deployment work")
        if role_type == "frontend" and not entities.github:
            recs.append("Add a portfolio or GitHub link with frontend UI examples")
        if not entities.summary:
            recs.append("Add a 2-3 sentence profile summary tailored to this role")
        if not entities.github:
            recs.append("Include a link to your GitHub profile to showcase your work")
        if not entities.linkedin:
            recs.append("Add your LinkedIn URL to improve recruiter reachability")
        if language_score < 60:
            recs.append("Add German language proficiency using CEFR format, e.g. German B2 or C1")
        if achievement_analysis and achievement_analysis.quantified_count == 0:
            recs.append("Include measurable achievements, e.g. reduced processing time by 20% or automated 10 hours per week")

        return strengths, gaps, list(dict.fromkeys(recs))[:8]

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
