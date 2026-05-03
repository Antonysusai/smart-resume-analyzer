"""
Confidence engine for German ATS simulation.

The score estimates how trustworthy the analysis is, based on document parsing
quality, extracted data completeness, and internal consistency.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ConfidenceResult:
    score: float
    level: str
    issues: List[str] = field(default_factory=list)
    components: Dict[str, float] = field(default_factory=dict)


class ConfidenceEngine:
    """Compute explainable confidence for the final ATS decision."""

    def evaluate(
        self,
        *,
        parsing_quality: float,
        entities: Any,
        raw_text: str = "",
    ) -> ConfidenceResult:
        quality = self._clamp(parsing_quality)
        completeness, completeness_issues = self._data_completeness(entities)
        consistency, consistency_issues = self._consistency_score(entities, raw_text)

        score = quality * 0.4 + completeness * 0.3 + consistency * 0.3
        issues = completeness_issues + consistency_issues
        if quality < 0.6:
            issues.append("Low OCR/parsing quality may reduce reliability")

        return ConfidenceResult(
            score=round(score, 4),
            level=self._level(score),
            issues=issues,
            components={
                "parsing_quality": round(quality, 4),
                "data_completeness": round(completeness, 4),
                "consistency_score": round(consistency, 4),
            },
        )

    def _data_completeness(self, entities: Any) -> tuple[float, List[str]]:
        checks = {
            "contact": bool(getattr(entities, "email", None) or getattr(entities, "phone", None)),
            "skills": bool(getattr(entities, "skills", [])),
            "experience": bool(getattr(entities, "experience", [])),
            "education": bool(getattr(entities, "education", [])),
            "languages": bool(getattr(entities, "languages", [])),
        }
        issues = [f"Missing {name} data" for name, present in checks.items() if not present]
        return sum(1 for present in checks.values() if present) / len(checks), issues

    def _consistency_score(self, entities: Any, raw_text: str) -> tuple[float, List[str]]:
        score = 1.0
        issues: List[str] = []
        text_length = len((raw_text or "").strip())

        if text_length < 300:
            score -= 0.25
            issues.append("Extracted CV text is unusually short")

        if getattr(entities, "total_experience_years", 0.0) > 0 and not getattr(entities, "experience", []):
            score -= 0.25
            issues.append("Experience years detected but no structured roles found")

        if len(getattr(entities, "skills", [])) == 0:
            score -= 0.25
            issues.append("No skills extracted")

        if getattr(entities, "total_experience_years", 0.0) > 35:
            score -= 0.2
            issues.append("Experience estimate appears unusually high")

        return self._clamp(score), issues

    def _level(self, score: float) -> str:
        if score >= 0.8:
            return "high"
        if score >= 0.6:
            return "medium"
        return "low"

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(float(value or 0.0), 1.0))
