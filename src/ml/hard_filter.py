"""
Production-grade deterministic hard filter engine for the German ATS simulator.

This module intentionally uses only standard-library code. It is the explainable
gatekeeper that rejects candidates before scoring when critical requirements are
not met.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


GERMAN_LEVEL_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2", "NATIVE"]
GERMAN_LEVELS = {level: index + 1 for index, level in enumerate(GERMAN_LEVEL_ORDER)}


@dataclass
class JobRequirements:
    mandatory_skills: List[str] = field(default_factory=list)
    optional_skills: List[str] = field(default_factory=list)
    min_years_experience: float = 0.0
    german_required: bool = False
    german_min_level: Optional[str] = None
    preferred_titles: List[str] = field(default_factory=list)
    education_required: Optional[str] = None
    mode: Literal["strict", "flexible"] = "strict"
    role_type: str = "general"


@dataclass
class ResumeEntities:
    skills: List[str] = field(default_factory=list)
    total_experience_years: float = 0.0
    languages: Dict[str, str] = field(default_factory=dict)
    job_titles: List[str] = field(default_factory=list)
    education: Optional[str] = None


@dataclass
class HardFilterResult:
    pass_to_next_stage: bool
    rejection_reason: str = ""
    failed_criteria: List[str] = field(default_factory=list)
    critical_missing_skills: List[str] = field(default_factory=list)
    diagnostics: Dict[str, object] = field(default_factory=dict)
    risk_flags: List[str] = field(default_factory=list)


def normalize(text: str) -> str:
    """Normalize text for deterministic case-insensitive comparisons."""
    return " ".join((text or "").strip().lower().split())


def fuzzy_match(skill: str, skills_list: List[str]) -> bool:
    """Match exact, substring, or meaningful token overlap skill variants."""
    normalized_skill = normalize(skill)
    if not normalized_skill:
        return False

    for candidate in skills_list or []:
        normalized_candidate = normalize(candidate)
        if not normalized_candidate:
            continue
        if normalized_skill == normalized_candidate:
            return True
        if normalized_skill in normalized_candidate or normalized_candidate in normalized_skill:
            return True
        if _token_overlap(normalized_skill, normalized_candidate) >= 0.5:
            return True

    return False


def _token_overlap(left: str, right: str) -> float:
    left_tokens = {token for token in left.replace("/", " ").replace("-", " ").split() if token}
    right_tokens = {token for token in right.replace("/", " ").replace("-", " ").split() if token}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))


def title_similarity(candidate_titles: List[str], preferred_titles: List[str]) -> float:
    """
    Return substring-match ratio for preferred titles.

    If no title constraint exists, the filter should not block the candidate.
    """
    if not preferred_titles:
        return 1.0
    if not candidate_titles:
        return 0.0

    matches = 0
    for preferred_title in preferred_titles:
        if fuzzy_match(preferred_title, candidate_titles):
            matches += 1

    return matches / max(len(preferred_titles), 1)


def level_to_index(level: str) -> int:
    """Map CEFR level to comparable index. Unknown levels return -1."""
    normalized_level = (level or "").strip().upper()
    if normalized_level in GERMAN_LEVEL_ORDER:
        return GERMAN_LEVEL_ORDER.index(normalized_level)
    return -1


def is_level_sufficient(candidate_level: str, required_level: str) -> bool:
    """Return True when candidate CEFR level is equal to or above requirement."""
    candidate_index = level_to_index(candidate_level)
    required_index = level_to_index(required_level)
    return candidate_index >= required_index and candidate_index >= 0 and required_index >= 0


class HardFilterEngine:
    """Evaluate non-negotiable requirements before scoring or ML inference."""

    def __init__(self, mode: Literal["strict", "flexible"] = "strict"):
        self.mode = mode

    def evaluate(
        self,
        resume: ResumeEntities,
        req: JobRequirements,
        mode: Optional[Literal["strict", "flexible"]] = None,
    ) -> HardFilterResult:
        active_mode = mode or req.mode or self.mode
        failures: List[tuple[str, int]] = []
        critical_missing_skills: List[str] = []
        diagnostics: Dict[str, object] = {}
        risk_flags: List[str] = []

        for skill in req.mandatory_skills:
            if not fuzzy_match(skill, resume.skills):
                critical_missing_skills.append(skill)

        missing_skill_count = len(critical_missing_skills)
        diagnostics["missing_mandatory_skill_count"] = missing_skill_count
        if active_mode == "flexible" and missing_skill_count == 1:
            risk_flags.append("Borderline skills match: one mandatory skill missing in flexible mode")
        elif missing_skill_count:
            failures.append(("Missing required skills", 100))

        experience_gap = max(0.0, round(req.min_years_experience - resume.total_experience_years, 2))
        diagnostics["experience_gap_years"] = experience_gap
        if experience_gap > 0:
            if active_mode == "flexible" and experience_gap <= 0.5:
                risk_flags.append(f"Borderline experience: {experience_gap:g} years below requirement")
            else:
                failures.append(("Insufficient experience", 90))

        if req.german_required:
            german_level = self._get_language_level(resume.languages, "german")
            diagnostics["candidate_german_level"] = german_level
            diagnostics["required_german_level"] = req.german_min_level

            if not german_level:
                diagnostics["german_gap"] = f"missing < {req.german_min_level or 'required'}"
                failures.append(("German language missing", 95))
            elif req.german_min_level:
                if not is_level_sufficient(german_level, req.german_min_level):
                    diagnostics["german_gap"] = f"{german_level} < {req.german_min_level}"
                    failures.append(("German level below requirement", 85))
                elif level_to_index(german_level) == level_to_index(req.german_min_level):
                    risk_flags.append("German level exactly meets minimum requirement")

        title_score = title_similarity(resume.job_titles, req.preferred_titles)
        diagnostics["title_similarity_score"] = round(title_score, 3)
        if req.preferred_titles and title_score < 0.3:
            failures.append(("Job title mismatch", 60))

        if req.education_required:
            if not resume.education:
                failures.append(("Education mismatch", 70))
            elif not education_matches(resume.education, req.education_required):
                failures.append(("Education mismatch", 70))

        if failures:
            failures.sort(key=lambda item: item[1], reverse=True)
            failed_criteria = [failure for failure, _priority in failures]
            return HardFilterResult(
                pass_to_next_stage=False,
                rejection_reason=failed_criteria[0],
                failed_criteria=failed_criteria,
                critical_missing_skills=critical_missing_skills,
                diagnostics=diagnostics,
                risk_flags=risk_flags,
            )

        return HardFilterResult(
            pass_to_next_stage=True,
            diagnostics=diagnostics,
            risk_flags=risk_flags,
        )

    @staticmethod
    def _get_language_level(languages: Dict[str, str], language_name: str) -> Optional[str]:
        normalized_language_name = normalize(language_name)
        for language, level in (languages or {}).items():
            if normalize(language) == normalized_language_name:
                return (level or "").strip().upper() or None
        return None

    @staticmethod
    def best_german_level(languages: "Dict[str, str] | List[str]") -> Optional[str]:
        """Compatibility helper for richer NLP outputs that store languages as strings."""
        if isinstance(languages, dict):
            return HardFilterEngine._get_language_level(languages, "german")

        best_level: Optional[str] = None
        for language in languages or []:
            upper = str(language).upper()
            if "GERMAN" not in upper and "DEUTSCH" not in upper:
                continue
            for level in GERMAN_LEVEL_ORDER:
                if level in upper and level_to_index(level) > level_to_index(best_level or ""):
                    best_level = level
        return best_level


EDUCATION_EQUIVALENTS = {
    "bachelor": {"bachelor", "b.tech", "btech", "b.sc", "bsc", "be", "b.e"},
    "master": {"master", "m.sc", "msc", "m.tech", "mtech", "ms", "mba"},
    "phd": {"phd", "ph.d", "doctorate", "doctor", "doktor"},
}


def education_matches(candidate_education: str, required_education: str) -> bool:
    candidate = normalize(candidate_education)
    required = normalize(required_education)
    accepted_terms = EDUCATION_EQUIVALENTS.get(required, {required})
    return any(term in candidate for term in accepted_terms)
