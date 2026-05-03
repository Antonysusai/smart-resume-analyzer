"""
Job description analyzer for German ATS simulation.

Extracts deterministic hiring requirements from free-text job ads while keeping
explicit form-provided skills as the strongest signal.
"""

import re

from src.ml.hard_filter import JobRequirements
from src.nlp.extractor import ALL_SKILLS


class JobDescriptionAnalyzer:
    """Small, fast rule parser for ATS-style requirements."""

    MUST_PATTERNS = (
        "required", "must have", "must-have", "mandatory", "essential",
        "erforderlich", "voraussetzung", "muss", "zwingend",
    )
    NICE_PATTERNS = (
        "nice to have", "preferred", "plus", "bonus", "advantage",
        "wunsch", "von vorteil", "optional",
    )

    def analyze(self, job_description: str, explicit_required_skills: list[str] | None = None) -> JobRequirements:
        text = job_description or ""
        explicit = [s.strip().lower() for s in explicit_required_skills or [] if s.strip()]
        found_skills = self._extract_skills(text)
        mandatory = set(explicit)
        optional = set()

        for skill in found_skills:
            context = self._context_window(text, skill)
            if any(marker in context for marker in self.NICE_PATTERNS):
                optional.add(skill)
            elif any(marker in context for marker in self.MUST_PATTERNS):
                mandatory.add(skill)
            elif not explicit:
                mandatory.add(skill)
            else:
                optional.add(skill)

        optional -= mandatory
        german_level = self.extract_german_level(text)
        language_required = german_level is not None or bool(
            re.search(r"\b(german|deutsch|deutsche sprache|german language)\b", text, re.IGNORECASE)
        )

        return JobRequirements(
            mandatory_skills=sorted(mandatory),
            optional_skills=sorted(optional),
            min_years_experience=self.extract_min_experience(text),
            german_min_level=german_level,
            german_required=language_required,
            preferred_titles=self.extract_preferred_titles(text),
            education_required=self.extract_education_requirement(text),
            role_type=self.detect_role(text),
        )

    def _extract_skills(self, text: str) -> list[str]:
        text_lower = text.lower()
        found = {
            skill for skill in ALL_SKILLS
            if re.search(rf"\b{re.escape(skill.lower())}\b", text_lower)
        }
        return sorted(found)

    def _context_window(self, text: str, term: str, radius: int = 90) -> str:
        match = re.search(re.escape(term), text, re.IGNORECASE)
        if not match:
            return ""
        start = max(0, match.start() - radius)
        end = min(len(text), match.end() + radius)
        return text[start:end].lower()

    @staticmethod
    def extract_min_experience(text: str) -> float:
        patterns = [
            r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs|jahre|jahren)\s+(?:of\s+)?(?:experience|erfahrung)",
            r"(?:minimum|min\.?|at least|mindestens)\s+(\d+(?:\.\d+)?)\s*(?:years|yrs|jahre|jahren)",
            r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs|jahre|jahren)",
        ]
        values = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    values.append(float(match.group(1)))
                except ValueError:
                    pass
        return min(values) if values else 0.0

    @staticmethod
    def extract_german_level(text: str) -> str | None:
        level_match = re.search(r"\b(A1|A2|B1|B2|C1|C2)\b", text, re.IGNORECASE)
        if level_match and re.search(r"\b(german|deutsch)\b", text, re.IGNORECASE):
            return level_match.group(1).upper()
        if re.search(r"\b(native|muttersprache|muttersprachlich)\b.*\b(german|deutsch)\b|\b(german|deutsch)\b.*\b(native|muttersprache|muttersprachlich)\b", text, re.IGNORECASE):
            return "NATIVE"
        if re.search(r"\b(fluent|verhandlungssicher)\b.*\b(german|deutsch)\b|\b(german|deutsch)\b.*\b(fluent|verhandlungssicher)\b", text, re.IGNORECASE):
            return "C1"
        if re.search(r"\b(good|gute|business|working)\b.*\b(german|deutsch)\b|\b(german|deutsch)\b.*\b(good|gute|business|working)\b", text, re.IGNORECASE):
            return "B2"
        return None

    @staticmethod
    def extract_preferred_titles(text: str) -> list[str]:
        title_patterns = [
            r"\b(?:senior|junior|lead|principal|staff)?\s*(?:software|data|ml|ai|backend|frontend|full.?stack|devops|cloud)?\s*(?:engineer|developer|scientist|analyst|architect|manager|consultant)\b",
            r"\b(?:softwareentwickler|datenanalyst|data scientist|entwickler|ingenieur|berater|projektmanager)\b",
        ]
        titles = set()
        for pattern in title_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                title = " ".join(match.group(0).split()).lower()
                if len(title) > 5:
                    titles.add(title)
        return sorted(titles)[:5]

    @staticmethod
    def extract_education_requirement(text: str) -> str | None:
        lower = text.lower()
        if re.search(r"\b(phd|doctorate|promotion|doktor)\b", lower):
            return "phd"
        if re.search(r"\b(master|m\.sc|msc|m\.tech)\b", lower):
            return "master"
        if re.search(r"\b(bachelor|b\.sc|bsc|b\.tech|degree|abschluss|studium)\b", lower):
            return "bachelor"
        return None

    @staticmethod
    def detect_role(text: str) -> str:
        lower = text.lower()
        role_signals = {
            "data": [
                "data scientist", "data analyst", "machine learning", "ml engineer",
                "analytics", "etl", "power bi", "tableau", "pandas", "sql",
            ],
            "backend": [
                "backend", "api", "fastapi", "django", "flask", "microservices",
                "distributed systems", "python developer", "java developer",
            ],
            "frontend": [
                "frontend", "react", "angular", "vue", "typescript", "javascript",
                "ui", "web app",
            ],
        }
        scores = {
            role: sum(1 for signal in signals if signal in lower)
            for role, signals in role_signals.items()
        }
        best_role = max(scores, key=scores.get)
        return best_role if scores[best_role] > 0 else "general"
