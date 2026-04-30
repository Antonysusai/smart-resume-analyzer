"""
NLP Module — Resume Entity Extraction & Semantic Understanding
Uses spaCy NER, regex patterns, and HuggingFace zero-shot classification.
"""

import re
import spacy
import nltk
from loguru import logger
from dataclasses import dataclass, field
from typing import Optional
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Education:
    degree: str
    institution: str
    year: Optional[str] = None
    field: Optional[str] = None


@dataclass
class Experience:
    title: str
    company: str
    duration: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: str = ""


@dataclass
class ResumeEntities:
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    location: Optional[str] = None
    skills: list[str] = field(default_factory=list)
    education: list[Education] = field(default_factory=list)
    experience: list[Experience] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    summary: Optional[str] = None
    total_experience_years: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Skills taxonomy
# ──────────────────────────────────────────────────────────────────────────────

SKILLS_TAXONOMY = {
    "programming": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "kotlin", "swift", "r", "scala", "matlab", "julia", "ruby", "php",
    ],
    "ml_ai": [
        "machine learning", "deep learning", "nlp", "computer vision", "pytorch",
        "tensorflow", "keras", "scikit-learn", "xgboost", "lightgbm", "huggingface",
        "transformers", "bert", "gpt", "llm", "reinforcement learning", "opencv",
        "yolo", "diffusion models",
    ],
    "data": [
        "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "spark", "hadoop", "kafka", "airflow", "dbt", "pandas", "numpy",
        "data engineering", "etl", "data pipeline",
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd",
        "github actions", "jenkins", "ansible", "linux", "bash", "mlops",
    ],
    "analytics": [
        "power bi", "tableau", "looker", "excel", "statistics", "a/b testing",
        "hypothesis testing", "regression", "forecasting",
    ],
    "web": [
        "fastapi", "django", "flask", "react", "angular", "vue", "node.js",
        "rest api", "graphql", "microservices",
    ],
}

ALL_SKILLS = {skill for skills in SKILLS_TAXONOMY.values() for skill in skills}


# ──────────────────────────────────────────────────────────────────────────────
# Main NLP Extractor
# ──────────────────────────────────────────────────────────────────────────────

class ResumeNLPExtractor:
    """
    Multi-stage NLP pipeline for structured resume parsing:
      1. spaCy NER for names, locations, organisations
      2. Regex for contacts, dates
      3. Skill taxonomy matching
      4. Zero-shot classification for section detection
      5. Sentence-transformer embeddings for semantic search
    """

    def __init__(self, spacy_model: str = "en_core_web_lg",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        logger.info("Loading NLP models...")
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(f"{spacy_model} not found, falling back to en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.embedder = SentenceTransformer(embedding_model)
        self._zero_shot = None  # lazy load — heavy model
        logger.info("NLP models loaded.")

    @property
    def zero_shot(self):
        if self._zero_shot is None:
            self._zero_shot = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1,
            )
        return self._zero_shot

    # ──────────────────────────────────────────────────────────────────────────
    # Main extraction method
    # ──────────────────────────────────────────────────────────────────────────

    def extract(self, raw_text: str) -> ResumeEntities:
        """Parse raw resume text into structured ResumeEntities."""
        doc = self.nlp(raw_text)
        entities = ResumeEntities()

        entities.name = self._extract_name(doc, raw_text)
        entities.email = self._extract_email(raw_text)
        entities.phone = self._extract_phone(raw_text)
        entities.linkedin = self._extract_url(raw_text, "linkedin")
        entities.github = self._extract_url(raw_text, "github")
        entities.location = self._extract_location(doc)
        entities.skills = self._extract_skills(raw_text)
        entities.education = self._extract_education(raw_text, doc)
        entities.experience = self._extract_experience(raw_text, doc)
        entities.certifications = self._extract_certifications(raw_text)
        entities.summary = self._extract_summary(raw_text)
        entities.total_experience_years = self._estimate_experience_years(
            entities.experience
        )

        return entities

    # ──────────────────────────────────────────────────────────────────────────
    # Contact extraction
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_name(self, doc, text: str) -> Optional[str]:
        # Try spaCy PERSON entity first
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                if 2 <= len(name.split()) <= 4:
                    return name
        # Fallback: first non-empty line that looks like a name
        for line in text.split("\n")[:5]:
            line = line.strip()
            if line and re.match(r"^[A-Z][a-z]+(?: [A-Z][a-z]+)+$", line):
                return line
        return None

    def _extract_email(self, text: str) -> Optional[str]:
        pattern = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
        match = re.search(pattern, text)
        return match.group(0).lower() if match else None

    def _extract_phone(self, text: str) -> Optional[str]:
        pattern = r"(?:\+?\d{1,3}[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}"
        match = re.search(pattern, text)
        return match.group(0).strip() if match else None

    def _extract_url(self, text: str, platform: str) -> Optional[str]:
        pattern = rf"(?:https?://)?(?:www\.)?{platform}\.com/[\w\-/]+"
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(0) if match else None

    def _extract_location(self, doc) -> Optional[str]:
        for ent in doc.ents:
            if ent.label_ in {"GPE", "LOC"}:
                return ent.text.strip()
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Skills extraction
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_skills(self, text: str) -> list[str]:
        """Match against skills taxonomy using case-insensitive search."""
        text_lower = text.lower()
        found = set()
        for skill in ALL_SKILLS:
            # Use word boundary matching for short skills
            pattern = rf"\b{re.escape(skill)}\b"
            if re.search(pattern, text_lower):
                found.add(skill)
        return sorted(found)

    def get_skills_by_category(self, skills: list[str]) -> dict[str, list[str]]:
        """Group extracted skills into taxonomy categories."""
        categorized = {}
        for category, cat_skills in SKILLS_TAXONOMY.items():
            matched = [s for s in skills if s in cat_skills]
            if matched:
                categorized[category] = matched
        return categorized

    # ──────────────────────────────────────────────────────────────────────────
    # Education extraction
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_education(self, text: str, doc) -> list[Education]:
        degree_patterns = [
            r"(?:B\.?E|B\.?Tech|B\.?Sc|B\.?S|M\.?S|M\.?Tech|M\.?Sc|MBA|Ph\.?D|Bachelor|Master|Doctor)[^\n,]*",
        ]
        year_pattern = r"\b(19|20)\d{2}\b"
        educations = []

        for pattern in degree_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                line = match.group(0).strip()
                year_match = re.search(year_pattern, text[match.start():match.start()+200])
                # Find nearby org entity
                institution = None
                for ent in doc.ents:
                    if ent.label_ == "ORG" and abs(ent.start_char - match.start()) < 300:
                        institution = ent.text
                        break
                educations.append(Education(
                    degree=line[:80],
                    institution=institution or "Unknown",
                    year=year_match.group(0) if year_match else None,
                ))

        return educations[:5]  # cap at 5

    # ──────────────────────────────────────────────────────────────────────────
    # Experience extraction
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_experience(self, text: str, doc) -> list[Experience]:
        """Extract job titles and companies using NER + regex patterns."""
        title_patterns = [
            r"(?:Senior|Junior|Lead|Principal|Staff|Associate)?\s*"
            r"(?:Software|Data|ML|AI|Backend|Frontend|Full.?Stack|DevOps|Cloud)?\s*"
            r"(?:Engineer|Developer|Scientist|Analyst|Architect|Manager|Consultant|Intern)",
        ]
        experiences = []
        for pattern in title_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                title = match.group(0).strip()
                # Find nearest ORG entity
                company = None
                for ent in doc.ents:
                    if ent.label_ == "ORG" and abs(ent.start_char - match.start()) < 400:
                        company = ent.text
                        break
                # Find date range near the match
                date_pattern = r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{4})\s*[–\-—to]+\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{4}|Present|Current)"
                date_match = re.search(date_pattern, text[match.start():match.start()+300], re.IGNORECASE)
                experiences.append(Experience(
                    title=title,
                    company=company or "Unknown",
                    start_date=date_match.group(1) if date_match else None,
                    end_date=date_match.group(2) if date_match else None,
                ))

        # Deduplicate by title
        seen = set()
        unique = []
        for exp in experiences:
            key = (exp.title.lower(), exp.company.lower())
            if key not in seen:
                seen.add(key)
                unique.append(exp)

        return unique[:8]

    # ──────────────────────────────────────────────────────────────────────────
    # Certifications
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_certifications(self, text: str) -> list[str]:
        cert_keywords = [
            "certified", "certification", "certificate", "aws certified",
            "azure certified", "google cloud", "pmp", "cpa", "cfa",
            "comptia", "cissp", "ccna", "scrum master",
        ]
        lines = text.split("\n")
        certs = []
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in cert_keywords):
                clean = line.strip()
                if 5 < len(clean) < 150:
                    certs.append(clean)
        return list(set(certs))[:10]

    # ──────────────────────────────────────────────────────────────────────────
    # Summary extraction
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_summary(self, text: str) -> Optional[str]:
        """Extract the professional summary / objective section."""
        pattern = r"(?:summary|objective|profile|about me)[:\s]*\n?(.*?)(?:\n\n|\Z)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            summary = match.group(1).strip()
            return summary[:500] if summary else None
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Experience estimation
    # ──────────────────────────────────────────────────────────────────────────

    def _estimate_experience_years(self, experiences: list[Experience]) -> float:
        """Rough total years from date ranges found in experience entries."""
        import datetime
        current_year = datetime.datetime.now().year
        total = 0.0

        for exp in experiences:
            if exp.start_date and exp.end_date:
                try:
                    start_year = int(re.search(r"\d{4}", exp.start_date).group())
                    end_str = exp.end_date
                    if re.search(r"present|current", end_str, re.IGNORECASE):
                        end_year = current_year
                    else:
                        end_year = int(re.search(r"\d{4}", end_str).group())
                    total += max(0, end_year - start_year)
                except (AttributeError, ValueError):
                    pass

        return round(total, 1)

    # ──────────────────────────────────────────────────────────────────────────
    # Semantic embedding
    # ──────────────────────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> np.ndarray:
        """Return sentence embedding for semantic similarity tasks."""
        return self.embedder.encode(text, convert_to_numpy=True)

    def semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts (0 to 1)."""
        emb_a = self.embed_text(text_a)
        emb_b = self.embed_text(text_b)
        cosine = np.dot(emb_a, emb_b) / (
            np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-9
        )
        return float(np.clip(cosine, 0, 1))
