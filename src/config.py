"""
Smart Resume Analyzer — Application Settings
"""
from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Smart Resume Analyzer"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Paths
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "data" / "models"
    SAMPLES_DIR: Path = BASE_DIR / "data" / "samples"

    # NLP Models
    SPACY_MODEL: str = "en_core_web_lg"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    ZERO_SHOT_MODEL: str = "facebook/bart-large-mnli"

    # CV
    DPI: int = 300
    MAX_FILE_SIZE_MB: int = 10

    # ML
    ML_MODEL_PATH: Path = BASE_DIR / "data" / "models" / "resume_scorer.joblib"
    SCORE_THRESHOLD: float = 0.6

    # CORS
    ALLOWED_ORIGINS: list[str] = ["http://localhost:8501", "http://localhost:3000"]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
