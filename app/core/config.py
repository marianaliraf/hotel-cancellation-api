from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_path: Path = ROOT / "models" / "modelo_cancelamento_hotel.joblib"
    model_config = SettingsConfigDict(env_prefix="HOTEL_", env_file=".env", extra="ignore")


settings = Settings()
