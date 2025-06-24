import sys
from pydantic import AnyHttpUrl, BaseSettings

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "House Price Prediction API"

    class Config:
        case_sensitive = True

settings = Settings()
