from pathlib import Path
from typing import Dict, List, Sequence

from pydantic import BaseModel
from strictyaml import YAML, load
import sys
sys.path.insert(0, r"C:\Users\subhol\deploying-credit model\section-1-packaging")
import linear_model

# Project Directories
PACKAGE_ROOT = Path(linear_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    test_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    features: Sequence[str]
    test_size: float
    random_state: int
    alpha: float


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


def load_config() -> Config:
    """Load and validate config from YAML using strictyaml."""
    with open(CONFIG_FILE_PATH, "r") as file:
        parsed_config: YAML = load(file.read())

    return Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data)
    )


# Create a global config object to import elsewhere
config = load_config()
print(config)
