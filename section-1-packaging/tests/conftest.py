import pytest
import pandas as pd
from pathlib import Path
import linear_model  # Ensure this works with your sys.path setup

# Locate datasets directory
DATASET_DIR = Path(linear_model.__file__).resolve().parent / "datasets"

@pytest.fixture
def sample_input_data():
    """
    Load sample test data from test.csv.
    Assumes test.csv includes both features and the target 'Sales'.
    """
    test_file_path = DATASET_DIR / "test.csv"

    if not test_file_path.exists():
        raise FileNotFoundError(f"Test file not found at {test_file_path}")

    df = pd.read_csv(test_file_path)

    return df