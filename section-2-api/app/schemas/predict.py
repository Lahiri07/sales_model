from typing import Any, List, Optional

from pydantic import BaseModel
from linear_model.processing.validation import LinearDataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


class MultipleLinearDataInputs(BaseModel):
    inputs: List[LinearDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "TV": 200.1,
                        "Radio": 21.6,
                        "Newspaper": 80.0
                    }
                ]
            }
        }


