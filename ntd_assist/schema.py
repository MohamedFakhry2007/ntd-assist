from pydantic import BaseModel
from typing import Literal


class ClinicalAnalysis(BaseModel):
    model_config = {"extra": "forbid", "validate_assignment": True}

    detected_disease: Literal[
        "Malaria", "Leishmaniasis", "Schistosomiasis",
        "Filariasis", "Trypanosomiasis", "Onchocerciasis",
        "Loiasis", "Negative for Parasites", "Unclear"
    ]
    severity: Literal["Scanty (+)", "Moderate (++)", "Heavy (+++)", "N/A"]
    morphology_proof: str
    confidence: Literal["High", "Medium", "Low"]
    findings: str
    recommendation: str
    species: str = "Unknown"
    observed_background: str = ""
    observed_organisms: str = ""
    organism_location: str = ""
