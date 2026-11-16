#Imports
import pickle
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

#Defining the PlantData class to validate the input data for the API:
class PlantData(BaseModel):
    fuel: Literal['gas', 'oil', 'lng_only', 'multi_fuel']
    chp: Literal['yes', 'no', 'not_found']
    capacity_mw: float
    start_year: int = 0
    latitude: float
    longitude: float

#Defining the PredictionResponse class to structure the output of the API:
class PredictionResponse(BaseModel):
    technology: str
    probability: dict[str, float]

#App proper:
app = FastAPI(title="Gas and Oil Plant Technology Prediction API")

folder = Path(__file__).parent

with open(folder / 'model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

#Defining the API endpoint:
@app.post('/predict')
def predict(plant: PlantData) -> PredictionResponse:
    X = dv.transform([plant.model_dump()])
    probs = model.predict_proba(X)[0]

    probabilities = {c: float(p) for c, p in zip(model.classes_, probs)}
    best = max(probabilities, key=probabilities.get)

    return PredictionResponse(technology=best, probability=probabilities)

