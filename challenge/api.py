import fastapi
import pandas as pd
from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import List

# Assuming DelayModel is in the same directory or properly imported
try:
    from .model import DelayModel
except:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model import DelayModel


app = fastapi.FastAPI()


# Custom exception handler to convert 422 to 400
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Convert validation errors (422) to bad request errors (400).
    This is needed to match the expected test behavior.
    """
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors()[0]["msg"]},
    )

# Initialize the model
model = DelayModel()


# Valid airlines (from your dataset)
VALID_AIRLINES = [
    "Aerolineas Argentinas",
    "Aeromexico",
    "Air Canada",
    "Air France",
    "Alitalia",
    "American Airlines",
    "Austral",
    "Avianca",
    "British Airways",
    "Copa Air",
    "Delta Air",
    "Gol Trans",
    "Grupo LATAM",
    "Iberia",
    "JetSmart SPA",
    "K.L.M.",
    "Lacsa",
    "Latin American Wings",
    "Oceanair Linhas Aereas",
    "Plus Ultra Lineas Aereas",
    "Qantas Airways",
    "Sky Airline",
    "United Airlines"
]


class Flight(BaseModel):
    """Single flight data for prediction"""
    OPERA: str
    TIPOVUELO: str
    MES: int
    
    @validator('MES')
    def validate_mes(cls, v):
        if not 1 <= v <= 12:
            raise ValueError('MES must be between 1 and 12')
        return v
    
    @validator('TIPOVUELO')
    def validate_tipovuelo(cls, v):
        if v not in ['N', 'I']:
            raise ValueError('TIPOVUELO must be either "N" or "I"')
        return v
    
    @validator('OPERA')
    def validate_opera(cls, v):
        if v not in VALID_AIRLINES:
            raise ValueError(f'OPERA must be one of the valid airlines')
        return v


class PredictRequest(BaseModel):
    """Request body for predictions"""
    flights: List[Flight]
    
    @validator('flights')
    def validate_flights(cls, v):
        if len(v) == 0:
            raise ValueError('flights list cannot be empty')
        return v


@app.get("/health", status_code=200)
async def get_health() -> dict:
    """
    Health check endpoint.
    
    Returns:
        dict: Status of the API
    """
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    """
    Predict flight delays.
    
    Args:
        request: PredictRequest with list of flights
        
    Returns:
        dict: Dictionary with 'predict' key containing list of predictions
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        # Convert flights to DataFrame
        flights_data = [flight.dict() for flight in request.flights]
        data = pd.DataFrame(flights_data)
        
        # Add dummy datetime columns for preprocessing
        # These are required by the model but not used for prediction without target
        data['Fecha-O'] = '2023-01-01 00:00:00'
        data['Fecha-I'] = '2023-01-01 00:00:00'
        
        # Preprocess the data
        features = model.preprocess(data=data)
        
        # Make predictions
        predictions = model.predict(features=features)
        
        # Return predictions
        return {
            "predict": predictions
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)