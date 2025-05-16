from fastapi import FastAPI, Request, Form, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Dict, Optional
import joblib
import numpy as np
from config.paths_config import MODEL_OUTPUT_PATH
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Categorical mappings
MARKET_SEGMENT_MAPPING = {
    "Aviation": 0,
    "Complimentary": 1,
    "Corporate": 2,
    "Offline": 3,
    "Online": 4
}

MEAL_PLAN_MAPPING = {
    "Meal Plan 1": 0,
    "Meal Plan 2": 1,
    "Meal Plan 3": 2,
    "Not Selected": 3
}

ROOM_TYPE_MAPPING = {
    "Room Type 1": 0,
    "Room Type 2": 1,
    "Room Type 3": 2,
    "Room Type 4": 3,
    "Room Type 5": 4,
    "Room Type 6": 5,
    "Room Type 7": 6
}

# Reverse mappings for display
REVERSE_MARKET_SEGMENT = {v: k for k, v in MARKET_SEGMENT_MAPPING.items()}
REVERSE_MEAL_PLAN = {v: k for k, v in MEAL_PLAN_MAPPING.items()}
REVERSE_ROOM_TYPE = {v: k for k, v in ROOM_TYPE_MAPPING.items()}

class PredictionResponse(BaseModel):
    prediction: Optional[float] = Field(None, description="Predicted booking status")
    message: str = Field(..., description="Prediction message")

app = FastAPI(title="Hotel Booking Prediction API", description="Predicting hotel booking status")

# Global variable to store the loaded model
model = None

def load_model():
    global model
    try:
        if os.path.exists(MODEL_OUTPUT_PATH):
            model = joblib.load(MODEL_OUTPUT_PATH)
            logger.info(f"Model loaded successfully from {MODEL_OUTPUT_PATH}")
            return True
        else:
            logger.error(f"Model file not found at {MODEL_OUTPUT_PATH}")
            return False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    global model
    model_loaded = load_model()
    if not model_loaded:
        logger.warning("Model could not be loaded during startup")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "model_available": model is not None,
            "market_segments": MARKET_SEGMENT_MAPPING.keys(),
            "meal_plans": MEAL_PLAN_MAPPING.keys(),
            "room_types": ROOM_TYPE_MAPPING.keys()
        }
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    lead_time: int = Form(...),
    no_of_special_request: int = Form(...),
    avg_price_per_room: float = Form(...),
    arrival_month: int = Form(...),
    arrival_date: int = Form(...),
    market_segment_type: str = Form(...),
    no_of_week_nights: int = Form(...),
    no_of_weekend_nights: int = Form(...),
    type_of_meal_plan: str = Form(...),
    room_type_reserved: str = Form(...)
):
    if model is None:
        return PredictionResponse(
            prediction=None,
            message="Model is not available. Please ensure the model file is properly loaded."
        )

    try:
        # Convert categorical features using mappings
        market_segment_int = MARKET_SEGMENT_MAPPING[market_segment_type]
        meal_plan_int = MEAL_PLAN_MAPPING[type_of_meal_plan]
        room_type_int = ROOM_TYPE_MAPPING[room_type_reserved]

        # Prepare input data for prediction
        prediction_input = np.array([[
            lead_time,
            no_of_special_request,
            avg_price_per_room,
            arrival_month,
            arrival_date,
            market_segment_int,
            no_of_week_nights,
            no_of_weekend_nights,
            meal_plan_int,
            room_type_int
        ]])
        
        # Make prediction
        prediction = model.predict(prediction_input)[0]
        message = "The booking is likely to be confirmed." if prediction == 1 else "The booking is likely to be cancelled."
        return PredictionResponse(prediction=float(prediction), message=message)
    
    except KeyError as ke:
        logger.error(f"Invalid categorical value: {str(ke)}")
        return PredictionResponse(
            prediction=None,
            message=f"Invalid categorical value: {str(ke)}"
        )
    except ValueError as ve:
        logger.error(f"Invalid input value: {str(ve)}")
        return PredictionResponse(
            prediction=None,
            message=f"Invalid input value: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return PredictionResponse(
            prediction=None,
            message=f"An error occurred during prediction: {str(e)}"
        )
        
        
        
        





