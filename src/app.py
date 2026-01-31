import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Ethiopian Crop Recommendation System",
    description="Recommends the best crop group for given soil and climate conditions in Ethiopia (based on real localized data).",
    version="2.0",
)


# Root redirects to the web UI (for Hugging Face Spaces)
@app.get("/")
def root():
    return RedirectResponse(url="/static/")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

model = None
scaler = None
le = None

GROUP_CROPS = {
    "Major_Cereals": ["Teff", "Maize", "Wheat", "Barley"],
    "Cereals": ["Dagussa", "Sorghum"],
    "Pulses": ["Bean", "Pea"],
    "Specialty": ["Niger seed", "Potato", "Red Pepper", "Fallow"],
}

CLASS_EXPLANATIONS = {
    "Major_Cereals": "Major Cereals – staple grains like Teff (for injera), Maize, Wheat, and Barley. Highly suitable for most Ethiopian farms.",
    "Cereals": "Minor Cereals – drought-resistant alternatives like Dagussa and Sorghum, good for drier areas.",
    "Pulses": "Pulses – nutritious legumes (Bean, Pea) that fix nitrogen and improve soil health.",
    "Specialty": "Specialty Crops – high-value or niche options (Niger seed, Potato, Red Pepper, Fallow rotation).",
}


class CropInput(BaseModel):
    N: float = Field(..., ge=0, le=200, description="Nitrogen (0-200)", example=70)
    P: float = Field(..., ge=0, le=150, description="Phosphorus (0-150)", example=40)
    K: float = Field(..., ge=0, le=200, description="Potassium (0-200)", example=60)
    ph: float = Field(..., ge=3, le=10, description="Soil pH", example=6.5)
    temperature: float = Field(
        ..., ge=5, le=45, description="Average Temperature °C", example=22.0
    )
    humidity: float = Field(
        ..., ge=10, le=100, description="Average Humidity %", example=65.0
    )
    rainfall: float = Field(
        ..., ge=100, le=2500, description="Annual Rainfall mm", example=1100.0
    )
    altitude_m: float = Field(
        ..., ge=0, le=4500, description="Altitude (meters)", example=2400.0
    )
    Zn: float = Field(..., ge=0, le=50, description="Zinc level", example=5.0)
    S: float = Field(..., ge=0, le=100, description="Sulfur level", example=20.0)
    soil_moisture: float = Field(
        ..., ge=0, le=1, description="Topsoil moisture (0-1)", example=0.6
    )


@app.on_event("startup")
def load_models():
    global model, scaler, le
    try:
        model_path = MODELS_DIR / "best_crop_model.pkl"
        scaler_path = MODELS_DIR / "scaler_merged.pkl"
        le_path = MODELS_DIR / "label_encoder_merged.pkl"

        print(f"DEBUG: Loading from: {MODELS_DIR.absolute()}")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        le = joblib.load(le_path)
        print("All models loaded successfully! 🌾")
    except Exception as e:
        print(f"CRITICAL: Failed to load models: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        field = " → ".join(str(loc) for loc in error["loc"])
        msg = error["msg"]
        errors.append(f"{field}: {msg}")

    friendly_message = (
        "Input validation failed:\n"
        + "\n".join(errors)
        + "\n\nPlease check all fields and try again."
    )
    return JSONResponse(status_code=422, content={"detail": friendly_message})


@app.post("/predict")
def predict_crop(input_data: CropInput):
    if model is None or scaler is None or le is None:
        raise HTTPException(status_code=500, detail="Server error: Models not loaded.")

    try:
        input_array = np.array(
            [
                [
                    input_data.N,
                    input_data.P,
                    input_data.K,
                    input_data.ph,
                    input_data.temperature,
                    input_data.humidity,
                    input_data.rainfall,
                    input_data.altitude_m,
                    input_data.Zn,
                    input_data.S,
                    input_data.soil_moisture,
                ]
            ]
        )
        scaled_input = scaler.transform(input_array)
        probs = model.predict_proba(scaled_input)[0]

        top3_idx = np.argsort(probs)[-3:][::-1]
        top3_crops = le.inverse_transform(top3_idx)
        top3_probs = probs[top3_idx]

        best_idx = np.argmax(probs)
        best_crop = le.inverse_transform([best_idx])[0]
        best_crops_list = GROUP_CROPS.get(best_crop, [])

        top3_details = []
        for crop, prob in zip(top3_crops, top3_probs):
            top3_details.append(
                {
                    "crop": crop,
                    "crops_in_group": GROUP_CROPS.get(crop, []),
                    "probability_pct": round(float(prob) * 100, 2),
                }
            )

        response = {
            "recommended_crop": best_crop,
            "crops_in_group": best_crops_list,
            "explanation": CLASS_EXPLANATIONS.get(best_crop, ""),
            "confidence_pct": round(float(probs[best_idx]) * 100, 2),
            "top3_recommendations": top3_details,
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health")
def health():
    return {"status": "healthy" if model is not None else "models not loaded"}
