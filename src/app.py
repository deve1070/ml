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
