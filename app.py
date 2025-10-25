# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI(
    title="Skin Disease Classifier API 🩺",
    description="Upload an image and get the predicted skin disease using a fine-tuned model.",
    version="1.0.0",
)

MODEL_PATH = "skin_disease_finetuned.h5"

# --- Load model safely
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# --- Class names (same order as training)
CLASS_NAMES = ["Acne", "Eczema", "Keratosis Pilaris", "Psoriasis", "Warts"]

# --- Confidence threshold
CONFIDENCE_THRESHOLD = 0.6


@app.get("/")
def home():
    return {"message": "🩺 Skin Disease Classifier API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess image
        img_resized = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

        # Predict
        preds = model.predict(img_array)[0]
        predicted_class = CLASS_NAMES[np.argmax(preds)]
        confidence = float(np.max(preds))

        # If not confident
        if confidence < CONFIDENCE_THRESHOLD:
            result = {
                "prediction": "Unknown",
                "confidence": round(confidence, 2),
                "message": "Model not confident enough",
            }
        else:
            result = {
                "prediction": predicted_class,
                "confidence": round(confidence, 2),
            }

        # Include all class probabilities
        class_probs = {name: float(prob) for name, prob in zip(CLASS_NAMES, preds)}
        result["all_probabilities"] = class_probs

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
