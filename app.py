# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import cv2
from io import BytesIO

app = FastAPI(
    title="Skin Disease Classifier API",
    description="Upload an image and get the predicted skin disease using EfficientNet preprocessing.",
    version="1.0.0",
)

# --- Model file path
MODEL_PATH = "skin_disease_finetuned.h5"

# --- Load model safely
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

# --- Class names
CLASS_NAMES = ["Acne", "Eczema", "Keratosis Pilaris", "Psoriasis", "Warts"]

# --- Confidence threshold
CONFIDENCE_THRESHOLD = 0.6

@app.get("/")
def home():
    return {"message": "Skin Disease Classifier API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded file bytes
        image_bytes = await file.read()

        # Convert bytes to numpy array for OpenCV
        file_bytes = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Resize to model input size
        img_resized = cv2.resize(img, (224, 224))

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # EfficientNet preprocessing
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_rgb.astype(np.float32))

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array)[0]
        pred_class = CLASS_NAMES[np.argmax(preds)]
        confidence = float(np.max(preds))

        # Check confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            result = {
                "prediction": "Unknown",
                "confidence": round(confidence, 2),
                "message": "Model not confident enough"
            }
        else:
            result = {
                "prediction": pred_class,
                "confidence": round(confidence, 2)
            }

        # Include all class probabilities
        class_probs = {name: float(prob) for name, prob in zip(CLASS_NAMES, preds)}
        result["all_probabilities"] = class_probs

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
