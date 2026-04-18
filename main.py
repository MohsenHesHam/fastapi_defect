from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import json
from detection import detect_defect

app = FastAPI(title="Surface Defect Detection API", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "transfer_learning_finetuned.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

model = None
class_names = None
startup_error = None

@app.on_event("startup")
async def load_model():
    global model, class_names, startup_error
    try:
        import tensorflow as tf
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        if not CLASS_NAMES_PATH.exists():
            raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

        # Rebuild architecture first
        base_model = tf.keras.applications.MobileNetV2(
            weights=None,
            include_top=False,
            input_shape=(200, 200, 3)
        )
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(6, activation='softmax')
        ])

        # Build the model before loading weights
        model.build(input_shape=(None, 200, 200, 3))

        print("Loading weights...")
        model.load_weights(str(MODEL_PATH))

        with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
            class_names = json.load(f)

        startup_error = None
        print("Model and class names loaded successfully")

    except Exception as e:
        model = None
        class_names = None
        startup_error = str(e)
        print(f"Error loading model or class names: {e}")

def preprocess_image(image_bytes):
    try:
        import cv2
        import numpy as np

        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image format")

        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to 200x200
        img = cv2.resize(img, (200, 200))

        # Normalize to [0,1]
        img = img.astype(np.float32) / 255.0

        return img
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

@app.post("/predict")
async def predict_defect(file: UploadFile = File(...)):
    if model is None or class_names is None:
        raise HTTPException(
            status_code=503,
            detail=f"Detection model is unavailable. {startup_error or 'Startup did not complete successfully.'}"
        )
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:

        # Read and Image.
        image_bytes = await file.read()

        # Preprocess the image.
        img_array = preprocess_image(image_bytes)

        # Run detection.
        result = detect_defect(model, img_array, class_names)
        
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Surface Defect Detection API",
        "status": "ready" if model is not None else "degraded",
        "model_loaded": model is not None,
    }

@app.get("/health")
async def health():
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "details": startup_error,
    }
