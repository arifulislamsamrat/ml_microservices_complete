from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import uvicorn
import random
import numpy as np
import os
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("service_b")

app = FastAPI(
    title="Service B - ML Prediction Service",
    description="Machine Learning prediction service with classification capabilities",
    version="1.0.0"
)

# Define input/output data models
class PredictionRequest(BaseModel):
    input: str

class PredictionResult(BaseModel):
    class_: str
    confidence: float
    input_length: int
    processing_time_ms: float

class PredictionResponse(BaseModel):
    prediction: PredictionResult
    message: str
    timestamp: str
    model_version: str

class HealthResponse(BaseModel):
    service: str
    status: str
    model_status: str
    timestamp: str
    version: str

# Enhanced ML Model simulation
class AdvancedMLModel:
    def __init__(self):
        self.classes = ["cat", "dog", "bird", "fish", "rabbit", "horse", "elephant", "tiger"]
        self.model_version = "v2.1.0"
        self.feature_weights = np.random.random(len(self.classes))
        logger.info(f"Advanced ML model initialized - Version: {self.model_version}")
        logger.info(f"Available classes: {self.classes}")

    def extract_features(self, input_text: str) -> np.ndarray:
        """Extract features from input text (simulated)"""
        # Simple feature extraction based on text characteristics
        features = [
            len(input_text),  # Length
            input_text.count(' '),  # Word count
            len(set(input_text.lower())),  # Unique characters
            input_text.count('a') + input_text.count('e'),  # Vowel count
            hash(input_text.lower()) % 100,  # Text hash feature
        ]

        # Normalize features
        features = np.array(features, dtype=float)
        features = features / (np.linalg.norm(features) + 1e-8)
        return features

    def predict(self, input_text: str) -> Dict[str, Any]:
        """Make prediction with enhanced logic"""
        start_time = datetime.now()

        logger.info(f"Processing prediction for input: '{input_text[:50]}...'")

        # Extract features
        features = self.extract_features(input_text)

        # Simulate model computation
        # In reality, this would be actual ML model inference
        text_lower = input_text.lower()

        # Enhanced prediction logic based on keywords
        class_probabilities = {}

        for i, class_name in enumerate(self.classes):
            # Base probability
            prob = 0.1 + random.random() * 0.3

            # Boost probability if class name appears in text
            if class_name in text_lower:
                prob += 0.4

            # Feature-based adjustment
            prob += np.dot(features, self.feature_weights) * 0.1

            class_probabilities[class_name] = prob

        # Normalize probabilities
        total_prob = sum(class_probabilities.values())
        class_probabilities = {k: v/total_prob for k, v in class_probabilities.items()}

        # Select predicted class
        predicted_class = max(class_probabilities, key=class_probabilities.get)
        confidence = class_probabilities[predicted_class]

        # Ensure confidence is realistic (0.6-0.99)
        confidence = max(0.6, min(0.99, confidence))

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        logger.info(f"Prediction completed: {predicted_class} (confidence: {confidence:.2f})")

        return {
            "class": predicted_class,
            "confidence": round(confidence, 3),
            "input_length": len(input_text),
            "processing_time_ms": round(processing_time, 2),
            "model_version": self.model_version
        }

# Initialize the ML model
model = AdvancedMLModel()

@app.get("/", response_model=dict)
def read_root():
    return {
        "service": "Service B - ML Prediction Service",
        "status": "running",
        "model_version": model.model_version,
        "available_classes": model.classes,
        "endpoints": ["/health", "/predict", "/model_info"],
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        service="Service B",
        status="healthy",
        model_status="ready",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make ML prediction on input data"""
    try:
        # Validate input
        if not request.input or len(request.input.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Input cannot be empty"
            )

        # Get prediction from model
        prediction_result = model.predict(request.input)

        # Create response
        prediction = PredictionResult(
            class_=prediction_result["class"],
            confidence=prediction_result["confidence"],
            input_length=prediction_result["input_length"],
            processing_time_ms=prediction_result["processing_time_ms"]
        )

        response = PredictionResponse(
            prediction=prediction,
            message=f"Predicted class: {prediction_result['class']} with {prediction_result['confidence']*100:.1f}% confidence",
            timestamp=datetime.now().isoformat(),
            model_version=prediction_result["model_version"]
        )

        return response

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Prediction failed",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/model_info")
def get_model_info():
    """Get information about the ML model"""
    return {
        "model_version": model.model_version,
        "available_classes": model.classes,
        "model_type": "Enhanced Classification Model",
        "features": "Text-based feature extraction",
        "last_updated": datetime.now().isoformat()
    }

@app.get("/metrics")
def get_metrics():
    """Simple metrics endpoint for monitoring"""
    return {
        "service": "Service B",
        "model_version": model.model_version,
        "available_classes_count": len(model.classes),
        "uptime": "available",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
