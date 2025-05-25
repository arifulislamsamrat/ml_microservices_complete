from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import logging
import uvicorn
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("service_a")

app = FastAPI(
    title="Service A - Input Logger & API Gateway",
    description="Logs input data and forwards ML prediction requests",
    version="1.0.0"
)

# Define input data model
class InputData(BaseModel):
    data: str
    forward_to_model: bool = True

class HealthResponse(BaseModel):
    service: str
    status: str
    timestamp: str
    version: str

# Service B URL - configurable for different environments
SERVICE_B_URL = os.getenv("SERVICE_B_URL", "http://localhost:8001/predict")

@app.get("/", response_model=dict)
def read_root():
    return {
        "service": "Service A - Input Logger & API Gateway",
        "status": "running",
        "endpoints": ["/health", "/process"],
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        service="Service A",
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/process")
async def process_input(input_data: InputData):
    # Log the received input with timestamp
    timestamp = datetime.now().isoformat()
    logger.info(f"[{timestamp}] Received input: {input_data.data}")

    # If forward_to_model is True, send the data to Service B
    if input_data.forward_to_model:
        try:
            logger.info(f"Forwarding request to Service B at: {SERVICE_B_URL}")
            response = requests.post(
                SERVICE_B_URL,
                json={"input": input_data.data},
                timeout=10
            )
            response.raise_for_status()

            # Return both the logged status and the prediction from Service B
            ml_response = response.json()
            return {
                "status": "Input logged successfully",
                "timestamp": timestamp,
                "service": "Service A",
                "model_prediction": ml_response,
                "forwarded_to": SERVICE_B_URL
            }

        except requests.RequestException as e:
            error_msg = f"Service B is unavailable: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Service B unavailable",
                    "message": error_msg,
                    "timestamp": timestamp,
                    "service_b_url": SERVICE_B_URL
                }
            )
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Internal server error",
                    "message": str(e),
                    "timestamp": timestamp
                }
            )
    else:
        # Just return the logged status
        return {
            "status": "Input logged successfully",
            "timestamp": timestamp,
            "service": "Service A",
            "forwarded": False
        }

@app.get("/metrics")
def get_metrics():
    """Simple metrics endpoint for monitoring"""
    return {
        "service": "Service A",
        "uptime": "available",
        "service_b_configured": SERVICE_B_URL,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
