from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import logging
from .config import settings
from .models.health import HealthResponse

logger = logging.getLogger(__name__)

SERVICE_NAME = "api-gateway"
SERVICE_VERSION = "0.1.0"
SERVICE_PORT = 8000

# Backend service URLs
RF_ACQUISITION_URL = "http://rf-acquisition:8001"
INFERENCE_URL = "http://inference:8002"
TRAINING_URL = "http://training:8003"
DATA_INGESTION_URL = "http://data-ingestion-web:8004"

app = FastAPI(title=f"Heimdall SDR - {SERVICE_NAME}", version=SERVICE_VERSION)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


async def proxy_request(request: Request, target_url: str):
    """
    Proxy HTTP request to backend service.
    
    Args:
        request: Incoming FastAPI request
        target_url: Target backend service URL
    
    Returns:
        JSONResponse with backend response
    """
    # Build full target URL
    path = request.url.path
    query = str(request.url.query)
    full_url = f"{target_url}{path}"
    if query:
        full_url = f"{full_url}?{query}"
    
    logger.info(f"🔄 Proxy: {request.method} {full_url}")
    
    # Get request body if present
    body = None
    if request.method in ["POST", "PUT", "PATCH"]:
        body = await request.body()
    
    # Forward headers (excluding host)
    headers = dict(request.headers)
    headers.pop('host', None)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.request(
                method=request.method,
                url=full_url,
                headers=headers,
                content=body,
            )
            
            logger.info(f"✅ Proxy response: {response.status_code} from {full_url}")
            
            # Parse response content safely
            try:
                content = response.json() if response.text else {}
            except Exception as json_err:
                logger.warning(f"⚠️ Response is not JSON: {str(json_err)}")
                content = {"raw": response.text[:500]}
            
            return JSONResponse(
                content=content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
        except httpx.TimeoutException:
            logger.error(f"❌ Timeout proxying request to {full_url}")
            raise HTTPException(status_code=504, detail="Backend service timeout")
        except httpx.ConnectError as e:
            logger.error(f"❌ Connection error proxying request to {full_url}: {str(e)}")
            raise HTTPException(status_code=503, detail="Backend service unavailable")
        except Exception as e:
            logger.exception(f"❌ Error proxying request to {full_url}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")


@app.api_route("/api/v1/acquisition/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_to_rf_acquisition(request: Request, path: str):
    """Proxy requests to RF Acquisition service."""
    logger.debug(f"📡 Acquisition route matched: path={path}")
    return await proxy_request(request, RF_ACQUISITION_URL)


@app.api_route("/api/v1/inference/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_to_inference(request: Request, path: str):
    """Proxy requests to Inference service."""
    logger.debug(f"🧠 Inference route matched: path={path}")
    return await proxy_request(request, INFERENCE_URL)


@app.api_route("/api/v1/training/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_to_training(request: Request, path: str):
    """Proxy requests to Training service."""
    logger.debug(f"📚 Training route matched: path={path}")
    return await proxy_request(request, TRAINING_URL)


@app.api_route("/api/v1/sessions/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_to_data_ingestion(request: Request, path: str):
    """Proxy requests to Data Ingestion service."""
    logger.debug(f"💾 Data Ingestion route matched: path={path}")
    return await proxy_request(request, DATA_INGESTION_URL)


@app.get("/")
async def root():
    return {"service": SERVICE_NAME, "status": "running", "timestamp": datetime.utcnow().isoformat()}


@app.get("/health")
async def health_check():
    return HealthResponse(status="healthy", service=SERVICE_NAME, version=SERVICE_VERSION, timestamp=datetime.utcnow())


@app.get("/ready")
async def readiness_check():
    return {"ready": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
