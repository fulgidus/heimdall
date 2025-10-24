from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import logging
import os

from .config import settings
from .models.health import HealthResponse

logger = logging.getLogger(__name__)

# Import authentication
try:
    from auth import get_current_user, require_role, require_admin, require_operator, User
    # TEMPORARILY DISABLED FOR FRONTEND TESTING
    AUTH_ENABLED = False  # Set to True when Keycloak is configured
    logger.info("⚠️ Authentication disabled for development")
except ImportError as e:
    logger.warning(f"⚠️ Authentication disabled - could not import auth module: {e}")
    AUTH_ENABLED = False
    # Define dummy user for when auth is disabled
    class User:
        def __init__(self):
            self.username = "anonymous"
            self.roles = []

SERVICE_NAME = "api-gateway"
SERVICE_VERSION = "0.1.0"
SERVICE_PORT = 8000

# Backend service URLs
RF_ACQUISITION_URL = "http://rf-acquisition:8001"
INFERENCE_URL = "http://inference:8003"
TRAINING_URL = "http://training:8002"
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


if AUTH_ENABLED:
    @app.api_route("/api/v1/acquisition/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy_to_rf_acquisition(
        request: Request,
        path: str,
        user: User = Depends(get_current_user)
    ):
        """Proxy requests to RF Acquisition service (requires authentication)."""
        if not user.is_operator:
            raise HTTPException(status_code=403, detail="Operator access required")
        logger.debug(f"📡 Acquisition route matched: path={path} (user={user.username})")
        return await proxy_request(request, RF_ACQUISITION_URL)
else:
    @app.api_route("/api/v1/acquisition/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy_to_rf_acquisition(request: Request, path: str):
        """Proxy requests to RF Acquisition service (no authentication required)."""
        logger.debug(f"📡 Acquisition route matched: path={path} (authentication disabled)")
        return await proxy_request(request, RF_ACQUISITION_URL)


if AUTH_ENABLED:
    @app.api_route("/api/v1/inference/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy_to_inference(request: Request, path: str, user: User = Depends(get_current_user)):
        """Proxy requests to Inference service (requires authentication)."""
        if not user.is_viewer:
            raise HTTPException(status_code=403, detail="Viewer access required")
        logger.debug(f"🧠 Inference route matched: path={path} (user={user.username})")
        return await proxy_request(request, INFERENCE_URL)
else:
    @app.api_route("/api/v1/inference/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy_to_inference(request: Request, path: str):
        """Proxy requests to Inference service (no authentication required)."""
        logger.debug(f"🧠 Inference route matched: path={path} (authentication disabled)")
        return await proxy_request(request, INFERENCE_URL)


if AUTH_ENABLED:
    @app.api_route("/api/v1/training/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy_to_training(request: Request, path: str, user: User = Depends(get_current_user)):
        """Proxy requests to Training service (requires authentication)."""
        if not user.is_operator:
            raise HTTPException(status_code=403, detail="Operator access required")
        logger.debug(f"📚 Training route matched: path={path} (user={user.username})")
        return await proxy_request(request, TRAINING_URL)
else:
    @app.api_route("/api/v1/training/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy_to_training(request: Request, path: str):
        """Proxy requests to Training service (no authentication required)."""
        logger.debug(f"📚 Training route matched: path={path} (authentication disabled)")
        return await proxy_request(request, TRAINING_URL)


if AUTH_ENABLED:
    @app.api_route("/api/v1/sessions/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy_to_data_ingestion(request: Request, path: str, user: User = Depends(get_current_user)):
        """Proxy requests to Data Ingestion service (requires authentication)."""
        if not user.is_operator:
            raise HTTPException(status_code=403, detail="Operator access required")
        logger.debug(f"💾 Data Ingestion route matched: path={path} (user={user.username})")
        return await proxy_request(request, DATA_INGESTION_URL)
else:
    @app.api_route("/api/v1/sessions/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy_to_data_ingestion(request: Request, path: str):
        """Proxy requests to Data Ingestion service (no authentication required)."""
        logger.debug(f"💾 Data Ingestion route matched: path={path} (authentication disabled)")
        return await proxy_request(request, DATA_INGESTION_URL)

# Analytics endpoints are public for demo/dev purposes (no authentication required)
# In production, you would use: if not user.is_viewer: raise HTTPException(...)
@app.api_route("/api/v1/analytics/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_to_inference_analytics(request: Request, path: str):
    """Proxy analytics requests to Inference service (public access for demo)."""
    logger.debug(f"📊 Analytics route matched: path={path}")
    return await proxy_request(request, INFERENCE_URL)

@app.get("/")
async def root():
    return {
        "service": SERVICE_NAME,
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "auth_enabled": AUTH_ENABLED
    }


@app.get("/health")
async def health_check():
    return HealthResponse(
        status="healthy",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        timestamp=datetime.utcnow()
    )


@app.get("/ready")
async def readiness_check():
    return {"ready": True}


@app.post("/auth/login")
async def login_proxy(request: Request):
    """
    Proxy OAuth2 token request to Keycloak.
    
    This endpoint proxies login requests to Keycloak, allowing the frontend
    to request tokens through the API Gateway (which has CORS enabled).
    """
    try:
        # Get form data from request
        body = await request.body()
        
        # Keycloak token endpoint
        keycloak_url = os.getenv("KEYCLOAK_URL", "http://keycloak:8080")
        keycloak_realm = os.getenv("KEYCLOAK_REALM", "heimdall")
        token_endpoint = f"{keycloak_url}/realms/{keycloak_realm}/protocol/openid-connect/token"
        
        # Forward to Keycloak
        async with httpx.AsyncClient() as client:
            response = await client.post(
                token_endpoint,
                content=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
        
        # Return Keycloak response with CORS headers (added by middleware)
        return JSONResponse(
            status_code=response.status_code,
            content=response.json() if response.status_code == 200 else {"error": "Authentication failed"}
        )
    except Exception as e:
        logger.error(f"Login proxy error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )


if AUTH_ENABLED:
    @app.get("/auth/check")
    async def auth_check(user: User = Depends(get_current_user)):
        """Check authentication status and return user info."""
        return {
            "authenticated": True,
            "auth_enabled": True,
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "roles": user.roles,
                "is_admin": user.is_admin,
                "is_operator": user.is_operator,
                "is_viewer": user.is_viewer,
            }
        }
else:
    @app.get("/auth/check")
    async def auth_check():
        """Check authentication status and return user info."""
        return {
            "authenticated": False,
            "auth_enabled": False,
            "message": "Authentication is disabled"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
