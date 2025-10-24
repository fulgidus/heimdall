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
AUTH_ENABLED = True  # ENABLED: Keycloak OAuth2 authentication
try:
    from auth import get_current_user, require_role, require_admin, require_operator, User
    logger.info("⚠️ Authentication module imported (but disabled)")
except ImportError as e:
    logger.warning(f"⚠️ Authentication disabled - could not import auth module: {e}")
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


@app.post("/api/v1/auth/login")
async def login_proxy(request: Request):
    """
    Proxy OAuth2 token request to Keycloak.
    
    This endpoint proxies login requests to Keycloak, allowing the frontend
    to request tokens through the API Gateway (which has CORS enabled).
    
    Accepts both:
    - application/x-www-form-urlencoded (from OAuth2 clients)
    - application/json (for convenience)
    """
    try:
        # Parse body based on content type
        content_type = request.headers.get("content-type", "").lower()
        
        if "application/json" in content_type:
            # Parse JSON body
            body = await request.json()
            email = body.get("email") or body.get("username")
            password = body.get("password")
            logger.debug(f"📋 Parsed JSON body: email={email}")
        else:
            # Parse form-urlencoded body (OAuth2 standard)
            form_data = await request.form()
            email = form_data.get("username") or form_data.get("email")
            password = form_data.get("password")
            logger.debug(f"📋 Parsed form-urlencoded body: email={email}")
        
        # Validate required fields
        if not email or not password:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing email/username or password"}
            )
        
        # Keycloak token endpoint
        keycloak_url = os.getenv("KEYCLOAK_URL", "http://keycloak:8080")
        keycloak_realm = os.getenv("KEYCLOAK_REALM", "heimdall")
        client_id = os.getenv("VITE_KEYCLOAK_CLIENT_ID", "heimdall-frontend")
        token_endpoint = f"{keycloak_url}/realms/{keycloak_realm}/protocol/openid-connect/token"
        
        logger.info(f"🔐 Proxying login to: {token_endpoint}")
        
        # Build form data for Keycloak (it expects form-urlencoded, not JSON)
        form_data = {
            "client_id": client_id,
            "username": email,
            "password": password,
            "grant_type": "password"
        }
        
        # Forward to Keycloak
        async with httpx.AsyncClient() as client:
            response = await client.post(
                token_endpoint,
                data=form_data,  # Use data= for form-urlencoded
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
        
        logger.info(f"🔐 Keycloak response: {response.status_code}")
        
        # Return Keycloak response with CORS headers (added by middleware)
        if response.status_code == 200:
            return JSONResponse(
                status_code=response.status_code,
                content=response.json()
            )
        else:
            error_content = response.json() if response.text else {}
            logger.warning(f"⚠️ Keycloak error: {error_content}")
            return JSONResponse(
                status_code=response.status_code,
                content=error_content or {"error": "Authentication failed"}
            )
    except Exception as e:
        logger.error(f"❌ Login proxy error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )


if AUTH_ENABLED:
    @app.get("/api/v1/auth/check")
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
    @app.get("/api/v1/auth/check")
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
