from datetime import datetime, timedelta
from fastapi import FastAPI, Request, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import httpx
import logging
import os
import sys
import asyncio

# Add parent directory to path for auth module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'common'))

from .config import settings
from .models.health import HealthResponse
from .websocket_manager import manager as ws_manager, heartbeat_task

# Import common health utilities (after path setup)
try:
    from common.health import HealthChecker
    HEALTH_CHECKER_ENABLED = True
except ImportError:
    HEALTH_CHECKER_ENABLED = False

# Import Keycloak authentication
try:
    from auth.keycloak_auth import get_current_user, require_admin, require_operator
    from auth.models import User
    AUTH_ENABLED = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Keycloak authentication enabled")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Authentication disabled - could not import auth module: {e}")
    AUTH_ENABLED = False
    
    # Define dummy user for when auth is disabled
    class User:
        def __init__(self):
            self.id = "anonymous"
            self.username = "anonymous"
            self.email = "anonymous@heimdall.local"
            self.roles = []
            self.is_admin = False
            self.is_operator = False
            self.is_viewer = False
    
    # Create a dummy dependency that never fails
    async def get_current_user():
        """Dummy auth - returns anonymous user when auth is disabled."""
        return User()
    
    def require_admin():
        async def _require_admin(user: User = Depends(get_current_user)):
            return user
        return _require_admin
    
    def require_operator():
        async def _require_operator(user: User = Depends(get_current_user)):
            return user
        return _require_operator

SERVICE_NAME = "api-gateway"
SERVICE_VERSION = "0.1.0"
SERVICE_PORT = 8000

# Backend service URLs - from settings
BACKEND_URL = settings.backend_url  # Renamed from rf_acquisition_url
INFERENCE_URL = settings.inference_url
TRAINING_URL = settings.training_url
# Legacy alias for backward compatibility
RF_ACQUISITION_URL = BACKEND_URL

app = FastAPI(title=f"Heimdall SDR - {SERVICE_NAME}", version=SERVICE_VERSION)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Initialize health checker if available
if HEALTH_CHECKER_ENABLED:
    health_checker = HealthChecker(SERVICE_NAME, SERVICE_VERSION)
    
    # Register backend service health checks
    async def check_backend():
        """Check backend service connectivity."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{BACKEND_URL}/health")
            if response.status_code != 200:
                raise Exception(f"Backend unhealthy: {response.status_code}")
    
    async def check_inference():
        """Check inference service connectivity."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{INFERENCE_URL}/health")
            if response.status_code != 200:
                raise Exception(f"Inference unhealthy: {response.status_code}")
    
    async def check_training():
        """Check training service connectivity."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{TRAINING_URL}/health")
            if response.status_code != 200:
                raise Exception(f"Training unhealthy: {response.status_code}")
    
    health_checker.register_dependency("backend", check_backend)
    health_checker.register_dependency("inference", check_inference)
    health_checker.register_dependency("training", check_training)


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


# =============================================================================
# PUBLIC HEALTH ENDPOINTS - No authentication required
# =============================================================================

@app.get("/health")
async def health_check():
    """
    Liveness probe - checks if API Gateway is alive.
    
    Returns basic health status without backend service checks.
    """
    return HealthResponse(
        status="healthy",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        timestamp=datetime.utcnow()
    )


@app.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check with backend service status.
    
    Returns comprehensive health information including all backend services.
    """
    if not HEALTH_CHECKER_ENABLED:
        return JSONResponse(
            status_code=200,
            content={
                "status": "up",
                "service_name": SERVICE_NAME,
                "version": SERVICE_VERSION,
                "message": "Health checker not available"
            }
        )
    
    result = await health_checker.check_all()
    return JSONResponse(
        status_code=200 if result.ready else 503,
        content=result.to_dict()
    )


@app.get("/ready")
async def readiness_check():
    """
    Readiness probe - checks if API Gateway can handle requests.
    
    Validates connectivity to backend services.
    """
    if not HEALTH_CHECKER_ENABLED:
        return {"ready": True, "service": SERVICE_NAME}
    
    try:
        result = await health_checker.check_all()
        
        # Gateway can be ready even if some backend services are down
        # It will still proxy requests and return appropriate errors
        return JSONResponse(
            status_code=200,
            content={
                "ready": True,
                "service": SERVICE_NAME,
                "backend_services": result.to_dict()["dependencies"]
            }
        )
    except Exception as e:
        logger.error("Readiness check failed: %s", str(e), exc_info=True)
        return JSONResponse(
            status_code=200,  # Gateway is still ready even if checks fail
            content={"ready": True, "service": SERVICE_NAME, "warning": str(e)}
        )


@app.get("/startup")
async def startup_check():
    """
    Startup probe - checks if API Gateway has finished starting up.
    """
    return await readiness_check()


# These must be defined BEFORE the catch-all {path:path} routes to take precedence
@app.get("/api/v1/acquisition/health")
async def acquisition_health_public(request: Request):
    """Public health check endpoint for RF Acquisition service."""
    logger.debug(f"📡 Public health check: /api/v1/acquisition/health")
    return await proxy_request(request, RF_ACQUISITION_URL)


@app.get("/api/v1/acquisition/websdrs")
async def acquisition_websdrs_public(request: Request):
    """Public endpoint to get WebSDR configuration."""
    logger.debug(f"📡 Public WebSDRs endpoint: /api/v1/acquisition/websdrs")
    return await proxy_request(request, RF_ACQUISITION_URL)


@app.get("/api/v1/inference/health")
async def inference_health_public(request: Request):
    """Public health check endpoint for Inference service."""
    logger.debug(f"🧠 Public health check: /api/v1/inference/health")
    return await proxy_request(request, INFERENCE_URL)


@app.get("/api/v1/api-gateway/health")
async def api_gateway_health_public():
    """Health check for API Gateway service."""
    logger.debug(f"🌐 API Gateway health check: /api/v1/api-gateway/health")
    return HealthResponse(
        status="healthy",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        timestamp=datetime.utcnow()
    )


@app.get("/api/v1/backend/health")
async def backend_health_public():
    """Health check endpoint for Backend service."""
    logger.debug(f"🔧 Backend health check: /api/v1/backend/health")
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{BACKEND_URL}/health")
            if response.status_code == 200:
                return response.json()
            else:
                return HealthResponse(status="unhealthy", service="backend", version="unknown", timestamp=datetime.utcnow())
        except Exception as e:
            logger.warning(f"⚠️ Could not reach backend health: {str(e)}")
            return HealthResponse(status="unhealthy", service="backend", version="unknown", timestamp=datetime.utcnow())


@app.get("/api/v1/rf-acquisition/health")
async def rf_acquisition_health_public():
    """
    Health check endpoint for RF Acquisition service - maps to backend /health.
    DEPRECATED: Use /api/v1/backend/health instead.
    """
    logger.debug(f"📡 RF Acquisition health check (DEPRECATED): /api/v1/rf-acquisition/health")
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{BACKEND_URL}/health")
            if response.status_code == 200:
                # Return response with rf-acquisition as service name for backward compatibility
                data = response.json()
                data["service"] = "rf-acquisition"  # Override service name for compatibility
                return data
            else:
                return HealthResponse(status="unhealthy", service="rf-acquisition", version="unknown", timestamp=datetime.utcnow())
        except Exception as e:
            logger.warning(f"⚠️ Could not reach backend health: {str(e)}")
            return HealthResponse(status="unhealthy", service="rf-acquisition", version="unknown", timestamp=datetime.utcnow())


@app.get("/api/v1/inference/health")
async def inference_health_check():
    """Health check endpoint for Inference service - maps to backend /health."""
    logger.debug(f"🧠 Inference health check: /api/v1/inference/health")
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{INFERENCE_URL}/health")
            if response.status_code == 200:
                return response.json()
            else:
                return HealthResponse(status="unhealthy", service="inference", version="unknown", timestamp=datetime.utcnow())
        except Exception as e:
            logger.warning(f"⚠️ Could not reach inference health: {str(e)}")
            return HealthResponse(status="unhealthy", service="inference", version="unknown", timestamp=datetime.utcnow())


# OLD ENDPOINTS - Kept for backward compatibility but use /api/v1/{service}/health instead
@app.get("/api/v1/acquisition/health")
async def acquisition_health_deprecated(request: Request):
    """DEPRECATED: Use /api/v1/rf-acquisition/health instead."""
    logger.debug(f"📡 Deprecated: /api/v1/acquisition/health → /api/v1/rf-acquisition/health")
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{RF_ACQUISITION_URL}/health")
            if response.status_code == 200:
                return response.json()
            else:
                return HealthResponse(status="unhealthy", service="rf-acquisition", version="unknown", timestamp=datetime.utcnow())
        except Exception as e:
            raise HTTPException(status_code=503, detail="rf-acquisition unavailable")


# =============================================================================
# PROTECTED BACKEND ENDPOINTS - Requires auth (falls back to anonymous if disabled)
# =============================================================================

@app.api_route("/api/v1/backend/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_to_backend(
    request: Request,
    path: str,
    user: User = Depends(get_current_user)
):
    """Proxy requests to Backend service (requires authentication)."""
    if AUTH_ENABLED and not user.is_operator:
        raise HTTPException(status_code=403, detail="Operator access required")
    logger.debug(f"🔧 Backend route matched: path={path} (user={user.username})")
    return await proxy_request(request, BACKEND_URL)


# =============================================================================
# PROTECTED ACQUISITION ENDPOINTS - Requires auth (falls back to anonymous if disabled)
# DEPRECATED: Use /api/v1/backend instead
# =============================================================================

@app.api_route("/api/v1/acquisition/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_to_rf_acquisition(
    request: Request,
    path: str,
    user: User = Depends(get_current_user)
):
    """
    Proxy requests to Backend service (requires authentication).
    DEPRECATED: Use /api/v1/backend instead.
    """
    if AUTH_ENABLED and not user.is_operator:
        raise HTTPException(status_code=403, detail="Operator access required")
    logger.debug(f"📡 Acquisition route matched (DEPRECATED): path={path} (user={user.username})")
    return await proxy_request(request, BACKEND_URL)


# =============================================================================
# PROTECTED INFERENCE ENDPOINTS - Requires auth (falls back to anonymous if disabled)
# =============================================================================

@app.api_route("/api/v1/inference/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_to_inference(
    request: Request,
    path: str,
    user: User = Depends(get_current_user)
):
    """Proxy requests to Inference service (requires authentication)."""
    if AUTH_ENABLED and not user.is_viewer:
        raise HTTPException(status_code=403, detail="Viewer access required")
    logger.debug(f"🧠 Inference route matched: path={path} (user={user.username})")
    return await proxy_request(request, INFERENCE_URL)


# =============================================================================
# PROTECTED TRAINING ENDPOINTS
# =============================================================================

# =============================================================================
# PROTECTED TRAINING ENDPOINTS - Requires auth (falls back to anonymous if disabled)
# =============================================================================

@app.api_route("/api/v1/training/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_to_training(
    request: Request,
    path: str,
    user: User = Depends(get_current_user)
):
    """Proxy requests to Training service (requires authentication)."""
    if AUTH_ENABLED and not user.is_operator:
        raise HTTPException(status_code=403, detail="Operator access required")
    logger.debug(f"📚 Training route matched: path={path} (user={user.username})")
    return await proxy_request(request, TRAINING_URL)


# =============================================================================
# PROTECTED DATA INGESTION ENDPOINTS
# =============================================================================

# =============================================================================
# PROTECTED DATA INGESTION ENDPOINTS - Requires auth (falls back to anonymous if disabled)
# =============================================================================

@app.api_route("/api/v1/sessions/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_to_sessions(
    request: Request,
    path: str,
    user: User = Depends(get_current_user)
):
    """Proxy requests to Backend service for sessions management (requires authentication)."""
    if request.method == "OPTIONS":
        return Response(status_code=200)
    if AUTH_ENABLED and not user.is_operator:
        raise HTTPException(status_code=403, detail="Operator access required")
    logger.debug(f"💾 Sessions route matched: path={path} (user={user.username})")
    return await proxy_request(request, BACKEND_URL)


# =============================================================================
# PUBLIC ANALYTICS ENDPOINTS
# =============================================================================

@app.api_route("/api/v1/analytics/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_to_inference_analytics(request: Request, path: str):
    """Proxy analytics requests to Inference service (public access for demo)."""
    logger.debug(f"📊 Analytics route matched: path={path}")
    return await proxy_request(request, INFERENCE_URL)


# =============================================================================
# ROOT & ROOT ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint - returns service info."""
    return {
        "service": SERVICE_NAME,
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "auth_enabled": AUTH_ENABLED
    }


# =============================================================================
# WEBSOCKET ENDPOINT - Real-time Updates
# =============================================================================

@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time dashboard updates.
    
    Events broadcasted:
    - services:health - Service health status updates
    - websdrs:status - WebSDR receiver status changes
    - signals:detected - New signal detections
    - localizations:updated - New localization points
    """
    await ws_manager.connect(websocket)
    
    # Start heartbeat task
    heartbeat = asyncio.create_task(heartbeat_task(websocket))
    
    try:
        while True:
            # Receive messages from client (e.g., ping, subscribe events)
            data = await websocket.receive_json()
            
            event = data.get("event")
            
            # Handle ping/pong
            if event == "ping":
                await websocket.send_json({
                    "event": "pong",
                    "data": {},
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Handle subscription requests (future enhancement)
            elif event == "subscribe":
                event_name = data.get("data", {}).get("event_name")
                if event_name:
                    ws_manager.subscribe(websocket, event_name)
                    
            elif event == "unsubscribe":
                event_name = data.get("data", {}).get("event_name")
                if event_name:
                    ws_manager.unsubscribe(websocket, event_name)
                    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        heartbeat.cancel()
        ws_manager.disconnect(websocket)


# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================

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


@app.post("/api/v1/auth/refresh")
async def refresh_token_proxy(request: Request):
    """
    Proxy OAuth2 refresh token request to Keycloak.
    
    Uses refresh token to obtain a new access token without requiring credentials.
    """
    try:
        content_type = request.headers.get("content-type", "").lower()
        
        if "application/json" in content_type:
            body = await request.json()
            refresh_token = body.get("refresh_token")
        else:
            form_data = await request.form()
            refresh_token = form_data.get("refresh_token")
        
        if not refresh_token:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing refresh_token"}
            )
        
        keycloak_url = os.getenv("KEYCLOAK_URL", "http://keycloak:8080")
        keycloak_realm = os.getenv("KEYCLOAK_REALM", "heimdall")
        client_id = os.getenv("VITE_KEYCLOAK_CLIENT_ID", "heimdall-frontend")
        token_endpoint = f"{keycloak_url}/realms/{keycloak_realm}/protocol/openid-connect/token"
        
        logger.info(f"🔄 Proxying token refresh to: {token_endpoint}")
        
        form_data = {
            "client_id": client_id,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                token_endpoint,
                data=form_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
        
        logger.info(f"🔄 Keycloak refresh response: {response.status_code}")
        
        if response.status_code == 200:
            return JSONResponse(
                status_code=response.status_code,
                content=response.json()
            )
        else:
            error_content = response.json() if response.text else {}
            logger.warning(f"⚠️ Keycloak refresh error: {error_content}")
            return JSONResponse(
                status_code=response.status_code,
                content=error_content or {"error": "Token refresh failed"}
            )
    except Exception as e:
        logger.error(f"❌ Token refresh proxy error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )


@app.get("/api/v1/auth/check")
async def auth_check(user: User = Depends(get_current_user)):
    """Check authentication status and return user info."""
    return {
        "authenticated": AUTH_ENABLED,
        "auth_enabled": AUTH_ENABLED,
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "roles": user.roles,
            "is_admin": user.is_admin,
            "is_operator": user.is_operator,
            "is_viewer": user.is_viewer,
        } if AUTH_ENABLED else None
    }


# =============================================================================
# USER PROFILE & PREFERENCES ENDPOINTS (Keycloak-based)
# =============================================================================

if AUTH_ENABLED:
    @app.get("/api/v1/auth/me")
    async def get_current_user_info(user: User = Depends(get_current_user)):
        """Get current authenticated user information from Keycloak token."""
        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "roles": user.roles,
            "is_admin": user.is_admin,
            "is_operator": user.is_operator,
            "is_viewer": user.is_viewer,
        }

    @app.get("/api/v1/profile")
    async def get_user_profile(user: User = Depends(get_current_user)):
        """Get user profile from Keycloak."""
        # TODO: Extend with additional profile data from database if needed
        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "roles": user.roles,
            "first_name": user.username.split("@")[0] if user.username else "",
            "last_name": "",
            "created_at": datetime.utcnow().isoformat(),
            "last_login": datetime.utcnow().isoformat(),
        }


# =============================================================================
# SYSTEM STATUS & METRICS ENDPOINTS
# =============================================================================

@app.get("/api/v1/config")
async def get_config():
    """Get application configuration."""
    return {
        "websdrs": 7,
        "supported_bands": ["2m", "70cm"],
        "max_duration_seconds": 300,
        "min_frequency_mhz": 144.0,
        "max_frequency_mhz": 146.0,
        "keycloak_realm": os.getenv("KEYCLOAK_REALM", "heimdall"),
        "keycloak_url": os.getenv("KEYCLOAK_URL", "http://keycloak:8080"),
    }


@app.get("/api/v1/stats")
async def get_dashboard_stats():
    """Get dashboard statistics from database."""
    # TODO: Implement real stats aggregation from database
    return {
        "total_sessions": 0,
        "active_sessions": 0,
        "completed_predictions": 0,
        "average_accuracy_m": 0.0,
        "websdrs_online": 7,
        "uptime_percentage": 100.0,
    }


@app.get("/api/v1/system/status")
async def get_system_status():
    """Aggregate health status from all services."""
    services_health = []
    service_urls = {
        "api-gateway": settings.api_gateway_url,
        "backend": BACKEND_URL,
        "inference": INFERENCE_URL,
        "training": TRAINING_URL,
    }
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        for service_name, url in service_urls.items():
            try:
                response = await client.get(f"{url}/health")
                status = "healthy" if response.status_code == 200 else "unhealthy"
            except Exception:
                status = "unreachable"
            
            services_health.append({
                "name": service_name,
                "status": status,
                "url": url,
            })
    
    overall_healthy = all(s["status"] == "healthy" for s in services_health)
    
    result = {
        "overall_status": "healthy" if overall_healthy else "degraded",
        "services": services_health,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    # Broadcast to WebSocket clients
    services_health_dict = {s["name"]: {"status": s["status"]} for s in services_health}
    asyncio.create_task(ws_manager.broadcast("services:health", services_health_dict))
    
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
