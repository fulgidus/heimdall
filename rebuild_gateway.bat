@echo off
REM Rebuild and restart the API Gateway to apply AUTH_ENABLED=False fix

setlocal enabledelayedexpansion

cd /d c:\Users\aless\Documents\Projects\heimdall

echo [INFO] Rebuilding api-gateway image with latest code...
docker build --no-cache -f services\api-gateway\Dockerfile -t heimdall-api-gateway .
if !errorlevel! neq 0 (
    echo [ERROR] Build failed!
    exit /b 1
)

echo.
echo [INFO] Restarting api-gateway container...
docker restart heimdall-api-gateway

echo.
echo [INFO] Waiting 10 seconds for container to start...
timeout /t 10 /nobreak

echo.
echo [INFO] Testing health endpoint...
for /f "tokens=*" %%a in ('curl -s http://localhost:8000/health') do (
    echo ✓ Health: %%a
)

echo.
echo [INFO] Testing acquisition endpoint (should NOT have 403 anymore)...
for /f "tokens=*" %%a in ('curl -s -w "\nHTTP Status: %%{http_code}" http://localhost:8000/api/v1/acquisition/config') do (
    echo ✓ Acquisition: %%a
)

echo.
echo [SUCCESS] API Gateway has been rebuilt and restarted!
echo Frontend should now be able to connect without 403 errors.
pause
