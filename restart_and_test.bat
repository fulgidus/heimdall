@echo off
REM Restart API Gateway and test 403 issue

echo [1] Stopping all containers...
docker-compose -f c:\Users\aless\Documents\Projects\heimdall\docker-compose.yml down

echo.
echo [2] Starting containers...
docker-compose -f c:\Users\aless\Documents\Projects\heimdall\docker-compose.yml up -d

echo.
echo [3] Waiting for API Gateway to start (15 seconds)...
timeout /t 15 /nobreak

echo.
echo [4] Checking API Gateway logs...
docker logs heimdall-api-gateway | find "AUTH"

echo.
echo [5] Testing API Gateway health endpoint...
powershell -Command "try { (Invoke-WebRequest -Uri 'http://localhost:8000/health' -UseBasicParsing).StatusCode } catch { $_.Exception.Response.StatusCode.Value }"

echo.
echo [6] Testing acquisition endpoint (should work now without 403)...
powershell -Command "try { (Invoke-WebRequest -Uri 'http://localhost:8000/api/v1/acquisition/config' -UseBasicParsing).StatusCode } catch { Write-Host 'Status:' $_.Exception.Response.StatusCode.Value }"

echo.
echo Done! Frontend should now be able to connect without 403 errors.
pause
