@echo off
REM Quick Start: Test and Run Frontend-Backend Integration
REM Questo script fa tutto in una volta

echo.
echo ============================================================================
echo           🚀 Heimdall Frontend-Backend Integration Test
echo ============================================================================
echo.

cd /d "c:\Users\aless\Documents\Projects\heimdall"

echo [1/3] Testing Backend Connectivity...
echo.
python test_backend_connectivity.py

if errorlevel 1 (
    echo.
    echo [ERROR] Backend connectivity test failed!
    echo.
    echo Avvio il backend per te...
    echo.
    cd services\rf-acquisition
    echo Avvio: python src\main.py
    echo [In un NUOVO TERMINAL] Poi lancia questo script di nuovo
    start cmd /k "cd /d services\rf-acquisition && python src\main.py"
    exit /b 1
)

echo.
echo.
echo ✅ Backend connectivity OK!
echo.
echo [2/3] Starting Frontend Dev Server...
echo.
timeout /t 2

cd /d "c:\Users\aless\Documents\Projects\heimdall\frontend"
npm run dev

echo.
echo.
echo [3/3] Frontend server avviato!
echo.
echo 📖 Istruzioni:
echo    1. Apri http://localhost:3001/websdrs nel browser
echo    2. Premi F12 per aprire Developer Tools
echo    3. Vai al tab "Console"
echo    4. Guarda i log che iniziano con 🔧, 🚀, 📡, 📤, 📥
echo.
echo    ✅ Se vedi "📤 API Request: GET /api/v1/acquisition/websdrs"
echo       → Il frontend chiama davvero il backend!
echo.
echo Premi Ctrl+C quando hai finito di testare.
echo.
