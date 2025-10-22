# Script per avviare tutti i 5 microservizi in parallelo

Write-Host "[MICROSERVICES LAUNCHER]" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan
Write-Host ""

$services = @(
    @{
        Name = "api-gateway"
        Path = "services/api-gateway"
        Port = 8000
        Description = "Main API Gateway"
    },
    @{
        Name = "rf-acquisition"
        Path = "services/rf-acquisition"
        Port = 8001
        Description = "WebSDR Data Collection"
    },
    @{
        Name = "training"
        Path = "services/training"
        Port = 8002
        Description = "ML Model Training"
    },
    @{
        Name = "inference"
        Path = "services/inference"
        Port = 8003
        Description = "ML Model Inference"
    },
    @{
        Name = "data-ingestion-web"
        Path = "services/data-ingestion-web"
        Port = 8004
        Description = "Data Ingestion Web UI"
    }
)

Write-Host "Starting services (each in new terminal)..." -ForegroundColor Yellow
Write-Host ""

foreach ($service in $services) {
    Write-Host "[STARTING] $($service.Name) - $($service.Description) (port $($service.Port))" -ForegroundColor Green
    
    # Crea comando per avviare il servizio
    $startCmd = "cd '$($service.Path)'; pip install -r requirements.txt -q; python -m uvicorn src.main:app --reload --port $($service.Port)"
    
    # Apri una nuova finestra PowerShell per ogni servizio
    Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", $startCmd -WindowStyle Normal
    
    Start-Sleep -Milliseconds 500
}

Write-Host ""
Write-Host "[INFO] All services are starting..." -ForegroundColor Cyan
Write-Host "[INFO] Check each terminal for startup messages" -ForegroundColor Cyan
Write-Host ""
Write-Host "API Gateway:          http://localhost:8000/docs" -ForegroundColor Magenta
Write-Host "RF Acquisition:       http://localhost:8001/docs" -ForegroundColor Magenta
Write-Host "Training:             http://localhost:8002/docs" -ForegroundColor Magenta
Write-Host "Inference:            http://localhost:8003/docs" -ForegroundColor Magenta
Write-Host "Data Ingestion Web:   http://localhost:8004/docs" -ForegroundColor Magenta
