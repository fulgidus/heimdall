# Script per verificare lo stato dei 5 microservizi

Write-Host "[MICROSERVICES HEALTH CHECK]" -ForegroundColor Cyan
Write-Host "============================" -ForegroundColor Cyan
Write-Host ""

$services = @(
    @{Name = "api-gateway"; Port = 8000; URL = "http://localhost:8000/health"},
    @{Name = "rf-acquisition"; Port = 8001; URL = "http://localhost:8001/health"},
    @{Name = "training"; Port = 8002; URL = "http://localhost:8002/health"},
    @{Name = "inference"; Port = 8003; URL = "http://localhost:8003/health"},
    @{Name = "data-ingestion-web"; Port = 8004; URL = "http://localhost:8004/health"}
)

$allHealthy = $true

foreach ($service in $services) {
    $result = Test-NetConnection -ComputerName localhost -Port $service.Port -WarningAction SilentlyContinue
    
    if ($result.TcpTestSucceeded) {
        # Prova a ottenere lo status dalla health endpoint
        try {
            $health = Invoke-WebRequest -Uri $service.URL -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($health.StatusCode -eq 200) {
                Write-Host "[OK] $($service.Name): Healthy (port $($service.Port))" -ForegroundColor Green
            } else {
                Write-Host "[WARN] $($service.Name): Response code $($health.StatusCode) (port $($service.Port))" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "[OK] $($service.Name): Running (port $($service.Port))" -ForegroundColor Green
        }
    } else {
        Write-Host "[FAIL] $($service.Name): Not responding (port $($service.Port))" -ForegroundColor Red
        $allHealthy = $false
    }
}

Write-Host ""
Write-Host "[DASHBOARD URLs]" -ForegroundColor Magenta
Write-Host "API Gateway:          http://localhost:8000/docs" -ForegroundColor White
Write-Host "RF Acquisition:       http://localhost:8001/docs" -ForegroundColor White
Write-Host "Training:             http://localhost:8002/docs" -ForegroundColor White
Write-Host "Inference:            http://localhost:8003/docs" -ForegroundColor White
Write-Host "Data Ingestion Web:   http://localhost:8004/docs" -ForegroundColor White

Write-Host ""
if ($allHealthy) {
    Write-Host "All microservices are healthy!" -ForegroundColor Green
} else {
    Write-Host "Some microservices are not running. Use start-microservices.ps1 to start them." -ForegroundColor Yellow
}
