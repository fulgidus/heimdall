# Script per testare se tutti i 5 microservizi si avviano correttamente

Write-Host "[MICROSERVICES TEST]" -ForegroundColor Cyan
Write-Host "====================" -ForegroundColor Cyan
Write-Host ""

$services = @(
    @{Name = "api-gateway"; Path = "services/api-gateway"; Port = 8000},
    @{Name = "rf-acquisition"; Path = "services/rf-acquisition"; Port = 8001},
    @{Name = "training"; Path = "services/training"; Port = 8002},
    @{Name = "inference"; Path = "services/inference"; Port = 8003},
    @{Name = "data-ingestion-web"; Path = "services/data-ingestion-web"; Port = 8004}
)

$allPassed = $true

foreach ($service in $services) {
    Write-Host "[TEST] $($service.Name)..." -ForegroundColor Yellow
    
    $result = & cmd /c "cd `"$($service.Path)`" && python -c `"from src.main import app; print('[PASS]')`"" 2>&1
    
    if ($result -like "*[PASS]*") {
        Write-Host "[PASS] $($service.Name): Import successful" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] $($service.Name): $result" -ForegroundColor Red
        $allPassed = $false
    }
}

Write-Host ""
if ($allPassed) {
    Write-Host "All services passed import tests!" -ForegroundColor Green
} else {
    Write-Host "Some services failed! Review errors above." -ForegroundColor Red
}
