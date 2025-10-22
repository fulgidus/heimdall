Write-Host "[HEALTH CHECK] Heimdall Infrastructure Status" -ForegroundColor Cyan;
Write-Host "==============================================" -ForegroundColor Cyan;
Write-Host "";

$services = @(
    @{Name="PostgreSQL"; Port=5432},
    @{Name="RabbitMQ"; Port=5672},
    @{Name="Redis"; Port=6379},
    @{Name="MinIO"; Port=9000},
    @{Name="Prometheus"; Port=9090},
    @{Name="Grafana"; Port=3000}
)

foreach ($service in $services) {
    $result = Test-NetConnection -ComputerName localhost -Port $service.Port -WarningAction SilentlyContinue;
    if ($result.TcpTestSucceeded) {
        Write-Host "[OK] $($service.Name): Running (port $($service.Port))" -ForegroundColor Green;
    } else {
        Write-Host "[FAIL] $($service.Name): Not responding (port $($service.Port))" -ForegroundColor Red;
    }
}

Write-Host "";
Write-Host "[INFO] Docker Container Status:";
docker-compose ps