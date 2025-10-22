# ============================================================================
# dev-setup.ps1 - Heimdall SDR Development Environment Setup
# ============================================================================
# Questo script automatizza la configurazione dell'ambiente di sviluppo su Windows
# Esecuzione: .\dev-setup.ps1 (con virtualenv attivo)
# ============================================================================

param(
    [switch]$SkipEnv = $false,        # Salta la configurazione .env
    [switch]$SkipDocker = $false,     # Salta la verifica Docker
    [switch]$SkipMicroservices = $false  # Salta l'installazione microservizi
)

$ErrorActionPreference = "Stop"
$WarningPreference = "SilentlyContinue"

# Colori per output
$Green = @{ ForegroundColor = 'Green' }
$Yellow = @{ ForegroundColor = 'Yellow' }
$Red = @{ ForegroundColor = 'Red' }
$Cyan = @{ ForegroundColor = 'Cyan' }
$Magenta = @{ ForegroundColor = 'Magenta' }

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "╔" + ("═" * 78) + "╗" @Cyan
    Write-Host "║ $Text" + (" " * (77 - $Text.Length)) + "║" @Cyan
    Write-Host "╚" + ("═" * 78) + "╝" @Cyan
    Write-Host ""
}

function Write-Success {
    param([string]$Text)
    Write-Host "[+] $Text" @Green
}

function Write-Warn {
    param([string]$Text)
    Write-Host "[!] $Text" @Yellow
}

function Write-Error-Custom {
    param([string]$Text)
    Write-Host "[-] $Text" @Red
}

function Write-Info {
    param([string]$Text)
    Write-Host "[*] $Text" @Cyan
}

# ============================================================================
# FASE 1: Verifica Prerequisites
# ============================================================================
Write-Header "FASE 1: Verifica Prerequisites"

# Verifica Python
Write-Info "Verificando Python..."
$pythonVersion = python --version 2>&1
if ($pythonVersion -match "Python (\d+\.\d+)") {
    Write-Success "Python trovato: $pythonVersion"
}
else {
    Write-Error-Custom "Python non trovato o non accessibile"
    exit 1
}

# Verifica virtualenv attivo
Write-Info "Verificando virtualenv..."
if ($env:VIRTUAL_ENV) {
    Write-Success "Virtualenv attivo: $env:VIRTUAL_ENV"
}
else {
    Write-Warn "Virtualenv non sembra attivo. Se necessario, esegui: .\.venv\Scripts\Activate.ps1"
}

# Verifica pip
Write-Info "Verificando pip..."
$pipVersion = pip --version 2>&1
if ($pipVersion -match "pip") {
    Write-Success "pip trovato: $pipVersion"
}
else {
    Write-Error-Custom "pip non trovato"
    exit 1
}

# ============================================================================
# FASE 2: Configurazione .env
# ============================================================================
if (-not $SkipEnv) {
    Write-Header "FASE 2: Configurazione .env"
    
    if (Test-Path ".env.example") {
        if (-not (Test-Path ".env")) {
            Write-Info "Copiando .env.example a .env..."
            Copy-Item ".env.example" ".env"
            Write-Success ".env creato da .env.example"
        }
        else {
            Write-Warn ".env già esiste (non sovrascritto)"
        }
    }
    else {
        Write-Warn ".env.example non trovato (creando .env minimo)"
        @"
# Heimdall SDR Development Environment
# ===================================

# PostgreSQL
POSTGRES_DB=heimdall
POSTGRES_USER=heimdall_user
POSTGRES_PASSWORD=changeme
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# RabbitMQ
RABBITMQ_DEFAULT_USER=guest
RABBITMQ_DEFAULT_PASS=guest
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672

# Redis
REDIS_PASSWORD=changeme
REDIS_HOST=localhost
REDIS_PORT=6379

# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_HOST=localhost
MINIO_PORT=9000

# MLflow
MLFLOW_TRACKING_URI=postgresql://heimdall_user:changeme@localhost:5432/heimdall_mlflow
MLFLOW_ARTIFACT_URI=s3://mlflow/artifacts
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

# API Gateway
API_GATEWAY_HOST=localhost
API_GATEWAY_PORT=8000

# Frontend
FRONTEND_HOST=localhost
FRONTEND_PORT=3000

# Environment
ENVIRONMENT=development
DEBUG=true
"@ | Out-File -FilePath ".env" -Encoding UTF8
        Write-Success ".env minimo creato"
    }
}

# ============================================================================
# FASE 3: Verifica Docker
# ============================================================================
if (-not $SkipDocker) {
    Write-Header "FASE 3: Verifica Docker"
    
    Write-Info "Verificando Docker..."
    $dockerVersion = docker --version 2>&1
    if ($dockerVersion -match "Docker version") {
        Write-Success "Docker trovato: $dockerVersion"
    }
    else {
        Write-Error-Custom "Docker non trovato o non accessibile"
        Write-Warn "Assicurati che Docker Desktop sia in esecuzione"
        exit 1
    }
    
    Write-Info "Verificando Docker daemon..."
    try {
        docker ps 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Docker daemon è in esecuzione"
        }
        else {
            Write-Warn "Docker daemon non risponde. Avvia Docker Desktop dal menu Start"
            Write-Warn "Continuo setup comunque (Docker sarà necessario dopo)"
        }
    }
    catch {
        Write-Warn "Docker daemon non risponde. Avvia Docker Desktop dal menu Start"
        Write-Warn "Continuo setup comunque (Docker sarà necessario dopo)"
    }
}

# ============================================================================
# FASE 4: Installazione Dipendenze Microservizi
# ============================================================================
if (-not $SkipMicroservices) {
    Write-Header "FASE 4: Installazione Dipendenze Microservizi"
    
    $services = @(
        "api-gateway",
        "rf-acquisition",
        "training",
        "inference",
        "data-ingestion-web"
    )
    
    $failedServices = @()
    
    foreach ($service in $services) {
        $reqPath = "services\$service\requirements.txt"
        
        if (Test-Path $reqPath) {
            Write-Info "Installando $service..."
            try {
                pip install -r $reqPath -q 2>&1 | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "$service`: dipendenze installate"
                }
                else {
                    Write-Error-Custom "$service`: errore installazione"
                    $failedServices += $service
                }
            }
            catch {
                Write-Error-Custom "$service`: eccezione $_"
                $failedServices += $service
            }
        }
        else {
            Write-Warn "$service`: requirements.txt non trovato (saltato)"
        }
    }
    
    if ($failedServices.Count -gt 0) {
        Write-Warn "Alcuni servizi hanno fallito: $($failedServices -join ', ')"
    }
    else {
        Write-Success "Tutte le dipendenze installate con successo"
    }
}

# ============================================================================
# FASE 5: Summary e Prossimi Passi
# ============================================================================
Write-Header "Setup Completato! 🎉"

Write-Info "Prossimi passi:"
Write-Host ""
Write-Host "1. Assicurati che Docker Desktop sia in esecuzione (menu Start > Docker Desktop)" @Magenta
Write-Host "2. Avvia l'infrastruttura Docker:" @Magenta
Write-Host "   docker compose up -d" @Cyan
Write-Host ""
Write-Host "3. Verifica che i servizi siano attivi:" @Magenta
Write-Host "   docker compose ps" @Cyan
Write-Host ""
Write-Host "4. Avvia i microservizi Python in nuove finestre:" @Magenta
Write-Host "   .\scripts\start-microservices.ps1" @Cyan
Write-Host ""
Write-Host "5. Accedi ai servizi:" @Magenta
Write-Host "   API Gateway:        http://localhost:8000/docs" @Cyan
Write-Host "   RF Acquisition:     http://localhost:8001/docs" @Cyan
Write-Host "   Training:           http://localhost:8002/docs" @Cyan
Write-Host "   Inference:          http://localhost:8003/docs" @Cyan
Write-Host "   Data Ingestion Web: http://localhost:8004/docs" @Cyan
Write-Host ""
Write-Host "6. Esegui i test di import:" @Magenta
Write-Host "   .\scripts\test-all-services.ps1" @Cyan
Write-Host ""

Write-Host "Per ulteriori dettagli: vedi SETUP.md" @Magenta
Write-Host ""
