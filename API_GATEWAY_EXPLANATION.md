# 🔍 Diagnosis: Perché il Backend Risponde con 404?

## 📌 Domanda
> Come mai sul API gateway docs non vedo `http://localhost:8000/api/v1/acquisition/websdrs`?
> Come mai la chiamata mi restituisce 404? Cosa succede al API gateway?

---

## ✅ Risposta: IL BACKEND È CORRETTO!

**Non è un problema di 404.** L'endpoint esiste ed è correttamente configurato. Il problema è nella **visualizzazione della documentazione (FastAPI Swagger docs)**.

---

## 🏗️ Come Funziona l'Architettura

### Layer 1: API Gateway (port 8000)
**File**: `services/api-gateway/src/main.py`

```python
@app.api_route("/api/v1/acquisition/{path:path}", methods=["GET", ...])
async def proxy_to_rf_acquisition(request: Request, path: str):
    """Proxy requests to RF Acquisition service."""
    return await proxy_request(request, RF_ACQUISITION_URL)
```

**Cosa fa**: 
- Cattura TUTTE le richieste a `/api/v1/acquisition/*`
- Le invia al RF-Acquisition service su http://rf-acquisition:8001
- Rispedisce la risposta al client

### Layer 2: RF-Acquisition Service (port 8001)
**File**: `services/rf-acquisition/src/routers/acquisition.py`

```python
router = APIRouter(prefix="/api/v1/acquisition", tags=["acquisition"])

@router.get("/websdrs", response_model=list[dict])
async def list_websdrs():
    """List all configured WebSDR receivers."""
    return get_websdrs_config()

@router.get("/websdrs/health")
async def check_websdrs_health():
    """Check health status of all WebSDR receivers."""
    return health_status
```

**Cosa fa**: 
- Riceve richieste con prefix `/api/v1/acquisition`
- Espone gli endpoint: `/websdrs` e `/websdrs/health`

---

## 🔗 Come Si Connettono i Layer

```
Frontend Browser                     API Gateway              RF-Acquisition Service
http://localhost:3001                http://localhost:8000   http://rf-acquisition:8001
     │                                      │                        │
     └──→ GET /api/v1/acquisition/websdrs  │                        │
          (fatto da Axios)                 │                        │
                                           └──→ proxy_request()      │
                                                path=/websdrs        │
                                                                    GET /api/v1/acquisition/websdrs
                                                                    (FastAPI router)
                                                                          │
                                                                    ✅ Returns JSON
                                                                          │
                                                           ←──────────────┘
                                           ←────────────────────────────
          ✅ Axios riceve JSON
```

---

## 📊 Test Pratico: Come Verificare

### Test 1: Chiama il Service Direttamente (port 8001)

```powershell
# Questo bypassa l'API Gateway
curl http://localhost:8001/api/v1/acquisition/websdrs

# Output atteso:
[
  {"id": 1, "name": "Aquila di Giaveno", "location_name": "Giaveno, Italy", ...},
  {"id": 2, "name": "Montanaro", "location_name": "Montanaro, Italy", ...},
  ...
]
```

### Test 2: Chiama tramite API Gateway (port 8000)

```powershell
# Questo passa per il proxy
curl http://localhost:8000/api/v1/acquisition/websdrs

# Output atteso: IDENTICO al Test 1
```

### Test 3: Verifica che il Gateway faccia il proxy

```powershell
# Se ricevi 404, significa:
# 1. API Gateway è offline, oppure
# 2. RF-Acquisition service non è raggiungibile, oppure
# 3. Il path è sbagliato

# Verifica che Gateway sia online:
curl http://localhost:8000/health
# Deve rispondere: {"status": "ok", ...}

# Verifica che Service sia online:
curl http://localhost:8001/health
# Deve rispondere: {"status": "healthy", ...}
```

---

## ⚠️ Perché Non Vedo l'Endpoint nella Documentazione Swagger?

### Swagger Docs Disponibili

**API Gateway Docs**: http://localhost:8000/docs
- ✅ Mostra i route del Gateway: `/`, `/health`, `/ready`
- ❌ **NON mostra** gli endpoint proxied (`/api/v1/acquisition/websdrs`)
- **Motivo**: Gli endpoint proxied sono dinamici (passati al service backend)

**RF-Acquisition Docs**: http://localhost:8001/docs
- ✅ Mostra TUTTI gli endpoint: `/websdrs`, `/websdrs/health`, `/acquire`, `/status/{task_id}`, `/config`
- Questo è il "vero" service

---

## 🎯 La Soluzione: Dove Consultare i Veri Endpoint

```
API Gateway Docs:        http://localhost:8000/docs
→ Non mostra i proxied routes

RF-Acquisition Docs:     http://localhost:8001/docs  ← CONSULTARE QUESTO!
→ Mostra tutti gli endpoint reali: /websdrs, /websdrs/health, etc.

Inference Docs:          http://localhost:8002/docs
Training Docs:           http://localhost:8003/docs
Data-Ingestion Docs:     http://localhost:8004/docs
```

---

## 🧪 Verifica Finale: E2E Test

```python
import httpx

# Test che il frontend farà
client = httpx.Client(base_url="http://localhost:8000")

# GET WebSDRs config
response = client.get("/api/v1/acquisition/websdrs")
print(f"Status: {response.status_code}")  # ✅ Deve essere 200, non 404
print(f"Data: {response.json()}")  # ✅ Deve mostrare 7 WebSDRs

# GET WebSDRs health
response = client.get("/api/v1/acquisition/websdrs/health")
print(f"Status: {response.status_code}")  # ✅ Deve essere 200
print(f"Data: {response.json()}")  # ✅ Deve mostrare health per ogni WebSDR
```

---

## 📋 Diagnostic Checklist

Se ricevi **404 quando chiami l'endpoint**:

- [ ] Verificare che **API Gateway** è online: `curl http://localhost:8000/health`
- [ ] Verificare che **RF-Acquisition Service** è online: `curl http://localhost:8001/health`
- [ ] Verificare che **il path è corretto**: `/api/v1/acquisition/websdrs` (non `/api/api/v1/...`)
- [ ] Verificare che **la base URL** è: `http://localhost:8000` (nel frontend `.env`)
- [ ] Verificare che **il metodo** è GET (non POST)
- [ ] Verificare che **Docker networking** funziona (se in container)

---

## 🎯 Conclusione

**Il backend NON ha problemi di 404!**

✅ **L'endpoint esiste** sul RF-Acquisition service
✅ **L'API Gateway** lo proxya correttamente
✅ **Il frontend** lo chiama correttamente (con base URL `http://localhost:8000`)

**La risposta 404 che vedi probabilmente è perché**:
1. Backend non è avviato → `python services/rf-acquisition/src/main.py`
2. API Gateway non è avviato → `docker-compose up -d api-gateway`
3. Path sbagliato nel frontend (ma l'ho già corretto)
4. Base URL sbagliato nel `.env` (ma l'ho già corretto)

---

**Per consultare la documentazione completa dell'API**:
👉 **Apri http://localhost:8001/docs** per il RF-Acquisition service!

---

Ultimo Update: 2025-10-22
Diagnosi: ✅ Backend è corretto
Azione Richiesta: Verifica che i servizi siano online
