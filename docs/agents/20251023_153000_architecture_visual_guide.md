# 🔗 API Gateway Routing: Visual Explanation

## Cosa Succede Quando Chiami `/api/v1/acquisition/websdrs`

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          BROWSER (port 3001)                            │
│                                                                          │
│  http://localhost:3001/websdrs                                          │
│  ↓                                                                       │
│  Frontend Component WebSDRManagement.tsx                                │
│  ↓                                                                       │
│  axios.get('/api/v1/acquisition/websdrs')                               │
│  ↓                                                                       │
│  Base URL: http://localhost:8000                                        │
│  Full URL: http://localhost:8000/api/v1/acquisition/websdrs             │
└────────────────────┬────────────────────────────────────────────────────┘
                     │ HTTP GET Request
                     ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    API GATEWAY (port 8000)                              │
│                                                                          │
│  Incoming: GET http://localhost:8000/api/v1/acquisition/websdrs         │
│  ↓                                                                       │
│  Routes.py Line 79:                                                     │
│  @app.api_route("/api/v1/acquisition/{path:path}", methods=["GET"...]) │
│  async def proxy_to_rf_acquisition(request, path: str):                 │
│  ↓                                                                       │
│  path = "websdrs"                                                       │
│  target_url = "http://rf-acquisition:8001"                              │
│  full_url = f"{target_url}/api/v1/acquisition/{path}"                   │
│  full_url = "http://rf-acquisition:8001/api/v1/acquisition/websdrs"     │
│  ↓                                                                       │
│  await proxy_request(request, "http://rf-acquisition:8001")             │
│  ↓                                                                       │
│  async with httpx.AsyncClient() as client:                              │
│      response = client.request(method=GET, url=full_url, ...)           │
│                                                                          │
│  ⚠️  QUESTO È UN PROXY - non è un endpoint reale del gateway!            │
│  ⚠️  Quindi non lo vedi in http://localhost:8000/docs                    │
│                                                                          │
└────────────────────┬────────────────────────────────────────────────────┘
                     │ Forward HTTP GET
                     │ "http://rf-acquisition:8001/api/v1/acquisition/websdrs"
                     ↓
┌─────────────────────────────────────────────────────────────────────────┐
│              RF-ACQUISITION SERVICE (port 8001)                         │
│                                                                          │
│  Incoming: GET /api/v1/acquisition/websdrs                              │
│  ↓                                                                       │
│  routers/acquisition.py Line ~245:                                      │
│  router = APIRouter(prefix="/api/v1/acquisition", tags=["acquisition"]) │
│  ↓                                                                       │
│  @router.get("/websdrs", response_model=list[dict])                     │
│  async def list_websdrs():                                              │
│      """List all configured WebSDR receivers."""                        │
│      return get_websdrs_config()                                        │
│  ↓                                                                       │
│  Returns: [                                                             │
│      {"id": 1, "name": "Aquila di Giaveno", ...},                       │
│      {"id": 2, "name": "Montanaro", ...},                               │
│      ...                                                                │
│      {"id": 7, "name": "Milano - Baggio", ...}                          │
│  ]                                                                      │
│  Status: 200 OK                                                        │
│                                                                          │
│  ✅ QUESTO ENDPOINT È REALE - lo vedi in http://localhost:8001/docs     │
│                                                                          │
└────────────────────┬────────────────────────────────────────────────────┘
                     │ HTTP Response (JSON)
                     │ Status 200 OK
                     ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    API GATEWAY (port 8000)                              │
│                                                                          │
│  Receives response from RF-Acquisition service                          │
│  ↓                                                                       │
│  return JSONResponse(                                                   │
│      content=response.json(),                                           │
│      status_code=response.status_code,                                  │
│      headers=dict(response.headers)                                     │
│  )                                                                      │
│                                                                          │
│  ✅ Gateway is transparent - just passes through the response           │
│                                                                          │
└────────────────────┬────────────────────────────────────────────────────┘
                     │ HTTP Response (JSON)
                     │ Status 200 OK
                     ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                          BROWSER (port 3001)                            │
│                                                                          │
│  axios.get() completes successfully                                     │
│  ↓                                                                       │
│  response.status = 200                                                  │
│  response.data = [7 WebSDRs JSON]                                       │
│  ↓                                                                       │
│  setWebSdrs(extended)                                                   │
│  ↓                                                                       │
│  Component re-renders                                                   │
│  ↓                                                                       │
│  ✅ Page shows 7 WebSDRs with real data!                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Perché Non Vedo l'Endpoint in Gateway Swagger?

### Opzione A: Endpoint Fisso (definito direttamente)
```python
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# ✅ Lo vedi in http://localhost:8000/docs
```

### Opzione B: Endpoint Proxy (inoltrato dinamicamente)
```python
@app.api_route("/api/v1/acquisition/{path:path}", ...)
async def proxy_to_rf_acquisition(request, path):
    # path potrebbe essere "websdrs", "health", "acquire", etc.
    # Dipende da cosa il service backend definisce
    return await proxy_request(request, ...)

# ❌ NON lo vedi in http://localhost:8000/docs
# Perché FastAPI non sa in advance quali path il backend supporta
```

**Soluzione**: Consulta il service direttamente!
```
Gateway docs: http://localhost:8000/docs        (non mostra proxy)
Service docs: http://localhost:8001/docs        (mostra TUTTI gli endpoint)
```

---

## Path Completo per Ogni Service

```
Acquisizione:     /api/v1/acquisition/*
                  Proxy target: http://rf-acquisition:8001

Inference:        /api/v1/inference/*
                  Proxy target: http://inference:8002

Training:         /api/v1/training/*
                  Proxy target: http://training:8003

Data Ingestion:   /api/v1/sessions/*
                  Proxy target: http://data-ingestion-web:8004
```

Tutti passano per il Gateway (8000), poi vengono inoltrati ai service backend.

---

## Debug: Come Vedere Cosa Succede

### Scenario 1: Test Diretto al Service (Bypassa Gateway)
```powershell
curl http://localhost:8001/api/v1/acquisition/websdrs
# Risposta diretta dal service
```

### Scenario 2: Test Via Gateway (Come Fa il Frontend)
```powershell
curl http://localhost:8000/api/v1/acquisition/websdrs
# Passa per il proxy del gateway, poi al service
```

Entrambi dovrebbero tornare lo **stesso JSON** e status 200.

### Scenario 3: Se Vedi 404
```
❌ Significa che il path non è raggiunto
```

Possibili cause:
1. Gateway non è online → `docker-compose up -d api-gateway`
2. Service non è online → `docker-compose up -d rf-acquisition`
3. Path è sbagliato nel frontend (ma dovrebbe essere `/api/v1/acquisition/websdrs`)
4. Service non ha definito this endpoint (ma lo ha, è nella linea ~245 di acquisition.py)

---

## Flusso Completo: Frontend → Backend

```
1. Browser (3001)
   ├─ Component: WebSDRManagement.tsx
   ├─ useEffect: chiama webSDRService.getWebSDRs()
   └─ Axios: GET http://localhost:8000/api/v1/acquisition/websdrs

2. Network Layer
   ├─ HTTP GET request
   ├─ Destination: 127.0.0.1:8000 (API Gateway)
   └─ Path: /api/v1/acquisition/websdrs

3. API Gateway (8000)
   ├─ Route Match: /api/v1/acquisition/{path:path}
   ├─ Extract: path = "websdrs"
   ├─ Proxy: forward to http://rf-acquisition:8001/api/v1/acquisition/websdrs
   └─ Wait for response

4. RF-Acquisition Service (8001)
   ├─ Route Match: prefix=/api/v1/acquisition + @router.get("/websdrs")
   ├─ Handler: async def list_websdrs()
   ├─ Logic: return get_websdrs_config()
   ├─ Response: [7 WebSDRs as JSON]
   └─ Status: 200 OK

5. API Gateway (8000) - Returns Response
   ├─ Receives: 200 OK + JSON
   ├─ Transform: JSONResponse(content, status_code, headers)
   └─ Send back to browser

6. Browser (3001)
   ├─ Axios interceptor: logs "📥 API Response: {status: 200, ...}"
   ├─ Response handler: response.data = [7 WebSDRs]
   ├─ useState: setWebSdrs(extended)
   ├─ Component re-render
   └─ UI: Table mostra 7 WebSDRs reali ✅
```

---

**Conclusione**: 
- ✅ L'endpoint **esiste** nel RF-Acquisition service
- ✅ L'API Gateway **lo proxya** correttamente
- ⚠️ Non lo vedi in Gateway Swagger perché è **dinamico**
- 👉 **Consulta http://localhost:8001/docs per la vera lista di endpoint**

---

Ultimo Update: 2025-10-22
