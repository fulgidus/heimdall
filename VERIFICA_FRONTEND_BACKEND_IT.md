# 🔗 Guida: Verifica che Frontend Chiama Davvero il Backend

**Status**: ✅ Debug logging aggiunto a tutti i layer

---

## TL;DR - Cosa Fare Subito

```powershell
# 1. Assicurati che il backend gira
python c:\Users\aless\Documents\Projects\heimdall\test_backend_connectivity.py

# 2. Se il test fallisce → avvia il backend
cd c:\Users\aless\Documents\Projects\heimdall\services\rf-acquisition
python src/main.py

# 3. Verifica Frontend (in nuova finestra PowerShell)
cd c:\Users\aless\Documents\Projects\heimdall\frontend
npm run dev

# 4. Apri http://localhost:3001/websdrs
# 5. Premi F12 → Console → Guarda i log

# Se vedi log come "📤 API Request: GET /api/v1/acquisition/websdrs"
# → ✅ Frontend chiama davvero il backend!
```

---

## 🏗️ Architettura: Come Comunica Frontend con Backend

```
┌─────────────────────────────┐
│  Browser (port 3001)        │
│  React App                  │
│  - http://localhost:3001    │
└──────────────┬──────────────┘
               │
               │ HTTP Request
               │ GET http://localhost:8000/api/v1/acquisition/websdrs
               ↓
┌─────────────────────────────┐
│  API Gateway (port 8000)    │
│  - FastAPI                  │
│  - CORS enabled             │
│  - Routes to services       │
└──────────────┬──────────────┘
               │
               │ Forward
               ↓
┌─────────────────────────────┐
│  RF-Acquisition Service     │
│  - Port 8001                │
│  - GET /api/v1/websdrs      │
│  - GET /api/v1/websdrs/health
│  - Returns: JSON            │
└─────────────────────────────┘
```

### Files Modificati per Debug

Ho aggiunto `console.log()` a 3 layer:

**1. `frontend/src/lib/api.ts`** - Livello Axios
```typescript
// Mostra: URL completo, base URL, ambiente
// Mostra: Richieste e risposte HTTP
// Mostra: Errori di connessione
```

**2. `frontend/src/pages/WebSDRManagement.tsx`** - Livello Component
```typescript
// Mostra: Quando useEffect si avvia
// Mostra: Quando chiama i servizi
// Mostra: Dati ricevuti
// Mostra: Quando fa auto-refresh
```

**3. `frontend/src/services/api/websdr.ts`** - Livello Service
```typescript
// Mostra: Esattamente quale endpoint chiama
// Mostra: Quanti items riceve
```

---

## ✅ Checklist: Frontend → Backend

### 1. Backend Deve Girare

```powershell
# Test manuale (simula una richiesta di curl)
python test_backend_connectivity.py
```

**Output Atteso**: 
```
✅ Testing: Get all WebSDRs
   Status 200 OK
   Received 7 items

✅ Testing: Check WebSDR health status
   Status 200 OK
   Data type: dict
```

**Output Non Atteso**:
```
❌ Connection Error
   [Errno 10061] No connection could be made because the target machine actively refused it
```

### 2. Frontend Dev Server Deve Girare

```powershell
cd c:\Users\aless\Documents\Projects\heimdall\frontend
npm run dev
```

**Output Atteso**:
```
VITE v4.x.x  ready in xxx ms

➜  Local:   http://localhost:3001/
➜  press h to show help
```

### 3. Browser Console Deve Mostrare i Log

Quando vai a http://localhost:3001/websdrs e apri F12:

**Console Tab - Output Atteso:**

```
🔧 API Configuration: {VITE_API_URL: 'http://localhost:8000', API_BASE_URL: 'http://localhost:8000', environment: 'development', isDev: true}

🔄 WebSDRManagement: setuping useEffect - caricamento iniziale

📡 WebSDRService.getWebSDRs(): calling GET /api/v1/acquisition/websdrs

📤 API Request: {method: 'GET', url: '/api/v1/acquisition/websdrs', fullURL: 'http://localhost:8000/api/v1/acquisition/websdrs'}

📥 API Response: {status: 200, url: '/api/v1/acquisition/websdrs', dataSize: 2847}

✅ WebSDRService.getWebSDRs(): ricevuti 7 WebSDRs

🏥 WebSDRService.checkWebSDRHealth(): calling GET /api/v1/acquisition/websdrs/health

📤 API Request: {method: 'GET', url: '/api/v1/acquisition/websdrs/health', fullURL: 'http://localhost:8000/api/v1/acquisition/websdrs/health'}

📥 API Response: {status: 200, url: '/api/v1/acquisition/websdrs/health', dataSize: 512}

✅ WebSDRService.checkWebSDRHealth(): ricevuto health status

📊 WebSDRs estesi (merged): (7) [...]

✅ WebSDRManagement: caricamento completato
```

**Se vedi questo** → ✅ **Frontend chiama davvero il backend!**

---

## ❌ Troubleshooting: Cosa Fare Se Fallisce

### Scenario 1: Log Console Mostra ❌ API Error

```
❌ API Error: {
  status: undefined,
  message: 'connect ECONNREFUSED 127.0.0.1:8000',
  data: undefined
}
```

**Significa**: Backend non è in ascolto su port 8000.

**Soluzione**:
```powershell
# Verifica se qualcosa gira sulla porta
netstat -ano | findstr :8000
# Se niente, avvia il backend:
cd c:\Users\aless\Documents\Projects\heimdall\services\rf-acquisition
python src/main.py
# Dovresti vedere: "Application startup complete"
```

---

### Scenario 2: CORS Error nella Console

```
❌ API Error: {
  message: 'Access to XMLHttpRequest from origin http://localhost:3001
   has been blocked by CORS policy'
}
```

**Significa**: Backend non permette richieste da localhost:3001.

**Soluzione**:
```bash
# File: services/api-gateway/src/main.py (oppure rf-acquisition)
# Verificare che CORS sia abilitato per il frontend

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Se non è abilitato → aggiungerlo e riavviare il backend.

---

### Scenario 3: Console Mostra Status 404

```
❌ API Error: {
  status: 404,
  message: 'Not Found',
  url: '/api/v1/acquisition/websdrs'
}
```

**Significa**: L'endpoint non esiste.

**Verifiche**:
```powershell
# 1. Verifica endpoint esatto
curl http://localhost:8000/api/v1/acquisition/websdrs

# 2. Verifica se API Gateway sta routando correttamente
curl http://localhost:8001/api/v1/acquisition/websdrs  # Direct to service

# 3. Verifica configurazione API Gateway
# File: services/api-gateway/src/routers/acquisition.py
# Deve avere: @router.get("/websdrs")
```

---

### Scenario 4: Console Non Mostra Nessun Log

```
[Silenzio totale in Console]
```

**Significa**: 
- Frontend non è in reload
- Oppure i file modificati non sono stati caricati

**Soluzione**:
```powershell
# Arresta Vite dev server (Ctrl+C)
cd c:\Users\aless\Documents\Projects\heimdall\frontend

# Cancella cache
rm -r node_modules\.vite

# Riavvia
npm run dev

# Ricarica pagina nel browser (F5)
```

---

### Scenario 5: Browser Mostra Dati Statici/Mock

Se la pagina mostra i 7 WebSDRs ma hanno sempre lo stesso status "online" e stesso SNR:

**Significa**: Codice non è stato aggiornato nel browser.

**Soluzione**:
```
1. Ctrl+Shift+Delete (Cache Browser)
2. Oppure: Ctrl+Shift+R (Force Reload)
3. Oppure: DevTools → Network → Disable Cache (mentre aperto)
```

---

## 🔍 Network Tab DevTools: Cosa Guardare

1. Apri **DevTools (F12)**
2. Vai al tab **"Network"**
3. Ricarica la pagina **F5**
4. Guarda le richieste:

**Atteso**:
```
Request URL: http://localhost:8000/api/v1/acquisition/websdrs
Status:      200
Size:        2.8 KB
Time:        125 ms
Response:    [JSON Array di 7 WebSDRs]
```

**Non Atteso**:
```
❌ (pending) - Non completa
❌ (canceled) - Browser cancella la richiesta
❌ 404 - Endpoint non trovato
❌ 500 - Errore server
❌ CORS error - Problema CORS
```

---

## 📝 Comando Curl per Testare Manualmente

```powershell
# Test 1: Get WebSDRs
curl -v http://localhost:8000/api/v1/acquisition/websdrs

# Test 2: Get Health
curl -v http://localhost:8000/api/v1/acquisition/websdrs/health

# Test 3: Con verbose (mostra headers)
curl -v -H "Content-Type: application/json" http://localhost:8000/api/v1/acquisition/websdrs
```

**Atteso**: 
```
< HTTP/1.1 200 OK
< content-type: application/json
< 
[
  {"id": 1, "name": "Turin", ...},
  {"id": 2, "name": "Milan", ...},
  ...
]
```

---

## 🎯 Verifica Passo-Passo (3 minuti)

```powershell
# 1. Test Backend (30 secondi)
python test_backend_connectivity.py

# Se non funziona: avvia il backend
cd services\rf-acquisition
python src\main.py
# Aspetta 3-5 secondi per startup

# 2. Avvia Frontend (in nuova finestra)
cd frontend
npm run dev

# 3. Apri Browser (30 secondi)
# - http://localhost:3001/websdrs
# - F12 per Console
# - Scroll su nei log per vedere il config

# 4. Guarda i Log (1 minuto)
# - Se vedi "📤 API Request: GET" → ✅ Funziona!
# - Se vedi "❌ API Error" → Leggi Troubleshooting

# 5. Verifica Pagina
# - Se mostra 7 WebSDRs con dati reali → ✅ Done!
# - Se mostra Mock/placeholder → Cache issue, F5 hard reload
```

---

## 🚀 Se Tutto Funziona

```
✅ Console mostra log senza errori
✅ Network tab mostra GET requests con status 200
✅ Pagina mostra 7 WebSDRs reali (non mock)
✅ Ogni 30s vedi "🔄 WebSDRManagement: auto-refresh"
```

**Successo!** 🎉 Il frontend chiama davvero il backend.

---

## 📞 Per Supporto

Se qualcosa non funziona, cattura e condividi:

1. **Console Log** (F12 → Console → Select All → Copy)
2. **Network Tab** (F12 → Network → Reload → Screenshot)
3. **Terminal Backend** (Output completo dell'avvio)
4. **Output Script Test**:
   ```powershell
   python test_backend_connectivity.py
   ```

Con questi dati posso diagnosticare il problema specifico.

---

**Ultimo Aggiornamento**: 2025-10-23
**Modifiche**: 3 file (api.ts, WebSDRManagement.tsx, websdr.ts)
**Stato**: ✅ Pronto per verifica runtime
