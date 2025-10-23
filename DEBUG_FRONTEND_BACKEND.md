# 🔍 Frontend → Backend Debug Guide

## Status: DEBUGGING IN PROGRESS ✅

Ho aggiunto **debug logging completo** per tracciare ogni API call. Quando apri http://localhost:3001/websdrs, vedrai nella **Console del Browser** (DevTools F12):

```
🔧 API Configuration: {...}
🚀 WebSDRManagement: inizio caricamento WebSDRs...
📡 Chiamata: webSDRService.getWebSDRs()
📤 API Request: {method: 'GET', url: '/api/v1/acquisition/websdrs', fullURL: 'http://localhost:8000/api/v1/acquisition/websdrs'}
[aspetta risposta...]
📥 API Response: {status: 200, url: '/api/v1/acquisition/websdrs', dataSize: 1234}
✅ WebSDRs config ricevuti: [7 WebSDRs...]
```

## 🛠️ Step-by-Step: Come Verificare

### 1️⃣ **Assicurati Backend Sia Running**

```powershell
# Verificare che il backend risponde
curl http://localhost:8000/api/v1/acquisition/websdrs
# Deve rispondere con JSON array di 7 WebSDRs (HTTP 200)
```

Se fallisce:
```powershell
# Vai nella cartella del backend
cd c:\Users\aless\Documents\Projects\heimdall\services\rf-acquisition

# Avvia il servizio
python src/main.py
# Oppure con Celery dual-mode
python src/entrypoint.py

# Verifica di nuovo
curl http://localhost:8000/api/v1/acquisition/websdrs
```

### 2️⃣ **Verification Frontend Sia Running su Port 3001**

```powershell
# Nel terminal, vai alla cartella frontend
cd c:\Users\aless\Documents\Projects\heimdall\frontend

# Avvia Vite dev server
npm run dev
# Dovresti vedere: "Local: http://localhost:3001"
```

### 3️⃣ **Apri Browser e Accedi a `/websdrs`**

1. Apri http://localhost:3001/websdrs
2. Premi **F12** per aprire Developer Tools
3. Vai al tab **"Console"**
4. **Guarda i log** che iniziano con 🔧, 🚀, 📡

### 4️⃣ **Analizza i Log Console**

#### ✅ **Scenario: Frontend → Backend funziona**

```
🔧 API Configuration: {
  VITE_API_URL: 'http://localhost:8000',
  API_BASE_URL: 'http://localhost:8000',
  environment: 'development',
  isDev: true
}

🚀 WebSDRManagement: setuping useEffect - caricamento iniziale

📡 WebSDRService.getWebSDRs(): calling GET /api/v1/acquisition/websdrs

📤 API Request: {
  method: 'GET',
  url: '/api/v1/acquisition/websdrs',
  fullURL: 'http://localhost:8000/api/v1/acquisition/websdrs'
}

📥 API Response: {
  status: 200,
  url: '/api/v1/acquisition/websdrs',
  dataSize: 2847
}

✅ WebSDRService.getWebSDRs(): ricevuti 7 WebSDRs

🏥 WebSDRService.checkWebSDRHealth(): calling GET /api/v1/acquisition/websdrs/health

📤 API Request: {
  method: 'GET',
  url: '/api/v1/acquisition/websdrs/health',
  fullURL: 'http://localhost:8000/api/v1/acquisition/websdrs/health'
}

📥 API Response: {
  status: 200,
  url: '/api/v1/acquisition/websdrs/health',
  dataSize: 512
}

✅ WebSDRService.checkWebSDRHealth(): ricevuto health status

📊 WebSDRs estesi (merged): [Array of 7 objects]

✅ WebSDRManagement: caricamento completato
```

**Risultato atteso**: Tabella con 7 WebSDRs + card riassuntivo

---

#### ❌ **Scenario: Backend è offline**

```
🔧 API Configuration: {...}
🚀 WebSDRManagement: setuping useEffect - caricamento iniziale
📡 Chiamata: webSDRService.getWebSDRs()
📤 API Request: {method: 'GET', url: '/api/v1/acquisition/websdrs', ...}

❌ API Error: {
  status: undefined,
  url: '/api/v1/acquisition/websdrs',
  message: 'connect ECONNREFUSED 127.0.0.1:8000',
  data: undefined
}

❌ Errore caricamento WebSDRs: Error: connect ECONNREFUSED 127.0.0.1:8000
```

**Azione**: Avvia il backend (vedi Step 1)

---

#### ❌ **Scenario: CORS Error**

```
❌ API Error: {
  status: 0,
  message: 'Network Error',
  data: 'Access to XMLHttpRequest has been blocked by CORS policy...'
}
```

**Azione**: Verification CORS nel backend:

```bash
# File: services/rf-acquisition/src/main.py

# Deve avere CORS per http://localhost:3001
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

#### ❌ **Scenario: Wrong Base URL**

Se vedi log come this:
```
📤 API Request: {
  url: '/api/v1/acquisition/websdrs',
  fullURL: 'http://localhost:WRONG_PORT/api/v1/acquisition/websdrs'
}
```

**Azione**: Verification `.env`:

```bash
# frontend/.env deve avere:
VITE_API_URL=http://localhost:8000

# Poi:
npm run dev  # Restart Vite dev server!
```

---

### 5️⃣ **Verification da Terminal (Curl)**

```powershell
# Testa gli stessi endpoint che il frontend chiama

# 1. Get WebSDRs config
$response = curl.exe -X GET http://localhost:8000/api/v1/acquisition/websdrs
Write-Host $response

# 2. Check WebSDR Health
$response = curl.exe -X GET http://localhost:8000/api/v1/acquisition/websdrs/health
Write-Host $response
```

---

### 6️⃣ **DevTools Network Tab**

1. Apri http://localhost:3001/websdrs
2. Premi F12 → tab "Network"
3. Guarda le richieste:

**Atteso**:
```
GET /api/v1/acquisition/websdrs       → 200 OK
GET /api/v1/acquisition/websdrs/health → 200 OK
```

**Non atteso**:
```
❌ (canceled)
❌ CORS error
❌ 404 Not Found
❌ 500 Internal Server Error
```

---

## 🚀 Se Tutto Funziona

1. ✅ Console mostra log senza ❌
2. ✅ Network tab mostra 200 OK
3. ✅ Pagina visualizza tabella con 7 WebSDRs reali
4. ✅ Ogni 30 secondi: `🔄 WebSDRManagement: auto-refresh`

**La sfida è completata!** 🎉

---

## 🐛 Se Qualcosa Non Funziona

**Fornisci i log da Console** (copia-incolla all il testo dalla Console F12) in modo che posso diagnosticare il problema specifico.

### Debug Checklist

- [ ] Backend è in esecuzione su port 8000
- [ ] Frontend è in esecuzione su port 3001
- [ ] Console mostra i log 🔧 API Configuration
- [ ] Network tab mostra GET requests a `/api/v1/acquisition/websdrs`
- [ ] Network requests tornano 200 OK (non 404, 500, CORS error)
- [ ] Pagina visualizza dati reali (non mock/placeholder)

---

## 📝 Files Modified for Debug

1. ✅ `frontend/src/lib/api.ts` - Added request/response interceptor logging
2. ✅ `frontend/src/pages/WebSDRManagement.tsx` - Added console.log in useEffect
3. ✅ `frontend/src/services/api/websdr.ts` - Added console.log nei metodi API

All files are in the repo. Just restart frontend dev server.
