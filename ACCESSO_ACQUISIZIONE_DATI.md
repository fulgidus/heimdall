# 🎯 Come Accedere al Frontend di Acquisizione Dati

**Data**: 22 Ottobre 2025  
**Status**: ✅ GUIDA OPERATIVA COMPLETA  
**Target**: Tutti gli utenti

---

## 📍 TL;DR - Accesso Rapido

```bash
# 1. Avvia il backend
docker-compose up -d

# 2. Avvia il frontend
cd frontend
npm run dev

# 3. Apri nel browser
http://localhost:5173

# 4. Login (credenziali di default)
Email: user@heimdall.dev
Password: password

# 5. Clicca "Data Ingestion" nella sidebar sinistra
```

**Fine**. Sei in acquisizione dati! ⚡

---

## 🚀 Guida Completa Step-by-Step

### Passo 1: Avvia il Backend & Infrastruttura

```powershell
# Da: C:\Users\aless\Documents\Projects\heimdall
docker-compose up -d

# Verifica che tutto sia up
docker-compose ps
```

**Output atteso**:
```
13 containers should show status "healthy" or "running"
- PostgreSQL ✅
- RabbitMQ ✅
- Redis ✅
- MinIO ✅
- Prometheus ✅
- Grafana ✅
- API Gateway ✅
- RF Acquisition ✅
- Training ✅
- Inference ✅
- Data Ingestion Web ✅
- pgAdmin ✅
- Redis Commander ✅
```

**Se non sono tutti healthy?** Aspetta 20 secondi e riconta. Docker prende tempo per fare health check.

---

### Passo 2: Avvia il Frontend React

```powershell
# Naviga in frontend
cd C:\Users\aless\Documents\Projects\heimdall\frontend

# Avvia dev server
npm run dev
```

**Output atteso**:
```
VITE v... dev server running at:

  ➜  Local:   http://localhost:5173/
  ➜  press h + enter to show help
```

---

### Passo 3: Apri il Browser

Apri: **http://localhost:5173**

Vedrai la **LOGIN PAGE**:
- Logo "Heimdall" con icona radar
- Campi: Email, Password
- Pulsante: "Sign In"

---

### Passo 4: Effettua il Login

**Credenziali di Default** (hardcoded in `Login.tsx`):

```
Email:    user@heimdall.dev
Password: password
```

Clicca: **"Sign In"** ✅

---

### Passo 5: Sei nel Dashboard!

Dopo login vedrai:
- **Sidebar sinistra** con menu di navigazione
- **Dashboard principale** con grafici e stats
- Icona **Radar viola** in alto a sinistra (logo Heimdall)

---

### Passo 6: Naviga a "Data Ingestion"

Nella **SIDEBAR SINISTRA** vedrai:

```
🏠 Dashboard       ← (sei qui al login)
📍 Localization
📻 Data Ingestion  ← CLICCA QUI
📊 Analytics
⚙️  Settings
```

**Clicca su**: **"📻 Data Ingestion"** 

---

### Passo 7: VOILÀ! Sei nell'Acquisizione Dati! 🎉

Vedrai la pagina Data Ingestion con 2 sezioni:

#### A. Recording Session Creator (Form)
- **Frequency (MHz)**: Default 145.5 (2m band italiana)
- **Duration (seconds)**: Default 10
- **Pulsante VERDE**: "START ACQUISITION"

#### B. Sessions Queue (Lista di acquisizioni)
- **Vuoto al primo accesso**
- Si popolerà quando avvii una sessione

---

## 🎬 Primo Test - Acquisizione Dati

### Passo 1: Configura l'Acquisizione

Nel form **Recording Session Creator**:
- ✅ Frequency: 145.5 MHz (OK così)
- ✅ Duration: 10 secondi (OK così)

### Passo 2: Clicca "START ACQUISITION"

```
┌─────────────────────────────┐
│  Recording Session Creator  │
├─────────────────────────────┤
│ Frequency (MHz):  145.5     │
│ Duration (s):     10        │
│                             │
│  [START ACQUISITION] ←     │
└─────────────────────────────┘
```

**Clicca il pulsante VERDE** ⚡

### Passo 3: Guarda la Queue Aggiornarsi

Nella sezione **Sessions Queue** vedrai una nuova riga:

```
Session ID: 1 (es.)
Status: ⏳ PENDING (dopo 2-3 sec) → 🔄 PROCESSING → ✅ COMPLETED
Frequency: 145.5 MHz
Duration: 10s
Created: 2025-10-22 14:30:45
```

### Passo 4: Attendi Completamento

- **PENDING**: 2-3 secondi (task in coda)
- **PROCESSING**: 70-80 secondi (acquisizione dai 7 WebSDR italiani in parallelo)
- **COMPLETED**: ✅ Dati salvati in MinIO e Database

**Timeline atteso**:
```
[Clicco] → 2sec → PENDING → 70sec → PROCESSING → ✅ COMPLETED
```

---

## 🔍 Dove Vanno i Dati Acquisiti?

### 1. Database PostgreSQL
```sql
-- Misurazioni salvate in:
SELECT * FROM measurements 
WHERE session_id = 1;
```

**Accedi via pgAdmin**:
- URL: http://localhost:5050
- Email: admin@pg.com
- Password: admin

### 2. MinIO (Object Storage)
```
Bucket: heimdall-raw-iq
Path: s3://heimdall-raw-iq/sessions/{session_id}/
```

**Accedi via UI MinIO**:
- URL: http://localhost:9001
- Username: minioadmin
- Password: minioadmin

### 3. Redis (Cache Task Results)
```bash
redis-cli -p 6379
KEYS *
```

**Accedi via Redis Commander**:
- URL: http://localhost:8081

---

## 🛠️ Troubleshooting - Se Non Funziona

### ❌ Pagina login non carica
**Soluzione**: 
```powershell
# Kill previous npm dev
# Pulisci cache
Remove-Item -Recurse node_modules
npm install
npm run dev
```

### ❌ "Cannot connect to API" error
**Soluzione**:
```powershell
# Verifica che backend sia up
docker-compose ps

# Se down:
docker-compose down
docker-compose up -d
```

### ❌ Login fallisce
**Soluzione**: 
- Username: **user@heimdall.dev** (case-sensitive!)
- Password: **password** (niente maiuscole)
- Clicca "Sign In" (non Enter)

### ❌ "START ACQUISITION" grayed out
**Soluzione**:
- Verifica che **Data Ingestion Web service** sia running:
```powershell
docker-compose ps | findstr data-ingestion
# Deve dire "running" o "healthy"
```

### ❌ Sessions non aggiornano da sole
**Soluzione** (aggiorna manualmente):
```powershell
# Premi F5 nel browser
# Oppure clicca il piccolo pulsante "refresh" se presente
```

---

## 📊 Componenti Che Vedi

### Sidebar Sinistra
```tsx
// Componente: DataIngestion.tsx (linea 65-100)
<aside className="w-64 bg-linear-to-b from-slate-900 to-slate-950">
  {/* Logo */}
  {/* Menu Items */}
  {/* User Dropdown */}
</aside>
```

**Elementi**:
- Logo Heimdall
- 5 menu item (Dashboard, Localization, Data Ingestion, Analytics, Settings)
- User dropdown (logout)

### Main Content Area
```tsx
// Componente: RecordingSessionCreator.tsx
// + SessionsList.tsx
```

**Elementi**:
1. **Form**: Frequency, Duration, Button
2. **Stats Card**: Total, Completed, Processing, Failed
3. **Queue List**: Tutte le acquisizioni

---

## 🔗 URL di Navigazione

| Pagina             | URL                                      | Accesso          |
| ------------------ | ---------------------------------------- | ---------------- |
| Frontend Home      | http://localhost:5173                    | Aperto (→ login) |
| Login              | http://localhost:5173                    | Pubblico         |
| Dashboard          | http://localhost:5173/dashboard          | Autenticato      |
| **Data Ingestion** | **http://localhost:5173/data-ingestion** | **Autenticato**  |
| Localization       | http://localhost:5173/localization       | Autenticato      |
| Analytics          | http://localhost:5173/analytics          | Autenticato      |
| Settings           | http://localhost:5173/settings           | Autenticato      |

---

## 📱 Mobile/Responsive

Il frontend è **responsive** (funziona su mobile):
- Sidebar si chiude su schermi piccoli
- Menu hamburger (☰) per aprirla
- Layout stack verticale

---

## 🎓 Per Capire Meglio

Leggi questi doc in ordine:

1. **START_HERE_DATA_INGESTION.md** (5 min) - Overview
2. **DATA_INGESTION_ITALIANO.md** (10 min) - Spiegazione completa
3. **DATA_INGESTION_COMPLETE_SUMMARY.md** (15 min) - Architettura
4. **DATA_INGESTION_IMPLEMENTATION.md** (20 min) - Dettagli tecnici

---

## ✅ Checklist di Setup

- [ ] Docker-compose up e tutti i container healthy
- [ ] Frontend npm run dev senza errori
- [ ] Browser apre http://localhost:5173 senza errori
- [ ] Login funziona con user@heimdall.dev / password
- [ ] Sidebar visibile e "Data Ingestion" cliccabile
- [ ] Pagina Data Ingestion carica il form
- [ ] Bottone "START ACQUISITION" è verde e cliccabile
- [ ] Clicco acquisizione e vedo session nella queue
- [ ] Session status cambia PENDING → PROCESSING → COMPLETED
- [ ] Dati visibili in MinIO http://localhost:9001

---

## 🎯 Prossimi Step

Dopo aver fatto funzionare l'acquisizione:

1. **Experiment**: Prova frequenze diverse (145.5, 145.575, ecc)
2. **Monitor**: Guarda i dati in MinIO e PostgreSQL
3. **Phase 5**: Training (il modello impara dai dati)
4. **Phase 6**: Inference (localizzazione in real-time)
5. **Phase 7**: Visualizzazione su mappa

---

## 💬 Contatti & Supporto

- **Bug?** Apri issue su GitHub
- **Domande?** Vedi `AGENTS.md` per ruoli
- **Contribuire?** Vedi `CONTRIBUTING.md`

---

**Buona acquisizione! 🚀**
