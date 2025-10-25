# 🧪 WebSDR Management - Istruzioni di Test

**Obiettivo**: Verificare che la pagina `/websdrs` carichi dati **reali** dal backend (NON mockati)

---

## ✅ Checklist Pre-Test

- [ ] Backend è in esecuzione (`docker-compose ps` mostra tutti i servizi UP)
- [ ] Frontend è in esecuzione su `http://localhost:3001`
- [ ] Browser Developer Tools è accessibile (F12)
- [ ] `.env` del frontend ha `VITE_API_URL=http://localhost:8000` (senza `/api`)

---

## 🔍 Test 1: Verificare le Chiamate API nel Network Tab

### Procedura:

1. **Apri browser** su `http://localhost:3001/websdrs`

2. **Apri Developer Tools** (F12 o Ctrl+Shift+I)

3. **Vai alla tab "Network"** (Network tab)

4. **Filtra per "acquisition"** (nel filtro di ricerca)

5. **Aggiorna la pagina** (F5 o Ctrl+R)

### Cosa dovresti vedere:

✅ **Due richieste**:

```
GET http://localhost:8000/api/v1/acquisition/websdrs
Status: 200 OK
Response type: JSON
Response preview:
    [
        {
            "id": 1,
            "name": "Aquila di Giaveno",
            "url": "http://sdr1.ik1jns.it:8076/",
            "location_name": "Giaveno, Italy",
            "latitude": 45.02,
            "longitude": 7.29,
            "is_active": true,
            "timeout_seconds": 30,
            "retry_count": 3
        },
        ...
    ]
```

```
GET http://localhost:8000/api/v1/acquisition/websdrs/health
Status: 200 OK
Response type: JSON
Response preview:
    {
        "1": {
            "websdr_id": 1,
            "name": "Aquila di Giaveno",
            "status": "online",
            "last_check": "2025-10-22T18:30:45.123456Z"
        },
        "2": {
            "websdr_id": 2,
            "name": "Montanaro",
            "status": "online",
            "last_check": "2025-10-22T18:30:45.456789Z"
        },
        ...
    }
```

### ❌ Cosa NON dovresti vedere:

- ❌ Status `"OK"` come risposta (era la risposta mockata)
- ❌ Status `404 Not Found`
- ❌ CORS errors
- ❌ `http://localhost:8000/api/api/v1/...` (doppio `/api`)

---

## 🎯 Test 2: Verificare i Dati nella Pagina

### Procedura:

1. Verifica che la tabella **"Receiver Configuration"** appaia con dati

### Cosa dovresti vedere:

| Receiver Name     | Location         | GPS Coordinates | Status   |
| ----------------- | ---------------- | --------------- | -------- |
| Aquila di Giaveno | Giaveno, Italy   | 45.0200, 7.2900 | 🟢 Online |
| Montanaro         | Montanaro, Italy | 45.2340, 7.8570 | 🟢 Online |
| Torino            | Torino, Italy    | 45.0440, 7.6720 | 🟢 Online |
| ...               | ...              | ...             | ...      |

### Card con Summary (in alto):

```
┌─────────────────────────────────────────────────┐
│ Online Receivers: 5/7    │ Average Uptime: 99.5%│
└─────────────────────────────────────────────────┘
```

---

## ⏱️ Test 3: Verificare l'Auto-Refresh

### Procedura:

1. Apri **Console** (tab Console nel Developer Tools)

2. Verifica che ogni **30 secondi** vedrai nuove righe di log:
   ```
   GET http://localhost:8000/api/v1/acquisition/websdrs
   GET http://localhost:8000/api/v1/acquisition/websdrs/health
   ```

3. Attendi almeno 60 secondi

### Cosa dovresti vedere:

✅ Ogni 30 secondi:
- Nuova richiesta GET ai due endpoint
- Status sempre `200 OK`
- `last_check` timestamp aggiornato

---

## 🚨 Test 4: Verificare la Gestione Errori

### Procedura A: Simulare errore di backend

1. **Ferma il backend**:
   ```bash
   docker-compose down
   ```

2. **Aggiorna la pagina** nel browser

### Cosa dovresti vedere:

✅ **Messaggio di errore in rosso** (in alto nella pagina):

```
⚠️ Errore caricamento WebSDRs
Failed to connect to backend service
```

### Procedura B: Riavvia il backend

1. **Riavvia il backend**:
   ```bash
   docker-compose up -d
   ```

2. **Aggiorna la pagina** nel browser

3. Entro 3-5 secondi dovrebbe ricaricare correttamente

---

## 📊 Test 5: Verificare il Caricamento

### Procedura:

1. Con il Network Throttling attivato (simula connessione lenta):
   - Chrome DevTools → Network tab → Throttling: "Slow 3G"

2. **Aggiorna la pagina**

### Cosa dovresti vedere:

✅ **Spinner animato** mentre carica:
```
🔄 Caricamento WebSDRs...
```

✅ Dopo 2-5 secondi (simulato throttling): Dati caricati

---

## 🔧 Debug Avanzato

Se riscontri problemi, esegui questi comandi:

### Verifica che l'API risponda:

```bash
# Test endpoint WebSDRs
curl http://localhost:8000/api/v1/acquisition/websdrs | jq

# Test endpoint health
curl http://localhost:8000/api/v1/acquisition/websdrs/health | jq
```

### Verifica che il frontend veda il backend:

Nel browser Console:
```javascript
// Apri Console (F12 → Console)
fetch('http://localhost:8000/api/v1/acquisition/websdrs')
    .then(r => r.json())
    .then(d => console.log(d))
    .catch(e => console.error(e))
```

Dovresti vedere l'array di WebSDRs nel console.

### Verifica il valore di VITE_API_URL:

Nel browser Console:
```javascript
console.log(import.meta.env.VITE_API_URL)
// Deve stampare: http://localhost:8000
// NON: http://localhost:8000/api
```

---

## ✨ Test Superato Se...

- ✅ Vedi **7 ricevitori italiani** nella tabella
- ✅ Tutti gli endpoint rispondono con **status 200**
- ✅ Nessun errore CORS
- ✅ I dati sono **JSON reali**, non stringhe "OK"
- ✅ Auto-refresh funziona ogni 30 secondi
- ✅ Gestione errori funziona (se backend cade)
- ✅ Nessun errore nella Console del browser

---

## 📋 Rapporto di Test

Quando il test è completato, fornisci un rapporto simile:

```
✅ Test 1 - Network API: PASS
   - Request 1: 200 OK (websdrs)
   - Request 2: 200 OK (websdrs/health)
   - No CORS errors
   
✅ Test 2 - Dati in Pagina: PASS
   - 7 ricevitori visualizzati
   - Campi corretti: name, location, coordinates, status
   
✅ Test 3 - Auto-Refresh: PASS
   - Ogni 30 secondi le richieste si ripetono
   - Timestamp aggiornato
   
✅ Test 4 - Gestione Errori: PASS
   - Messaggio di errore visibile quando backend è down
   - Ripristino automatico quando backend è up
   
✅ Test 5 - Caricamento: PASS
   - Spinner animato durante il caricamento
   - Dati caricati dopo ~2 secondi
   
RISULTATO FINALE: ✅ ALL TESTS PASSED
```

---

**Autore**: GitHub Copilot  
**Data**: 22 Ottobre 2025  
**Versione**: 1.0
