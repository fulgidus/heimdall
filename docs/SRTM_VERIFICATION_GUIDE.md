# Verifica Integrazione SRTM nel Sistema di Generazione Sintetica

**Data**: 2025-11-02  
**Scopo**: Guida per verificare che il sistema di generazione dati sintetici utilizzi effettivamente i dati SRTM per calcolare il deterioramento del segnale e le ostruzioni line-of-sight.

---

## 🎯 Obiettivo

Verificare che il **generatore di dati sintetici** utilizzi i **dati terreno SRTM** per:
1. **Calcolare il deterioramento del segnale** in base alle ostruzioni geografiche
2. **Effettuare controlli line-of-sight (LOS)** con analisi della zona di Fresnel
3. **Applicare correzioni per la curvatura terrestre** nei calcoli di propagazione RF

---

## 🔍 Metodi di Verifica

### **Metodo 1: Script di Verifica Automatico** ⭐ (Raccomandato)

Esegui lo script dedicato che effettua 4 test automatici:

```bash
# All'interno del container training
docker compose exec training python /app/scripts/verify_srtm_integration.py
```

Lo script verifica:
1. ✅ **Disponibilità tiles SRTM**: Controlla che i tiles siano stati scaricati
2. ✅ **Funzionalità TerrainLookup**: Testa le query di elevazione
3. ✅ **Propagazione con terreno**: Confronta calcoli con/senza SRTM
4. ✅ **Configurazione generatore**: Verifica parametri `use_srtm_terrain`

**Output Atteso**:
```
================================================================
SUMMARY
================================================================
✅ PASSED: Terrain Tile Availability
✅ PASSED: Terrain Lookup Functionality
✅ PASSED: RF Propagation with Terrain
✅ PASSED: Synthetic Generation Config

Total: 4/4 tests passed

🎉 All tests passed! SRTM integration is working correctly.
```

---

### **Metodo 2: Confronto Manuale - Prima/Dopo**

Genera due dataset sintetici identici, uno con e uno senza SRTM, e confronta i risultati.

#### **Passo 1: Genera Dataset SENZA Terreno (Baseline)**

```bash
curl -X POST http://localhost:8001/api/v1/training/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "job_name": "baseline_no_terrain",
    "job_type": "synthetic_data",
    "config": {
      "name": "Dataset Baseline (No Terrain)",
      "num_samples": 100,
      "frequency_mhz": 145.0,
      "tx_power_dbm": 37.0,
      "min_snr_db": 3.0,
      "use_srtm_terrain": false,
      "seed": 42
    }
  }'
```

#### **Passo 2: Genera Dataset CON Terreno (SRTM)**

```bash
curl -X POST http://localhost:8001/api/v1/training/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "job_name": "srtm_enabled",
    "job_type": "synthetic_data",
    "config": {
      "name": "Dataset SRTM (With Terrain)",
      "num_samples": 100,
      "frequency_mhz": 145.0,
      "tx_power_dbm": 37.0,
      "min_snr_db": 3.0,
      "use_srtm_terrain": true,
      "seed": 42
    }
  }'
```

#### **Passo 3: Confronta i Risultati**

```sql
-- Confronto SNR medio per ricevitore
SELECT 
    d1.name as baseline_dataset,
    d2.name as srtm_dataset,
    AVG(CAST(s1.receivers->>'0'->>'snr_db' AS FLOAT)) as baseline_avg_snr,
    AVG(CAST(s2.receivers->>'0'->>'snr_db' AS FLOAT)) as srtm_avg_snr,
    AVG(CAST(s2.receivers->>'0'->>'snr_db' AS FLOAT)) - 
    AVG(CAST(s1.receivers->>'0'->>'snr_db' AS FLOAT)) as snr_difference
FROM heimdall.synthetic_datasets d1
JOIN heimdall.synthetic_training_samples s1 ON s1.dataset_id = d1.id
JOIN heimdall.synthetic_datasets d2 ON d2.name LIKE '%SRTM%'
JOIN heimdall.synthetic_training_samples s2 ON s2.dataset_id = d2.id
WHERE d1.name LIKE '%Baseline%'
AND s1.id = s2.id  -- Stesso seed, stesso sample
GROUP BY d1.name, d2.name;
```

**Risultati Attesi**:
- SNR più basso con SRTM (maggiore deterioramento realistico)
- Maggiore varianza nei valori SNR con SRTM
- Alcuni ricevitori potrebbero avere `signal_present=false` con SRTM ma `true` senza

---

### **Metodo 3: Ispezione Visiva dei Log**

Durante la generazione sintetica con SRTM abilitato, cerca nei log:

```bash
docker compose logs training | grep -i "terrain\|srtm\|los"
```

**Log Attesi**:
```
INFO: Using SRTM terrain data for synthetic generation
INFO: Terrain lookup initialized with MinIO bucket: heimdall-terrain
DEBUG: Loaded tile N44E007 from MinIO cache
DEBUG: LOS check: TX (45.0, 7.5) -> RX (45.5, 9.0): BLOCKED by terrain (intrusion: 78%)
DEBUG: Terrain loss calculated: 32.5 dB
```

**Log Problematici** (indicano che SRTM NON è usato):
```
WARNING: Failed to initialize SRTM terrain, using simplified model
DEBUG: Using simplified terrain model (altitude-based)
```

---

### **Metodo 4: Query Diretta su un Sample**

Estrai un singolo sample e analizza i metadati della propagazione:

```sql
SELECT 
    s.tx_lat, 
    s.tx_lon,
    s.receivers->'Torino'->>'distance_km' as distance_km,
    s.receivers->'Torino'->>'snr_db' as snr_db,
    s.receivers->'Torino'->>'signal_present' as signal_present,
    d.config->>'use_srtm_terrain' as srtm_enabled
FROM heimdall.synthetic_training_samples s
JOIN heimdall.synthetic_datasets d ON d.id = s.dataset_id
WHERE d.name LIKE '%SRTM%'
LIMIT 5;
```

Poi usa l'API per calcolare manualmente la propagazione:

```bash
# Con terreno
curl "http://localhost:8001/api/v1/terrain/propagation?tx_lat=45.0&tx_lon=7.5&tx_alt=300&rx_lat=45.5&rx_lon=9.0&rx_alt=200&freq_mhz=145"

# Senza terreno (rimuovi terrain tiles temporaneamente)
```

---

## 📊 Metriche di Successo

| Metrica | Con SRTM | Senza SRTM |
|---------|----------|------------|
| **SNR medio** | 8-15 dB | 12-18 dB |
| **Terrain loss** | 10-40 dB | 5-15 dB (semplificato) |
| **% samples con signal_present=false** | 15-25% | 5-10% |
| **Varianza SNR** | Alta (realistico) | Bassa (uniforme) |

---

## 🐛 Troubleshooting

### Problema: "SRTM data not being used"

**Sintomi**:
- Terrain loss identico con/senza SRTM
- Log dice "using simplified model"
- Nessun accesso a MinIO nei log

**Soluzioni**:
1. Verifica che i tiles siano scaricati:
   ```bash
   curl http://localhost:8001/api/v1/terrain/tiles
   ```
2. Controlla connessione MinIO:
   ```bash
   docker compose logs minio
   docker compose exec training curl http://minio:9000/minio/health/live
   ```
3. Verifica parametro `use_srtm_terrain=true` nel config del job

### Problema: "Tiles not found"

**Sintomi**:
- Log dice "Tile N44E007 not found in MinIO"
- Fallback al modello semplificato

**Soluzioni**:
1. Scarica i tiles mancanti:
   - Via UI: Vai su "Terrain Management" → "Download WebSDR Region Tiles"
   - Via API: `POST /api/v1/terrain/download`
2. Verifica copertura tiles per la tua area:
   ```bash
   curl http://localhost:8001/api/v1/terrain/coverage
   ```

### Problema: "Terrain loss sempre 0 dB"

**Sintomi**:
- Terrain loss è 0 anche con SRTM
- Tutti i path hanno LOS chiaro

**Possibili Cause**:
- TX/RX molto vicini (< 5 km)
- TX/RX molto alti (> 1000m ASL)
- Area pianeggiante (pianura padana)

**Soluzione**: Usa TX/RX più distanti o in aree montuose per test

---

## 🔬 Test Avanzato: Validazione con Dati Reali

Per la massima sicurezza, confronta con misure reali:

1. **Prendi una misura reale** da `recording_sessions`
2. **Genera sample sintetico** con stesse coordinate TX/RX
3. **Confronta SNR reale vs sintetico**:
   - Dovrebbero essere simili (±5 dB) con SRTM
   - Differenze maggiori senza SRTM

```sql
-- Confronto reale vs sintetico
SELECT 
    'Real' as source,
    AVG(m.snr_db) as avg_snr
FROM heimdall.measurements m
WHERE m.recording_session_id = '<session_id>'

UNION ALL

SELECT 
    'Synthetic (SRTM)' as source,
    AVG(CAST(s.receivers->'Torino'->>'snr_db' AS FLOAT)) as avg_snr
FROM heimdall.synthetic_training_samples s
WHERE s.dataset_id = '<srtm_dataset_id>'
AND s.tx_lat BETWEEN 44.9 AND 45.1
AND s.tx_lon BETWEEN 7.4 AND 7.6;
```

---

## ✅ Checklist Finale

Prima di considerare SRTM completamente integrato:

- [ ] Script di verifica passa tutti i 4 test
- [ ] Dataset con SRTM ha SNR più basso del baseline
- [ ] Log mostra "Using SRTM terrain data"
- [ ] Terrain loss varia tra 0-40 dB (non costante)
- [ ] Alcuni samples hanno `signal_present=false` per ricevitori lontani
- [ ] Query di elevazione via API restituiscono valori corretti
- [ ] Map UI mostra tiles verdi (ready)
- [ ] MinIO contiene files `.tif` in bucket `heimdall-terrain`

---

## 📚 Riferimenti

- **Codice sorgente**:
  - `services/training/src/data/propagation.py` → Modello RF con SRTM
  - `services/common/terrain/terrain.py` → TerrainLookup e SRTMDownloader
  - `services/training/src/data/synthetic_generator.py` → Generatore sintetico

- **Documentazione**:
  - `SRTM_IMPLEMENTATION.md` → Dettagli implementazione completa
  - `docs/ARCHITECTURE.md` → Architettura sistema
  
- **API Endpoints**:
  - `GET /api/v1/terrain/tiles` → Lista tiles
  - `GET /api/v1/terrain/coverage` → Coverage status
  - `GET /api/v1/terrain/elevation?lat=X&lon=Y` → Query elevazione

---

**Autore**: Heimdall Development Team  
**Ultima Modifica**: 2025-11-02
