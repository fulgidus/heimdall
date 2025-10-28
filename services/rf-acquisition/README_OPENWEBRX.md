# OpenWebRX Integration - Implementation Summary

## ✅ What Was Implemented

### 1. OpenWebRX WebSocket Client (`src/fetchers/openwebrx_client.py`)

**Classe principale:** `OpenWebRXClient`

**Features:**
- ✅ WebSocket connection to single OpenWebRX receiver
- ✅ Handshake protocol: `"SERVER DE CLIENT client=openwebrx.js type=receiver"`
- ✅ Multiplexed stream handling (FFT + Audio + Control on single WebSocket)
- ✅ Binary frame parsing:
  - `0x01` → FFT/Waterfall frames (2054 bytes, ~7 fps)
  - `0x02` → ADPCM audio frames (variable size, ~20 fps)
- ✅ JSON control message parsing
- ✅ Async context manager support (`async with`)
- ✅ Callbacks for FFT, Audio, and Control messages
- ✅ Statistics tracking

**Dataclasses:**
- `FFTFrame`: Represents spectrum data with `to_spectrum()` method for freq→power conversion
- `AudioFrame`: Represents ADPCM audio with `decompress()` method for PCM extraction

**Example usage:**
```python
from src.fetchers.openwebrx_client import OpenWebRXClient, FFTFrame, AudioFrame

client = OpenWebRXClient("http://sdr1.ik1jns.it:8076")

def on_fft(frame: FFTFrame):
    freqs, power = frame.to_spectrum()
    print(f"FFT: {len(freqs)} bins, peak={power.max():.1f} dBm")

client.on_fft = on_fft

async with client:
    await client.receive_loop(duration=10)

print(f"Stats: {client.stats}")
```

### 2. Celery Tasks (`src/tasks/acquire_openwebrx.py`)

**Tasks implementati:**

#### `acquire_openwebrx_single(websdr_url, duration_seconds, save_fft, save_audio)`
Acquisisce da un singolo WebSDR per N secondi.

**Parameters:**
- `websdr_url`: URL base (es. `"http://sdr1.ik1jns.it:8076"`)
- `duration_seconds`: Durata acquisizione (default 60s)
- `save_fft`: Salva frame FFT a database (default True)
- `save_audio`: Salva frame audio a database (default False, troppo storage!)

**Returns:**
```python
{
    "websdr_url": "http://sdr1.ik1jns.it:8076",
    "duration": 60,
    "fft_frames": 438,  # ~7 fps × 60s
    "audio_frames": 1236,  # ~20 fps × 60s
    "text_messages": 245,
    "errors": 0,
    "fft_saved": 438,
    "success": True
}
```

#### `acquire_openwebrx_all(duration_seconds, save_fft, save_audio)`
Acquisisce da TUTTI i 7 WebSDR simultaneamente usando `asyncio.gather()`.

**Returns:**
```python
{
    "duration": 300,
    "num_receivers": 7,
    "total_fft_frames": 15330,  # 7 × 7fps × 300s
    "total_audio_frames": 42000,  # 7 × 20fps × 300s
    "errors": 0,
    "success": True,
    "receivers": {
        "http://sdr1.ik1jns.it:8076": {...},
        "http://sdr3.ik1jns.it:8073": {...},
        ...
    }
}
```

#### `health_check_openwebrx()`
Verifica connettività a tutti i 7 WebSDR.

**Returns:**
```python
{
    "http://sdr1.ik1jns.it:8076": True,
    "http://sdr3.ik1jns.it:8073": True,
    "http://iz1pnv.duckdns.org:8074": False,  # Offline
    ...
}
```

---

## 🚀 How to Use

### Test del Client (Standalone)

```bash
# Activate venv
cd /home/fulgidus/Documents/heimdall/services/rf-acquisition
source ../../.venv/bin/activate

# Run client test (has __main__ block)
python src/fetchers/openwebrx_client.py
```

Output atteso:
```
2025-10-28 23:00:00 - INFO - Connecting to ws://sdr1.ik1jns.it:8076/ws/...
2025-10-28 23:00:00 - INFO - ✅ WebSocket connected to sdr1.ik1jns.it:8076
2025-10-28 23:00:01 - INFO - ✅ Handshake complete - center_freq=145575000 Hz, bw=2400000 Hz
FFT #1: 2049 bins, peak=-25.3 dBm
Audio #1: 294 bytes ADPCM
FFT #11: 2049 bins, peak=-28.1 dBm
Audio #51: 2054 bytes ADPCM
...
2025-10-28 23:00:10 - INFO - ✅ Disconnected from sdr1.ik1jns.it:8076

Final: 73 FFT frames, 206 Audio frames
```

### Test Celery Task

```bash
# In terminal 1: Start Celery worker
celery -A src.main worker --loglevel=info

# In terminal 2: Trigger task
python -c "
from src.tasks import acquire_openwebrx_single

result = acquire_openwebrx_single.delay(
    websdr_url='http://sdr1.ik1jns.it:8076',
    duration_seconds=10,
    save_fft=False,  # Don't save to DB yet (models not implemented)
    save_audio=False
)

print('Task ID:', result.id)
print('Result:', result.get(timeout=30))
"
```

### Acquisizione Multi-SDR

```python
from src.tasks import acquire_openwebrx_all

# Acquire from all 7 WebSDRs for 5 minutes
result = acquire_openwebrx_all.delay(
    duration_seconds=300,
    save_fft=True,
    save_audio=False  # Audio uses ~1 GB/hour, disable by default
)

stats = result.get(timeout=360)
print(f"Total FFT frames: {stats['total_fft_frames']}")
print(f"Online receivers: {stats['num_receivers'] - stats['errors']}/{stats['num_receivers']}")
```

---

## 📊 Performance Characteristics

### Bandwidth per Receiver
- **FFT:** 2054 bytes/frame × 7 fps = ~14 KB/sec
- **Audio:** 2000 bytes/frame × 20 fps = ~40 KB/sec
- **Total:** ~54 KB/sec = **432 Kbps**

### Storage per Receiver (if saving to DB)
- **FFT only:** 14 KB/sec × 3600 = **50 MB/hour**
- **FFT + Audio:** 54 KB/sec × 3600 = **194 MB/hour**

### Multi-SDR (7 receivers)
- **Bandwidth:** 432 Kbps × 7 = **~3 Mbps**
- **Storage (FFT only):** 50 MB/hour × 7 = **350 MB/hour** = 8.4 GB/day
- **Storage (FFT + Audio):** 194 MB/hour × 7 = **1.36 GB/hour** = 32.6 GB/day

**Raccomandazione:** Salva solo FFT, non Audio (troppo storage).

---

## ⚠️ TODO - Database Models

I task salvano i frame usando placeholder `_save_fft_frame()` e `_save_audio_frame()`.

**Da implementare:**

### 1. Model SQLAlchemy per FFT

```python
# src/models/acquisitions.py

from sqlalchemy import Column, Integer, BigInteger, LargeBinary, Float, DateTime, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class FFTCapture(Base):
    __tablename__ = 'fft_captures'
    
    id = Column(Integer, primary_key=True)
    websdr_id = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    center_freq = Column(BigInteger, nullable=False)  # Hz
    bandwidth = Column(BigInteger, nullable=False)  # Hz
    num_bins = Column(Integer, nullable=False)
    spectral_data = Column(LargeBinary, nullable=False)  # uint8 array
    waterfall_min = Column(Float)  # dBm
    waterfall_max = Column(Float)  # dBm
```

### 2. Implementa Save Functions

```python
# In src/tasks/acquire_openwebrx.py

def _save_fft_frame(frame: FFTFrame):
    """Save FFT frame to TimescaleDB."""
    from ..models.acquisitions import FFTCapture
    from ..storage.timescaledb import get_session
    
    session = get_session()
    
    try:
        capture = FFTCapture(
            websdr_id=frame.websdr_id,
            timestamp=frame.timestamp,
            center_freq=frame.center_freq,
            bandwidth=frame.bandwidth,
            num_bins=len(frame.bins),
            spectral_data=frame.bins
        )
        
        session.add(capture)
        session.commit()
        
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()
```

### 3. Database Migration

```bash
# Create migration
alembic revision --autogenerate -m "Add OpenWebRX FFT captures table"

# Apply migration
alembic upgrade head
```

---

## 🎯 Next Steps

### Phase 1: Database Integration (1-2 giorni)
- [ ] Crea model `FFTCapture` in SQLAlchemy
- [ ] Crea model `AudioCapture` (opzionale)
- [ ] Implementa `_save_fft_frame()` con insert reale
- [ ] Test acquisizione + salvataggio

### Phase 2: Feature Extraction (1 settimana)
- [ ] Crea `FFTDecoder.find_peaks()` per rilevare segnali
- [ ] Implementa burst detection (attività radio sopra threshold)
- [ ] Estrai features spettrali per AI (bandwidth, potenza, modulazione)
- [ ] Integra con pipeline ML esistente

### Phase 3: TDOA Geolocation (2 settimane)
- [ ] Sincronizza timestamp tra 7 ricevitori
- [ ] Implementa cross-correlation tra audio streams
- [ ] Calcola TDOA deltas (differenze temporali)
- [ ] Algoritmo hyperbolic positioning → lat/lon
- [ ] Validazione accuratezza (±500m-2km)

### Phase 4: Scheduled Acquisition (2 giorni)
- [ ] Aggiungi Celery Beat schedule per acquisizione continua
- [ ] Task ogni 5 minuti: `acquire_openwebrx_all.delay(300)`
- [ ] Retention policy: DELETE dati >30 giorni
- [ ] Monitoring: Grafana dashboard per FFT rate, errori, storage

---

## 📚 References

- **Main Documentation:** `docs/WEBSDR_INTEGRATION_GUIDE.md` (unified guide)
- **Test Script:** `scripts/test_openwebrx_multiplexed.py` (standalone test)
- **WebSDR List:** `WEBSDRS.md` (7 receivers in Piemonte)

---

## ✅ Summary

**Implementato:**
- ✅ Client WebSocket OpenWebRX completo e funzionante
- ✅ 3 Celery tasks (single, all, health_check)
- ✅ Parsing FFT e Audio frames
- ✅ Statistiche e logging
- ✅ Async multi-receiver orchestration

**Da fare:**
- ⚠️ Database models e save functions
- ⚠️ Feature extraction per AI
- ⚠️ TDOA geolocation engine
- ⚠️ Celery Beat scheduling

**Status:** **PRONTO PER TESTING E INTEGRAZIONE DB** 🚀
