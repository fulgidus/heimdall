# OpenWebRX+ WebSocket Protocol Documentation

**Date**: 2025-10-29  
**Status**: ✅ WORKING  
**Last Tested**: heimdall rf-acquisition service v1.0

## Overview

OpenWebRX+ uses a WebSocket-based protocol for streaming IQ/audio data to clients. This document describes the protocol implementation used in the Heimdall project for RF signal acquisition.

## Critical Discovery

**IMPORTANT**: The WebSocket server does **NOT** send any data until it receives a proper handshake message from the client first.

### Handshake Protocol

#### 1. Client Connects
```python
ws_url = "ws://<hostname>:<port>/ws/"
async with websockets.connect(ws_url, max_size=10_000_000) as ws:
```

#### 2. Client MUST Send Handshake First
```python
handshake_msg = "SERVER DE CLIENT client=heimdall type=receiver"
await ws.send(handshake_msg)
```

**Without this handshake, the server will accept the WebSocket connection but will never send any data!**

#### 3. Server Responds
After receiving the handshake, the server sends:

1. **Server acknowledgment**:
   ```
   CLIENT DE SERVER server=openwebrx version=v1.2.94
   ```

2. **Receiver details** (JSON):
   ```json
   {
     "type": "receiver_details",
     "value": {
       "receiver_gps": {"lat": 45.03, "lon": 7.27},
       "receiver_asl": 1340,
       ...
     }
   }
   ```

3. **Configuration messages** (JSON):
   - `{"type": "config", "value": {...}}`
   - `{"type": "modes", "value": [...]}`
   - `{"type": "profiles", "value": [...]}`
   - `{"type": "bands", "value": [...]}`
   - `{"type": "bookmarks", "value": [...]}`

## IQ Data Streaming

### Setting Frequency and Mode

```python
# Set frequency (in Hz) and mode to IQ
await ws.send(f"SET mod=iq freq={frequency_hz}")

# Start streaming
await ws.send("START")
```

### Receiving IQ Data

After sending `START`, the server streams binary IQ data frames:

```python
while True:
    message = await ws.recv()
    
    if isinstance(message, bytes):
        # Binary IQ data
        iq_samples = parse_iq_data(message)
    elif isinstance(message, str):
        # JSON status/config updates (ignore during streaming)
        pass
```

### IQ Data Format

OpenWebRX sends audio-format IQ data as **16-bit PCM interleaved I/Q samples**:

```python
def parse_iq_data(data: bytes) -> np.ndarray:
    """
    Parse OpenWebRX IQ data.
    
    Format: int16 PCM, alternating I and Q samples
    """
    # Unpack as signed 16-bit integers
    num_samples = len(data) // 2
    samples = struct.unpack(f'<{num_samples}h', data[:num_samples*2])
    
    # Normalize to [-1, 1] range
    normalized = np.array(samples, dtype=np.float32) / 32768.0
    
    # De-interleave to complex
    i_samples = normalized[0::2]
    q_samples = normalized[1::2]
    
    iq_complex = i_samples + 1j * q_samples
    
    return iq_complex.astype(np.complex64)
```

### Stopping Stream

```python
await ws.send("STOP")
```

## Complete Working Example

```python
import asyncio
import websockets
import numpy as np
import struct

async def acquire_iq_data(url: str, frequency_hz: int, duration_seconds: float):
    """Acquire IQ data from OpenWebRX."""
    
    async with websockets.connect(url, max_size=10_000_000) as ws:
        # 1. CRITICAL: Send handshake first!
        await ws.send("SERVER DE CLIENT client=heimdall type=receiver")
        
        # 2. Wait for server handshake and config
        for _ in range(10):
            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            if isinstance(msg, str) and "CLIENT DE SERVER" in msg:
                break
        
        # 3. Set frequency and mode
        await ws.send(f"SET mod=iq freq={frequency_hz}")
        await asyncio.sleep(0.5)
        
        # 4. Start streaming
        await ws.send("START")
        
        # 5. Collect IQ data
        iq_samples = []
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < duration_seconds:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                
                if isinstance(message, bytes):
                    # Parse IQ data
                    samples = parse_iq_data(message)
                    if len(samples) > 0:
                        iq_samples.append(samples)
                        
            except asyncio.TimeoutError:
                break
        
        # 6. Stop streaming
        await ws.send("STOP")
        
        # 7. Return concatenated samples
        if iq_samples:
            return np.concatenate(iq_samples)
        else:
            raise RuntimeError("No IQ data received")

def parse_iq_data(data: bytes) -> np.ndarray:
    """Parse OpenWebRX binary IQ data."""
    if len(data) < 4:
        return np.array([], dtype=np.complex64)
    
    # Unpack 16-bit PCM
    num_samples = len(data) // 2
    samples = struct.unpack(f'<{num_samples}h', data[:num_samples*2])
    
    # Normalize
    normalized = np.array(samples, dtype=np.float32) / 32768.0
    
    # De-interleave I/Q
    i_samples = normalized[0::2]
    q_samples = normalized[1::2]
    
    min_len = min(len(i_samples), len(q_samples))
    if min_len == 0:
        return np.array([], dtype=np.complex64)
    
    return (i_samples[:min_len] + 1j * q_samples[:min_len]).astype(np.complex64)
```

## Sample Rate

OpenWebRX typically streams at **12 kHz audio sample rate** for IQ mode. The actual sample rate can be extracted from the `config` JSON message:

```json
{
  "type": "config",
  "value": {
    "samp_rate": 2400000,  // SDR sample rate
    "center_freq": 145000000,
    ...
  }
}
```

However, for audio-based IQ streaming, the effective sample rate is ~12 kHz.

## Tested WebSDR Stations

All 7 Italian WebSDR stations tested successfully:

| Name              | URL                            | Status |
| ----------------- | ------------------------------ | ------ |
| Aquila di Giaveno | http://sdr1.ik1jns.it:8076     | ✅      |
| Montanaro         | http://cbfenis.ddns.net:43510  | ✅      |
| Torino            | http://vst-aero.it:8073        | ✅      |
| Coazze            | http://94.247.189.130:8076     | ✅      |
| Passo del Giovi   | http://iz1mlt.ddns.net:8074    | ✅      |
| Genova            | http://iq1zw.ddns.net:42154    | ✅      |
| Milano - Baggio   | http://iu2mch.duckdns.org:8073 | ✅      |

## Troubleshooting

### Problem: WebSocket connects but no data received

**Solution**: Make sure you're sending the handshake message first!
```python
await ws.send("SERVER DE CLIENT client=heimdall type=receiver")
```

### Problem: Complex numbers not JSON serializable

If you need to return IQ data in JSON (e.g., from Celery tasks), either:
1. Remove `iq_data` from the result dict (store only in MinIO/S3)
2. Convert to list of `[real, imag]` pairs:
   ```python
   iq_serializable = [[np.real(x), np.imag(x)] for x in iq_data]
   ```

### Problem: Connection timeout

- Check that the WebSDR station is online
- Verify network connectivity
- Increase connection timeout (default 30s)

## References

- OpenWebRX+ GitHub: https://github.com/luarvique/openwebrx
- OpenWebRX original: https://github.com/jketterl/openwebrx
- JavaScript client code: `/compiled/receiver.js` on any OpenWebRX instance

## Implementation in Heimdall

See: `services/rf-acquisition/src/fetchers/openwebrx_fetcher.py`

Key classes:
- `OpenWebRXClient`: WebSocket client for single receiver
- `OpenWebRXFetcher`: Parallel fetcher for multiple receivers

---

**Note**: This documentation was created after extensive reverse engineering and testing on 2025-10-29. The protocol is not officially documented by OpenWebRX, so this represents empirical findings from successful implementation.
