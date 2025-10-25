# WebSDR Receivers Configuration

This document contains the configuration and details for the 7 WebSDR receivers used in the Heimdall localization network.

## Network Overview

The Heimdall system uses a distributed network of WebSDR receivers strategically positioned across Northwestern Italy (Piedmont and Liguria regions) to enable triangulation of radio sources on the 2m and 70cm amateur bands.

**Target Frequencies:**
- 2m band: 144-146 MHz
- 70cm band: 430-440 MHz

**Network Focus Area:**
- Primary region: Piedmont (Piemonte) - Northern Italy
- Network density: 7 stations covering ~8,000 km² area
- Optimal for tracking amateur radio activity in the region

## WebSDR Stations

### 1. Aquila di Giaveno
- **URL**: `http://sdr1.ik1jns.it:8076/`
- **Location**: Giaveno, Italy (45.02°N, 7.29°E)
- **Altitude**: 600m ASL
- **Frequency Range**: 144-146 MHz, 430-440 MHz
- **Status**: Active
- **Notes**: Primary station for Northern Italy coverage

### 2. Montanaro
- **URL**: `http://cbfenis.ddns.net:43510/`
- **Location**: Montanaro, Italy (45.234°N, 7.857°E)
- **Altitude**: ~300m ASL
- **Frequency Range**: 144-146 MHz, 430-440 MHz
- **Status**: Active
- **Notes**: Coverage for Monferrato region

### 3. Torino
- **URL**: `http://vst-aero.it:8073/`
- **Location**: Torino, Italy (45.044°N, 7.672°E)
- **Altitude**: ~250m ASL
- **Frequency Range**: 144-146 MHz, 430-440 MHz
- **Status**: Active
- **Notes**: Aeronautical society station, Turin area coverage

### 4. Coazze
- **URL**: `http://94.247.189.130:8076/`
- **Location**: Coazze, Italy (45.03°N, 7.27°E)
- **Altitude**: ~700m ASL
- **Frequency Range**: 144-146 MHz, 430-440 MHz
- **Status**: Active
- **Notes**: High-altitude station in Torino valleys

### 5. Passo del Giovi
- **URL**: `http://iz1mlt.ddns.net:8074/`
- **Location**: Passo del Giovi, Italy (44.561°N, 8.956°E)
- **Altitude**: ~480m ASL
- **Frequency Range**: 144-146 MHz, 430-440 MHz
- **Status**: Active
- **Notes**: Mountain pass station, Genova connection

### 6. Genova
- **URL**: `http://iq1zw.ddns.net:42154/`
- **Location**: Genova, Italy (44.395°N, 8.956°E)
- **Altitude**: ~100m ASL
- **Frequency Range**: 144-146 MHz, 430-440 MHz
- **Status**: Active
- **Notes**: Coastal station, Liguria region coverage

### 7. Milano - Baggio
- **URL**: `http://iu2mch.duckdns.org:8073/`
- **Location**: Milano (Baggio), Italy (45.478°N, 9.123°E)
- **Altitude**: ~120m ASL
- **Frequency Range**: 144-146 MHz, 430-440 MHz
- **Status**: Active
- **Notes**: Metropolitan area coverage, Lombardy region

## Coverage Analysis

### Geographic Distribution
The network provides excellent coverage across Northwestern Italy with stations strategically distributed to optimize triangulation geometry:

- **North-South span**: ~90 km (Milano to Genova)
- **East-West span**: ~140 km (Milano to Genova coast)
- **Optimal baseline**: 50-150 km between stations for triangulation
- **Altitude diversity**: Sea level to 700m ASL
- **Strategic positioning**: Mountain passes and valley coverage for optimal signal reception

### Network Geometry
The 7-station network creates overlapping coverage zones across Northwestern Italy:

**Key Network Characteristics:**
- **Triangular core**: Giaveno ↔ Torino ↔ Montanaro (optimal for localization)
- **Eastern extension**: Genova at south + Passo del Giovi at mountain pass
- **Northern reach**: Milano (Baggio) for Lombardy coverage
- **Altitude profile**: Mixed valley, mountain, and urban locations

### Expected Accuracy
Based on the network geometry and typical VHF/UHF propagation:

- **Best accuracy zone**: Central Piedmont region (Giaveno-Torino-Montanaro triangle)
- **Expected precision**: ±20-50m in optimal conditions (strong signals, good geometry)
- **Coverage range**: ~150-200 km radius from network center
- **Minimum detectable signal**: -110 dBm (typical WebSDR sensitivity)
- **Maximum baseline**: ~200 km (Milano to Genova)
- **Minimum baseline**: ~30 km (Coazze to Giaveno)

## WebSDR API Integration

### Connection Parameters
```python
WEBSDR_CONFIG = {
    'timeout': 10,          # Connection timeout (seconds)
    'retry_count': 3,       # Number of retries on failure
    'sample_rate': 12000,   # IQ sample rate (Hz)
    'bandwidth': 2400,      # Reception bandwidth (Hz)
    'format': 'iq'          # Data format (iq/audio)
}
```

### Known Limitations
- **Rate limiting**: Some stations limit concurrent connections
- **Availability**: 90-95% uptime typical for amateur stations
- **Data format**: Not all stations support IQ output
- **Frequency accuracy**: ±100 Hz typical for amateur equipment
- **Time synchronization**: No GPS disciplining on some stations

## Monitoring and Health Checks

### Automated Monitoring
The system performs health checks every 60 seconds:
- Connection availability
- Signal level measurements
- Frequency stability checks
- Data quality validation

### Backup Procedures
- Alternative frequencies when primary is occupied
- Graceful degradation with fewer stations
- Manual station enable/disable capability
- Automatic failover to backup stations

## Contact Information

For issues with specific WebSDR stations in the Northwestern Italy network:
- **IK1JNS (Giaveno)**: Available via QRZ.com
- **IZ1MUX (Montanaro)**: Contact via CBFENIS network
- **VST-AERO (Torino)**: Aeronautical society - vst-aero.it
- **IK1JNS (Coazze)**: Available via QRZ.com
- **IZ1MLT (Passo del Giovi)**: Available via QRZ.com
- **IQ1ZW (Genova)**: Available via QRZ.com
- **IU2MCH (Milano)**: DuckDNS contact information

For network-wide issues or improvements, contact the Heimdall project team on GitHub.

## Updates and Changes

This configuration is subject to change based on:
- Station availability and reliability
- Network optimization analysis
- New station additions
- Technical improvements

Last updated: October 2025
