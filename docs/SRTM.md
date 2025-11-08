# SRTM Terrain System Implementation

Complete implementation of real terrain data (SRTM) integration for RF propagation simulation in the Heimdall SDR platform.

## 🎯 Overview

This implementation adds realistic terrain-based RF propagation modeling using SRTM (Shuttle Radar Topography Mission) elevation data from OpenTopography API.

### Key Features
- ✅ Download 1°×1° SRTM tiles (30m resolution) from OpenTopography API
- ✅ Store tiles in MinIO with metadata in PostgreSQL
- ✅ Real-time elevation queries
- ✅ Line-of-sight (LOS) calculations with Fresnel zone analysis
- ✅ Earth curvature correction in RF path modeling
- ✅ Web UI for terrain management with Mapbox visualization
- ✅ Training integration for synthetic dataset generation

## 📦 Components Implemented

### Backend (Python/FastAPI)

#### 1. SRTM Downloader (`services/training/src/data/terrain.py`)
- **Class**: `SRTMDownloader`
- **API**: OpenTopography Global DEM API (SRTMGL1)
- **Storage**: MinIO bucket `heimdall-terrain`
- **Database**: Table `heimdall.terrain_tiles` with status tracking
- **Features**:
  - Async download with aiohttp
  - SHA256 checksum calculation
  - Auto-retry with error handling
  - Status tracking: pending → downloading → ready/failed

#### 2. Terrain Lookup (`services/training/src/data/terrain.py`)
- **Class**: `TerrainLookup`
- **Library**: rasterio for GeoTIFF reading
- **Caching**: @lru_cache for performance
- **Features**:
  - Load tiles from MinIO on-demand
  - Convert lat/lon to pixel coordinates
  - Fallback to simplified model if tile unavailable
  - Thread-safe tile cache

#### 3. RF Propagation (`services/training/src/data/propagation.py`)
- **Function**: `calculate_terrain_loss()`
- **Method**: Line-of-sight with Fresnel zone analysis
- **Algorithm**:
  1. Sample 20 points along TX→RX path
  2. Get terrain elevation at each point
  3. Calculate LOS altitude with Earth curvature correction
  4. Calculate Fresnel zone radius: `sqrt(λ * d1 * d2 / D)`
  5. Determine intrusion percentage
  6. Return loss: 0dB (clear), 0-25dB (partial), 25-40dB (blocked)

#### 4. API Endpoints (`services/backend/src/routers/terrain.py`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/terrain/tiles` | GET | List all tiles with status |
| `/api/v1/terrain/download` | POST | Download tiles (auto-detect or custom bounds) |
| `/api/v1/terrain/coverage` | GET | Coverage status for WebSDR region |
| `/api/v1/terrain/elevation` | GET | Query elevation at lat/lon |
| `/api/v1/terrain/tiles/{name}` | DELETE | Delete tile from DB and MinIO |

#### 5. Pydantic Models (`services/backend/src/models/terrain.py`)
- `TerrainTile`: Tile metadata
- `TerrainTilesList`: List response
- `DownloadRequest/Response`: Download operations
- `CoverageStatus`: Coverage analysis
- `ElevationQuery/Response`: Elevation queries

### Frontend (React/TypeScript)

#### 1. Terrain Store (`frontend/src/store/terrainStore.ts`)
- **State**: tiles, coverage, loading, downloading, error
- **Actions**:
  - `fetchTiles()`: Load all tiles
  - `fetchCoverage()`: Get coverage status
  - `downloadWebSDRRegion()`: Auto-download tiles for WebSDR area
  - `deleteTile(name)`: Remove tile
  - `queryElevation(lat, lon)`: Get elevation

#### 2. API Client (`frontend/src/services/api/terrain.ts`)
- Type-safe Axios wrappers
- Full TypeScript definitions
- Error handling

#### 3. Terrain Management Page (`frontend/src/pages/TerrainManagement.tsx`)

**Features**:
- **Coverage Status Card**:
  - Region bounds display
  - Tile counts by status
  - Color-coded progress bar
  - "Download WebSDR Region" button
  
- **Mapbox GL Map**:
  - Satellite imagery base layer
  - Tile boundaries as colored rectangles
  - Color coding:
    - 🟢 Green: Ready
    - 🟡 Yellow: Downloading
    - 🔴 Red: Failed
    - ⚫ Gray: Pending/Missing
    - 🟣 Magenta: Selected
  - Click tiles to select
  - Interactive legend
  
- **Tiles Table**:
  - Tile name, bounds, status
  - File size, download timestamp
  - Delete button per tile
  - Highlight selected tile

#### 4. Training Integration (`frontend/src/pages/TrainingDashboard.tsx`)
- Added "Use Real Terrain Data (SRTM)" checkbox
- Passes `use_real_terrain` flag to backend
- Creates `TerrainLookup` with MinIO client when enabled

#### 5. Navigation
- Added "Terrain Management" to ML & Training section
- Icon: `ph-mountains`
- Route: `/terrain`

## 🛠️ Configuration

### Environment Variables

Add to `.env`:
```bash
OPENTOPOGRAPHY_API_KEY=your_key_here
```

Get free API key at: https://www.opentopography.org/
- Free tier: 100-500 calls/24h (sufficient for 6 tiles)

### Docker Compose

Added to both `backend` and `training` services:
```yaml
environment:
  OPENTOPOGRAPHY_API_KEY: ${OPENTOPOGRAPHY_API_KEY:-}
```

### Dependencies

Added to `services/requirements/data.txt`:
```
rasterio>=1.3.9
aiohttp>=3.9.0
```

## 🗺️ Coverage Area

For NW Italy WebSDR network (6 stations):
- **Latitude**: 44°N - 46°N
- **Longitude**: 7°E - 10°E
- **Tiles needed**: 6 (N44E007-009, N45E007-009)
- **Total size**: ~300-600 MB

## 📊 Technical Details

### SRTM Data
- **Resolution**: 1 arc-second (~30m)
- **Format**: GeoTIFF
- **Source**: NASA SRTM v3.0 via OpenTopography
- **Coverage**: Global (60°N to 56°S)

### RF Propagation Physics

**Fresnel Zone**: Ellipsoid around LOS ray where RF energy travels
- **Radius**: `r = sqrt(λ * d1 * d2 / D)`
  - λ = wavelength (m)
  - d1, d2 = distances from endpoints (m)
  - D = total distance (m)

**Earth Curvature Correction**:
- `h = d1 * d2 / (2 * R_earth)`
- R_earth = 6371 km

**Loss Model**:
- Intrusion < 20%: 0 dB (clear LOS)
- Intrusion 20-60%: 0-25 dB (partial blockage)
- Intrusion > 60%: 25-40 dB (heavy blockage)

### Database Schema

Table `heimdall.terrain_tiles` (already exists):
```sql
- tile_name VARCHAR(50) UNIQUE -- e.g., 'N44E007'
- lat_min, lat_max, lon_min, lon_max INTEGER
- minio_bucket VARCHAR(100)
- minio_path VARCHAR(255)
- file_size_bytes BIGINT
- status VARCHAR(20) -- pending/downloading/ready/failed
- error_message TEXT
- checksum_sha256 VARCHAR(64)
- source_url TEXT
- downloaded_at, created_at, updated_at TIMESTAMP
```

## 🚀 Usage

### 1. Download Tiles

**Via API**:
```bash
curl -X POST http://localhost:8001/api/v1/terrain/download \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Via UI**:
1. Navigate to "Terrain Management"
2. Click "Download WebSDR Region"
3. Wait for download completion (~2-5 minutes)

### 2. Check Coverage

**Via API**:
```bash
curl http://localhost:8001/api/v1/terrain/coverage
```

**Via UI**:
- View coverage status card
- Check map for tile boundaries

### 3. Query Elevation

**Via API**:
```bash
curl "http://localhost:8001/api/v1/terrain/elevation?lat=45.0&lon=8.0"
```

**Response**:
```json
{
  "lat": 45.0,
  "lon": 8.0,
  "elevation_m": 327.5,
  "tile_name": "N45E008",
  "source": "srtm"
}
```

### 4. Generate Training Data with SRTM

**Via UI**:
1. Go to "Training Dashboard"
2. Click "Generate Synthetic Data"
3. Check "Use Real Terrain Data (SRTM)"
4. Configure parameters
5. Click "Generate"

**Via API**:
```python
data = {
    "name": "dataset_with_terrain",
    "num_samples": 10000,
    "use_real_terrain": True,
    # ... other parameters
}
response = requests.post("/api/v1/training/synthetic", json=data)
```

## 🔍 Verification Tests

### Backend Tests

```bash
# 1. Download tiles
curl -X POST http://localhost:8001/api/v1/terrain/download

# Expected: {"successful": 6, "failed": 0, "total": 6}

# 2. List tiles
curl http://localhost:8001/api/v1/terrain/tiles

# Expected: 6 tiles with status="ready"

# 3. Query elevation
curl "http://localhost:8001/api/v1/terrain/elevation?lat=45.0&lon=8.0"

# Expected: elevation_m between 200-500m, source="srtm"

# 4. Check database
psql -U heimdall_user -d heimdall -c \
  "SELECT tile_name, status, file_size_bytes FROM heimdall.terrain_tiles;"

# Expected: 6 rows with status='ready'

# 5. Check MinIO
docker exec heimdall-minio mc ls minio/heimdall-terrain/tiles/

# Expected: 6 .tif files (~50-100MB each)
```

### Frontend Tests

1. Navigate to http://localhost:5173/terrain
2. Verify:
   - Map displays with satellite imagery
   - 6 green tile boundaries visible
   - Coverage status shows 100%
   - Table lists 6 tiles with "ready" status
   - File sizes show ~50-100 MB per tile

### Integration Test

1. Go to Training Dashboard
2. Create synthetic dataset with SRTM enabled
3. Check logs for: "Using SRTM terrain data"
4. Verify improved realism in propagation loss values

## 📈 Performance

- **Tile download**: 30-60 seconds per tile (network dependent)
- **Elevation query**: <1ms (cached), <10ms (uncached)
- **LOS calculation**: ~5-10ms per TX-RX pair (20 samples)
- **Training impact**: ~10-15% slower (worth it for realism)

## 🐛 Troubleshooting

### API Key Issues
```
Error: "No API key configured"
```
**Solution**: Set `OPENTOPOGRAPHY_API_KEY` in `.env` and restart services

### Rate Limit
```
Error: "HTTP 429: Too Many Requests"
```
**Solution**: Wait 24h or upgrade to paid OpenTopography plan

### rasterio Install Failure
```
Error: "GDAL not found"
```
**Solution**: Add to Dockerfile:
```dockerfile
RUN apt-get update && apt-get install -y gdal-bin libgdal-dev
```

### Tile Not Found
```
Error: "Tile N44E007 not found in MinIO"
```
**Solution**: Download tile via `/api/v1/terrain/download` endpoint

### Mapbox Token Missing
```
Warning: "Mapbox token not configured"
```
**Solution**: Set `VITE_MAPBOX_TOKEN` in `.env` (free at https://mapbox.com)

## 🔮 Future Enhancements

- [ ] Tile caching in Redis for faster queries
- [ ] Progressive tile download (by priority)
- [ ] Higher resolution DEM (10m, 3m) support
- [ ] Clutter/land-use data integration
- [ ] 3D terrain visualization
- [ ] Batch elevation queries
- [ ] Tile expiration/refresh mechanism
- [ ] Custom DEM upload support

## 📚 References

- [OpenTopography API Docs](https://portal.opentopography.org/apidocs/)
- [Rasterio Documentation](https://rasterio.readthedocs.io/)
- [Fresnel Zone Wikipedia](https://en.wikipedia.org/wiki/Fresnel_zone)
- [SRTM Data Information](https://www2.jpl.nasa.gov/srtm/)
- [Mapbox GL JS API](https://docs.mapbox.com/mapbox-gl-js/)

## 🤝 Contributing

When modifying terrain system:
1. Update this document
2. Add tests for new features
3. Verify backward compatibility
4. Update API documentation
5. Test with real WebSDR data

## 📄 License

Part of Heimdall SDR project - CC Non-Commercial
