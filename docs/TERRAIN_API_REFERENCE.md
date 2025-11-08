# Terrain Management API - Quick Reference

## Training Service Terrain Endpoints

Base URL: `http://localhost:8002/synthetic/terrain`

### 1. Check Terrain System Status

**Endpoint**: `GET /synthetic/terrain/status`

**Purpose**: Check if SRTM is enabled and MinIO is accessible

**Example**:
```bash
curl http://localhost:8002/synthetic/terrain/status
```

**Response**:
```json
{
  "srtm_enabled": true,
  "minio_configured": true,
  "minio_connection": "healthy",
  "bucket_exists": true,
  "bucket_name": "heimdall-terrain",
  "backend_service_url": "http://backend:8000"
}
```

---

### 2. Check Terrain Coverage

**Endpoint**: `POST /synthetic/terrain/coverage`

**Purpose**: Check which SRTM tiles are available for a region before generating data

**Example** (Northwestern Italy - WebSDR region):
```bash
curl -X POST http://localhost:8002/synthetic/terrain/coverage \
  -H "Content-Type: application/json" \
  -d '{
    "lat_min": 44.0,
    "lat_max": 46.0,
    "lon_min": 7.0,
    "lon_max": 13.0
  }'
```

**Response**:
```json
{
  "total_tiles": 18,
  "available_tiles": 12,
  "missing_tiles": 6,
  "coverage_percent": 66.67,
  "tiles": [
    {
      "tile_name": "N44E007",
      "exists": true,
      "lat_min": 44,
      "lat_max": 45,
      "lon_min": 7,
      "lon_max": 8
    },
    {
      "tile_name": "N44E008",
      "exists": false,
      "lat_min": 44,
      "lat_max": 45,
      "lon_min": 8,
      "lon_max": 9
    }
  ],
  "missing_tile_names": ["N44E008", "N44E009", "N45E010", "N45E011", "N45E012", "N45E013"]
}
```

---

### 3. Download Missing Tiles (Redirect)

**Endpoint**: `POST /synthetic/terrain/download`

**Purpose**: Get information about downloading missing tiles (redirects to backend service)

**Example**:
```bash
curl -X POST http://localhost:8002/synthetic/terrain/download \
  -H "Content-Type: application/json" \
  -d '{
    "lat_min": 44.0,
    "lat_max": 45.0,
    "lon_min": 7.0,
    "lon_max": 8.0
  }'
```

**Response**:
```json
{
  "message": "Please use the backend service to download 4 tiles",
  "tiles_to_download": ["N44E007", "N44E008", "N45E007", "N45E008"],
  "backend_url": "http://backend:8000/api/v1/terrain/download"
}
```

**To Actually Download**: Use the backend service endpoint:
```bash
curl -X POST http://localhost:8000/api/v1/terrain/download \
  -H "Content-Type: application/json" \
  -d '{
    "bounds": {
      "lat_min": 44.0,
      "lat_max": 45.0,
      "lon_min": 7.0,
      "lon_max": 8.0
    }
  }'
```

---

## Backend Service Terrain Endpoints

Base URL: `http://localhost:8000/api/v1/terrain`

### Full List of Backend Endpoints

1. **`GET /tiles`** - List all downloaded tiles
2. **`POST /download`** - Download tiles with real-time progress (WebSocket)
3. **`GET /coverage`** - Check coverage for WebSDR region (auto-detect bounds)
4. **`GET /elevation?lat=45.0&lon=8.0`** - Query elevation at specific coordinates
5. **`DELETE /tiles/{tile_name}`** - Delete a specific tile

See `/services/backend/src/routers/terrain.py` for full documentation.

---

## Workflow: Generate Synthetic Data with SRTM

### Step 1: Check Terrain Status
```bash
curl http://localhost:8002/synthetic/terrain/status
```

Expected: `minio_connection: "healthy"`, `bucket_exists: true`

### Step 2: Check Coverage for Your Region
```bash
curl -X POST http://localhost:8002/synthetic/terrain/coverage \
  -H "Content-Type: application/json" \
  -d '{
    "lat_min": 44.0,
    "lat_max": 46.0,
    "lon_min": 7.0,
    "lon_max": 13.0
  }'
```

Expected: `coverage_percent: 100.0` (or close to it)

### Step 3: Download Missing Tiles (if any)
```bash
# Use backend service for downloads
curl -X POST http://localhost:8000/api/v1/terrain/download \
  -H "Content-Type: application/json" \
  -d '{
    "bounds": {
      "lat_min": 44.0,
      "lat_max": 46.0,
      "lon_min": 7.0,
      "lon_max": 13.0
    }
  }'
```

Monitor progress via WebSocket or poll `/api/v1/terrain/tiles` endpoint.

### Step 4: Generate Synthetic Data
```bash
curl -X POST http://localhost:8002/synthetic/generate \
  -H "Content-Type: application/json" \
  -d '{
    "name": "srtm_enabled_dataset",
    "description": "Dataset with realistic terrain propagation",
    "num_samples": 1000,
    "frequency_mhz": 145.0,
    "tx_power_dbm": 37.0,
    "min_snr_db": 10.0,
    "min_receivers": 4,
    "max_gdop": 5.0,
    "dataset_type": "feature_based",
    "seed": 42
  }'
```

The generator will automatically:
- Initialize TerrainLookup with SRTM support in each worker thread
- Load SRTM tiles from MinIO as needed
- Use real elevation data in propagation calculations
- Fall back to simplified model if specific tiles are missing

### Step 5: Monitor Progress
```bash
# Use job_id from step 4 response
curl http://localhost:8002/synthetic/jobs/{job_id}
```

---

## Configuration

### Enable SRTM in Synthetic Generation

SRTM is automatically enabled when:
1. MinIO is configured and accessible
2. `heimdall-terrain` bucket exists
3. SRTM tiles are available for the region

The generator uses `SyntheticGenerationConfig.use_srtm_terrain` flag (defaults to `True` if MinIO is available).

### Environment Variables

Training service requires:
```bash
BACKEND_URL=http://backend:8000  # For terrain download redirects
MINIO_URL=http://minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
BACKEND_SRC_PATH=/app/backend/src  # For MinIO client import
```

---

## Troubleshooting

### Problem: `minio_connection: "failed"`

**Solution**: Check if MinIO is running and accessible:
```bash
docker-compose ps minio
curl http://localhost:9000/minio/health/live
```

### Problem: `coverage_percent: 0.0`

**Solution**: Download SRTM tiles for your region:
```bash
curl -X POST http://localhost:8000/api/v1/terrain/download \
  -H "Content-Type: application/json" \
  -d '{"bounds": {"lat_min": 44.0, "lat_max": 46.0, "lon_min": 7.0, "lon_max": 13.0}}'
```

### Problem: Generation job fails with terrain errors

**Solution**: 
1. Check logs for specific error
2. Verify SRTM tiles exist: `curl http://localhost:8000/api/v1/terrain/tiles`
3. If tiles missing, generator will fall back to simplified model (logs will show warnings)

### Problem: `OPENTOPOGRAPHY_API_KEY not configured`

**Solution**: Set API key in environment:
```bash
export OPENTOPOGRAPHY_API_KEY=your_api_key
# Or add to .env file
```

Get free API key at: https://opentopography.org/

---

## References

- **Full Documentation**: `/docs/SRTM.md`
- **Backend Terrain Router**: `/services/backend/src/routers/terrain.py`
- **Training Synthetic API**: `/services/training/src/api/synthetic.py`
- **Terrain Module**: `/services/common/terrain/terrain.py`

---

**Last Updated**: 2025-11-04  
**Author**: OpenCode Session (SRTM Terrain Integration)
