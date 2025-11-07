# Audio Library Import/Export Feature

**Status**: ✅ Complete  
**Date**: 2025-11-07  
**Phase**: Phase 7 (Frontend Development)

## Overview

This document describes the complete implementation of audio library import/export functionality for the Heimdall SDR project. This feature allows users to export and import audio library entries (with their associated 1-second audio chunks) via `.heimdall` bundle files.

## Architecture

### Backend Implementation

**File**: `services/backend/src/routers/import_export.py`

#### 1. Export Metadata Endpoint (`GET /api/import-export/export/metadata`)

Returns metadata about available audio library entries for export:

```python
audio_library_rows = await conn.fetch(
    """
    SELECT 
        al.id, al.filename, al.category,
        al.file_size_bytes, al.duration_seconds,
        al.total_chunks, al.created_at
    FROM heimdall.audio_library al
    WHERE al.processing_status = 'completed'
    ORDER BY al.created_at DESC
    """
)
```

**Response includes**:
- `id`: Audio library entry UUID
- `filename`: Original filename
- `category`: Classification (music, speech, noise, etc.)
- `duration_seconds`: Total duration
- `total_chunks`: Number of 1-second chunks
- `file_size_bytes`: Total size of all chunks
- `created_at`: Creation timestamp

**Size Estimation**:
```python
# Each chunk is ~200KB (1 second @ 200kHz sample rate)
# Plus base64 encoding overhead (~1.33x) and JSON metadata (~500 bytes per chunk)
audio_library_size = sum(
    (al["total_chunks"] * 200000 * 1.33) + (al["total_chunks"] * 500) 
    for al in audio_library
)
```

#### 2. Export Data Endpoint (`POST /api/import-export/export`)

Exports selected audio library entries with their chunks:

**Process**:
1. Query `heimdall.audio_library` table for selected IDs
2. Query `heimdall.audio_chunks` table for associated chunks
3. Download `.npy` files from MinIO bucket `heimdall-audio-chunks`
4. Base64-encode chunk data for JSON transport
5. Build `ExportedAudioLibrary` objects with `ExportedAudioChunk` arrays

**Request**:
```json
{
  "creator": {"username": "user@example.com", "name": "User Name"},
  "description": "Export description",
  "audio_library_ids": ["uuid1", "uuid2", ...],
  "include_settings": true,
  "include_sources": true,
  ...
}
```

**Response** (`.heimdall` file):
```json
{
  "metadata": {
    "version": "1.0.0",
    "created_at": "2025-11-07T...",
    "creator": {...}
  },
  "sections": {
    "audio_library": [
      {
        "id": "uuid",
        "filename": "example.wav",
        "category": "music",
        "chunks": [
          {
            "chunk_index": 0,
            "duration_seconds": 1.0,
            "sample_rate": 200000,
            "num_samples": 200000,
            "audio_data_base64": "..."
          },
          ...
        ]
      }
    ]
  },
  "section_sizes": {
    "audio_library": 12345678
  }
}
```

#### 3. Import Data Endpoint (`POST /api/import-export/import`)

Imports audio library entries from `.heimdall` file:

**Process**:
1. Parse `heimdall_file.sections.audio_library` array
2. Insert/update `heimdall.audio_library` table entries
3. Base64-decode audio chunk data
4. Upload chunks to MinIO as `imported/{audio_id}/{chunk_index:04d}.npy`
5. Insert/update `heimdall.audio_chunks` table entries

**Request**:
```json
{
  "heimdall_file": {...},
  "import_audio_library": true,
  "overwrite_existing": false
}
```

**Response**:
```json
{
  "message": "Import completed successfully",
  "imported_counts": {
    "sources": 10,
    "websdrs": 7,
    "sessions": 5,
    "sample_sets": 2,
    "models": 1,
    "audio_library": 3
  },
  "errors": []
}
```

**Error Handling**:
- Graceful degradation: If one chunk fails, continue with others
- Partial success: Import counts track successful imports
- Error reporting: Detailed error messages in response

### Data Models

**File**: `services/backend/src/models/import_export.py`

#### AvailableAudioLibrary
```python
class AvailableAudioLibrary(BaseModel):
    id: str
    filename: str
    category: str
    duration_seconds: float
    total_chunks: int
    file_size_bytes: int
    created_at: str
```

#### ExportedAudioChunk
```python
class ExportedAudioChunk(BaseModel):
    id: str
    chunk_index: int
    duration_seconds: float
    sample_rate: int
    num_samples: int
    file_size_bytes: int
    original_offset_seconds: float
    rms_amplitude: float | None
    created_at: str
    audio_data_base64: str | None
```

#### ExportedAudioLibrary
```python
class ExportedAudioLibrary(BaseModel):
    id: str
    filename: str
    category: str
    tags: list[str] | None
    file_size_bytes: int
    duration_seconds: float
    sample_rate: int
    channels: int
    audio_format: str
    processing_status: str
    total_chunks: int
    enabled: bool
    created_at: str
    updated_at: str
    chunks: list[ExportedAudioChunk] | None
```

### Frontend Implementation

**File**: `frontend/src/services/api/import-export.ts`

TypeScript interfaces matching backend models (see Data Models section).

**File**: `frontend/src/pages/ImportExport.tsx`

#### State Management
```typescript
const [selectedAudioLibrary, setSelectedAudioLibrary] = useState<Set<string>>(new Set());
```

#### Auto-Selection on Load
```typescript
useEffect(() => {
  if (metadata?.audio_library && metadata.audio_library.length > 0) {
    setSelectedAudioLibrary(new Set(metadata.audio_library.map(a => a.id)));
  }
}, [metadata]);
```

#### Export Section UI (lines 489-530)
- Displays scrollable list of audio library entries
- Shows filename, chunk count, and file size
- Allows individual selection/deselection
- Shows selection count

#### Import Section UI (lines 657-669)
- Shows audio library checkbox when section is present in file
- Displays count of audio library entries in file
- Toggleable via `importForm.import_audio_library`

#### Success Message (lines 710-713)
- Displays imported audio library count
- Included in overall import summary

## Storage Structure

### MinIO Bucket: `heimdall-audio-chunks`

#### Original Chunks
```
s3://heimdall-audio-chunks/{audio_id}/{chunk_index:04d}.npy
```
Example: `s3://heimdall-audio-chunks/123e4567-e89b-12d3-a456-426614174000/0000.npy`

#### Imported Chunks
```
s3://heimdall-audio-chunks/imported/{audio_id}/{chunk_index:04d}.npy
```
Example: `s3://heimdall-audio-chunks/imported/123e4567-e89b-12d3-a456-426614174000/0000.npy`

### Database Tables

#### `heimdall.audio_library`
- Stores audio file metadata
- Fields: id, filename, category, tags, duration_seconds, sample_rate, channels, etc.
- Only exports entries with `processing_status = 'completed'`

#### `heimdall.audio_chunks`
- Stores chunk metadata (linked to audio_library via `audio_id`)
- Fields: id, audio_id, chunk_index, duration_seconds, sample_rate, num_samples, minio_path, etc.
- Each chunk represents 1 second of audio @ 200kHz sample rate

## Usage

### Exporting Audio Library

1. Navigate to Import/Export page
2. Audio Library section shows all completed audio entries
3. Select desired audio files (all selected by default)
4. Click "Export & Download"
5. `.heimdall` file downloads with embedded audio chunk data

### Importing Audio Library

1. Navigate to Import/Export page
2. Click "Load .heimdall File"
3. Select file from file system
4. Review "Audio Library" checkbox (enabled by default)
5. Choose "Overwrite existing data" if desired
6. Click "Confirm Import"
7. Audio chunks uploaded to MinIO, metadata inserted into database

## Performance Considerations

### Export
- **Chunk Size**: ~200KB per 1-second chunk (raw .npy file)
- **Base64 Overhead**: +33% size increase for JSON transport
- **Network Bound**: Download from MinIO is the bottleneck
- **Parallel Processing**: Currently sequential, could be parallelized

### Import
- **Base64 Decoding**: Minimal overhead
- **Upload to MinIO**: Network bound, sequential
- **Transaction Safety**: All imports in single database transaction

### Size Estimates
- **1-minute audio file**: ~60 chunks × 200KB × 1.33 = ~16MB in .heimdall file
- **10-minute file**: ~160MB
- **Recommendation**: Export audio library separately for large collections

## Testing Checklist

- [x] TypeScript types compile without errors
- [ ] Export metadata includes audio library entries
- [ ] Export includes selected audio library items with chunks
- [ ] Audio chunks correctly downloaded from MinIO during export
- [ ] Base64 encoding/decoding works correctly
- [ ] Import creates database entries correctly
- [ ] Import uploads chunks to MinIO with correct paths
- [ ] Import counts show audio_library correctly
- [ ] Round-trip integrity: export → import → verify chunks match
- [ ] Error handling: Partial failures don't break entire import
- [ ] UI displays audio library selection correctly
- [ ] UI shows import success with audio_library count

## Known Limitations

1. **Sequential Processing**: Chunks are processed one at a time (could be parallelized)
2. **Memory Usage**: Large audio files held in memory during export/import
3. **No Compression**: Audio data is base64-encoded but not compressed
4. **No Progress Indicator**: UI doesn't show progress during long operations

## Future Enhancements

1. **Streaming Export/Import**: Process chunks in batches to reduce memory usage
2. **Compression**: Add gzip compression to reduce file size
3. **Progress Indicators**: Show progress bars for large audio library exports/imports
4. **Selective Chunk Export**: Allow exporting subset of chunks (e.g., first 30 seconds)
5. **Audio Preview**: Play audio chunks before export/import
6. **Batch Upload**: Parallel MinIO uploads during import

## References

- **Backend Router**: `services/backend/src/routers/import_export.py`
- **Backend Models**: `services/backend/src/models/import_export.py`
- **Frontend Types**: `frontend/src/services/api/import-export.ts`
- **Frontend UI**: `frontend/src/pages/ImportExport.tsx`
- **Database Schema**: `db/migrations/` (audio_library and audio_chunks tables)
- **Phase 5 Export/Import Docs**: `docs/agents/20251103_phase5_heimdall_export_import_complete.md`

## Session Continuity

This implementation was completed across two sessions:

1. **Session 1** (2025-11-06): Backend implementation (routes, models, database queries)
2. **Session 2** (2025-11-07): Frontend UI components (selection, display, import controls)

**Key Decision**: Used RabbitMQ event bus pattern (see AGENTS.md) if real-time progress updates needed in future.

**Next Steps**: End-to-end testing with real audio data.
