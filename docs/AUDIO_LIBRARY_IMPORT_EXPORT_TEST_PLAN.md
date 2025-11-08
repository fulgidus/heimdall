# Audio Library Import/Export Test Plan

**Feature**: Audio Library Import/Export  
**Date**: 2025-11-07  
**Status**: Ready for Testing

## Prerequisites

1. **Backend Running**: Heimdall backend service operational
2. **Database**: PostgreSQL with `heimdall.audio_library` and `heimdall.audio_chunks` tables
3. **MinIO**: Bucket `heimdall-audio-chunks` accessible
4. **Test Data**: At least 2-3 audio library entries with `processing_status = 'completed'`
5. **Frontend**: React app running and accessible

## Test Scenarios

### Test 1: Export Metadata Retrieval

**Objective**: Verify metadata endpoint returns audio library entries

**Steps**:
```bash
curl -X GET http://localhost:8001/api/import-export/export/metadata \
  -H "Content-Type: application/json" | jq .
```

**Expected Result**:
- Response status: 200 OK
- Response contains `audio_library` array
- Each entry has: `id`, `filename`, `category`, `duration_seconds`, `total_chunks`, `file_size_bytes`, `created_at`
- `estimated_sizes.audio_library` is a positive integer

**Pass Criteria**: ✅ All fields present and correctly typed

---

### Test 2: Export Single Audio Library Entry

**Objective**: Export one audio file with all chunks

**Steps**:
1. Get audio library ID from metadata endpoint
2. Create export request:
```bash
curl -X POST http://localhost:8001/api/import-export/export \
  -H "Content-Type: application/json" \
  -d '{
    "creator": {"username": "test@example.com", "name": "Test User"},
    "description": "Test export of single audio file",
    "audio_library_ids": ["<AUDIO_ID>"],
    "include_settings": false,
    "include_sources": false,
    "include_websdrs": false,
    "include_sessions": false
  }' | jq . > test-export.heimdall
```

**Expected Result**:
- File created: `test-export.heimdall`
- File is valid JSON
- Contains `sections.audio_library` array with 1 entry
- Entry has `chunks` array with all chunks
- Each chunk has `audio_data_base64` field (non-null)

**Verification**:
```bash
cat test-export.heimdall | jq '.sections.audio_library[0]' | head -20
cat test-export.heimdall | jq '.sections.audio_library[0].chunks | length'
```

**Pass Criteria**: ✅ Chunks count matches expected, base64 data present

---

### Test 3: Export Multiple Audio Library Entries

**Objective**: Export multiple audio files in one .heimdall file

**Steps**:
```bash
curl -X POST http://localhost:8001/api/import-export/export \
  -H "Content-Type: application/json" \
  -d '{
    "creator": {"username": "test@example.com"},
    "audio_library_ids": ["<AUDIO_ID_1>", "<AUDIO_ID_2>", "<AUDIO_ID_3>"]
  }' | jq . > multi-export.heimdall
```

**Expected Result**:
- `sections.audio_library` array has 3 entries
- Each entry has unique `id`
- All entries have chunks with base64 data

**Verification**:
```bash
cat multi-export.heimdall | jq '.sections.audio_library | length'
cat multi-export.heimdall | jq '.sections.audio_library[].id'
```

**Pass Criteria**: ✅ 3 entries exported, all have chunks

---

### Test 4: Import Audio Library (New Entries)

**Objective**: Import audio library entries that don't exist in database

**Steps**:
1. Backup database (optional):
```sql
SELECT * INTO TEMP TABLE audio_library_backup FROM heimdall.audio_library;
SELECT * INTO TEMP TABLE audio_chunks_backup FROM heimdall.audio_chunks;
```

2. Delete test audio entries:
```sql
DELETE FROM heimdall.audio_chunks WHERE audio_id IN ('<AUDIO_ID_1>', '<AUDIO_ID_2>');
DELETE FROM heimdall.audio_library WHERE id IN ('<AUDIO_ID_1>', '<AUDIO_ID_2>');
```

3. Import .heimdall file:
```bash
curl -X POST http://localhost:8001/api/import-export/import \
  -H "Content-Type: application/json" \
  -d @test-export.heimdall | jq .
```

**Expected Result**:
- Response: `"message": "Import completed successfully"`
- `imported_counts.audio_library` > 0
- No errors in `errors` array

**Verification**:
```sql
-- Check audio_library table
SELECT id, filename, total_chunks FROM heimdall.audio_library WHERE id IN ('<AUDIO_ID_1>', '<AUDIO_ID_2>');

-- Check audio_chunks table
SELECT audio_id, chunk_index, minio_path FROM heimdall.audio_chunks WHERE audio_id IN ('<AUDIO_ID_1>', '<AUDIO_ID_2>') ORDER BY chunk_index;

-- Verify MinIO paths start with "imported/"
SELECT DISTINCT minio_path FROM heimdall.audio_chunks WHERE audio_id = '<AUDIO_ID_1>';
```

**Pass Criteria**: 
- ✅ Entries exist in database
- ✅ Chunks have `minio_path` starting with `s3://heimdall-audio-chunks/imported/`
- ✅ Chunk count matches original

---

### Test 5: Import with Overwrite

**Objective**: Test overwrite_existing flag

**Steps**:
1. Modify an audio library entry:
```sql
UPDATE heimdall.audio_library SET filename = 'OLD_FILENAME.wav' WHERE id = '<AUDIO_ID_1>';
```

2. Import with `overwrite_existing: true`:
```bash
curl -X POST http://localhost:8001/api/import-export/import \
  -H "Content-Type: application/json" \
  -d '{
    "heimdall_file": ...,
    "import_audio_library": true,
    "overwrite_existing": true
  }'
```

**Expected Result**:
- Filename updated to original value from .heimdall file
- Chunks updated with new MinIO paths

**Verification**:
```sql
SELECT filename FROM heimdall.audio_library WHERE id = '<AUDIO_ID_1>';
```

**Pass Criteria**: ✅ Filename matches exported value (not "OLD_FILENAME.wav")

---

### Test 6: Import with Conflict (No Overwrite)

**Objective**: Test skipping existing entries when overwrite is false

**Steps**:
1. Import with `overwrite_existing: false` (already existing entries)
2. Check response counts

**Expected Result**:
- `imported_counts.audio_library` = 0 (skipped)
- No errors
- Database unchanged

**Verification**:
```sql
SELECT COUNT(*) FROM heimdall.audio_library; -- Should not increase
```

**Pass Criteria**: ✅ No new entries created, existing entries unchanged

---

### Test 7: Chunk Data Integrity

**Objective**: Verify audio chunk data survives round-trip

**Steps**:
1. Export audio library entry
2. Extract base64 data from first chunk
3. Decode and compare with original MinIO file:

```python
import base64
import json
import numpy as np

# Load exported file
with open('test-export.heimdall', 'r') as f:
    data = json.load(f)

# Get first chunk
chunk = data['sections']['audio_library'][0]['chunks'][0]
b64_data = chunk['audio_data_base64']

# Decode
audio_bytes = base64.b64decode(b64_data)

# Load as numpy array (assuming .npy format)
import io
audio_array = np.load(io.BytesIO(audio_bytes))

print(f"Shape: {audio_array.shape}")
print(f"Dtype: {audio_array.dtype}")
print(f"Sample values: {audio_array[:10]}")
```

**Expected Result**:
- Array shape matches expected (200000 samples for 1-second @ 200kHz)
- Dtype is float32 or float64
- Values in reasonable range (e.g., -1.0 to 1.0 for normalized audio)

**Pass Criteria**: ✅ Decoded array matches expected format

---

### Test 8: MinIO File Upload Verification

**Objective**: Verify imported chunks exist in MinIO

**Steps**:
1. After import, check MinIO bucket:
```bash
# Using AWS CLI (configured for MinIO)
aws --endpoint-url http://localhost:9000 s3 ls s3://heimdall-audio-chunks/imported/<AUDIO_ID>/ --recursive
```

**Expected Result**:
- Files exist: `imported/<AUDIO_ID>/0000.npy`, `0001.npy`, ...
- File count matches `total_chunks` from database

**Pass Criteria**: ✅ All chunk files present in MinIO

---

### Test 9: Frontend UI - Export Selection

**Objective**: Verify frontend displays and allows selection of audio library entries

**Steps**:
1. Navigate to http://localhost:5173/import-export
2. Scroll to "Audio Library" section in Export form
3. Verify display

**Expected Result**:
- Section visible with title "Audio Library (N)" where N > 0
- Scrollable list of audio entries
- Each entry shows: filename, chunk count, file size
- Checkboxes for each entry
- Selection counter shows "N of N selected" (all selected by default)

**Manual Interaction**:
- Uncheck some entries → counter updates
- Check some entries → counter updates
- Click "Export & Download" → file downloads

**Pass Criteria**: ✅ UI functional, selection works, export triggered

---

### Test 10: Frontend UI - Import

**Objective**: Verify frontend import workflow

**Steps**:
1. Navigate to http://localhost:5173/import-export
2. Click "Load .heimdall File"
3. Select `test-export.heimdall`
4. Verify import options

**Expected Result**:
- File loaded successfully
- "Audio Library (N)" checkbox visible
- Checkbox checked by default
- Click "Confirm Import" → import succeeds
- Success message shows: "Audio Library: N"

**Pass Criteria**: ✅ Import completes, count displayed correctly

---

### Test 11: Error Handling - Missing MinIO File

**Objective**: Test graceful handling of missing audio chunks

**Steps**:
1. Delete a chunk from MinIO:
```bash
aws --endpoint-url http://localhost:9000 s3 rm s3://heimdall-audio-chunks/<AUDIO_ID>/0001.npy
```

2. Attempt export with this audio_id

**Expected Result**:
- Export completes (partial success)
- Log message: "Failed to download audio chunk"
- Affected chunk has `audio_data_base64: null`

**Pass Criteria**: ✅ Export doesn't crash, error logged

---

### Test 12: Error Handling - Invalid Base64 Data

**Objective**: Test import with corrupted data

**Steps**:
1. Edit .heimdall file, corrupt base64 data:
```bash
# Change some characters in audio_data_base64 field
sed -i 's/AAA/###/g' test-export.heimdall
```

2. Attempt import

**Expected Result**:
- Import fails gracefully
- Error message: "Failed to upload audio chunk"
- Other entries still imported (if valid)

**Pass Criteria**: ✅ Import doesn't crash, error reported

---

## Performance Tests

### Test 13: Large Audio File Export

**Objective**: Test export of 10-minute audio file

**Setup**: Upload 10-minute audio file (~600 chunks)

**Steps**:
1. Export with this audio_id
2. Measure time and file size

**Expected Result**:
- Export completes within reasonable time (< 2 minutes)
- File size ~160MB (600 chunks × 200KB × 1.33)

**Pass Criteria**: ✅ Completes without timeout or memory issues

---

### Test 14: Multiple Large Files Import

**Objective**: Test import of 5 × 10-minute files

**Steps**:
1. Import .heimdall file with 5 large audio entries
2. Monitor memory usage and time

**Expected Result**:
- Import completes (may take several minutes)
- No out-of-memory errors
- All chunks uploaded to MinIO

**Pass Criteria**: ✅ Completes successfully, no crashes

---

## Test Results Summary

| Test # | Test Name | Status | Notes |
|--------|-----------|--------|-------|
| 1 | Export Metadata | ⏳ Pending | |
| 2 | Export Single Entry | ⏳ Pending | |
| 3 | Export Multiple Entries | ⏳ Pending | |
| 4 | Import New Entries | ⏳ Pending | |
| 5 | Import with Overwrite | ⏳ Pending | |
| 6 | Import with Conflict | ⏳ Pending | |
| 7 | Chunk Data Integrity | ⏳ Pending | |
| 8 | MinIO Upload Verification | ⏳ Pending | |
| 9 | Frontend Export UI | ⏳ Pending | |
| 10 | Frontend Import UI | ⏳ Pending | |
| 11 | Error: Missing MinIO File | ⏳ Pending | |
| 12 | Error: Invalid Base64 | ⏳ Pending | |
| 13 | Performance: Large Export | ⏳ Pending | |
| 14 | Performance: Large Import | ⏳ Pending | |

## Test Environment

- **Backend URL**: http://localhost:8001
- **Frontend URL**: http://localhost:5173
- **MinIO Console**: http://localhost:9001
- **Database**: PostgreSQL on localhost:5432

## Known Issues

- None yet (pending testing)

## Test Data Requirements

Prepare test data:
1. Upload 2-3 audio files (1-2 minutes each) via Audio Library page
2. Wait for processing to complete (`processing_status = 'completed'`)
3. Verify chunks exist in MinIO bucket
4. Note audio IDs for test scenarios

## Regression Testing

After any changes to import/export code, re-run:
- Test 1 (metadata)
- Test 2 (single export)
- Test 4 (import new)
- Test 7 (data integrity)
- Test 9 (frontend export)
- Test 10 (frontend import)

## Sign-off

- [ ] All tests passed
- [ ] Performance acceptable
- [ ] Error handling verified
- [ ] Documentation updated
- [ ] Ready for production

**Tester**: _____________  
**Date**: _____________
