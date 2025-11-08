# Audio Preprocessing Frontend - Test Plan

**Date**: 2025-11-06  
**Session**: Resumed from previous session  
**Status**: Ready for testing  

## 🎯 Testing Objective

Verify that the audio preprocessing frontend correctly displays real-time status updates when audio files are uploaded and processed.

## 📋 Pre-Test Status

### System State
- ✅ All Docker containers running and healthy (13/13)
- ✅ Backend API responding: `http://localhost:8001`
- ✅ Frontend accessible: `http://localhost:80`
- ✅ Database operational with 2 existing files

### Known Issues
**Two files stuck in PENDING status** (uploaded before bug fix):
1. `so-fresh-315255.mp3` - Status: PENDING, Chunks: null
2. `Il Vecchio e il Mare, Ernest Hemingway - Audiolibro Integrale.mp3` - Status: PENDING, Chunks: null

**Root cause**: Files were uploaded before `audio_storage.py` line 668 bug fix (async/sync mismatch)

## 🧪 Test Scenarios

### Test 1: View Audio Library Page
**Objective**: Verify page loads and displays existing files

**Steps**:
1. Open browser: `http://localhost:80`
2. Navigate to Audio Library page (Training → Audio Library)
3. Verify page loads without errors

**Expected Results**:
- ✅ Page displays with breadcrumb: Home → Training → Audio Library
- ✅ Statistics cards show:
  - Total Files: 2
  - Ready: 0 / 2 (since both are PENDING)
  - Total Duration: ~2h 31m
  - Total Size: ~86.5 MB
- ✅ Table shows 2 files with yellow "Processing" badges
- ✅ Spinner icon next to "Processing" badge
- ✅ Enable/disable toggle is DISABLED for both files
- ✅ No processing alert (since these are old PENDING files, not being actively polled)

**Browser Console**: Should be clean (no errors)

---

### Test 2: Upload New Audio File
**Objective**: Verify complete upload → processing → ready flow

**Test File Available**: `test_audio_upload.wav` (431 KB, 5 seconds, 440Hz sine wave)

**Steps**:
1. On Audio Library page, scroll to "Upload Audio Sample" section
2. Click the drag-and-drop area OR drag `test_audio_upload.wav` into it
3. Verify file is selected
4. Select category: "Voice" (or any category)
5. Add tags (optional): "test, sine-wave"
6. Add description (optional): "Test upload for preprocessing verification"
7. Click "Upload" button
8. **WATCH CAREFULLY** - this is where the magic happens!

**Expected Results - Immediate (0-1 second)**:
- ✅ Success alert appears: "Successfully uploaded test_audio_upload.wav. Preprocessing chunks..."
- ✅ Upload form resets (file input cleared)
- ✅ Table refreshes and shows 3 files (2 old + 1 new)
- ✅ New file has yellow "Processing" badge with spinner
- ✅ Blue alert banner appears: "Processing 1 file... Audio chunks are being generated."

**Expected Results - After 2-5 seconds (automatic)**:
- ✅ Badge changes from yellow "Processing" to green "Ready" ✓
- ✅ Spinner disappears
- ✅ Text appears below badge: "X chunks" (e.g., "10 chunks")
- ✅ Success toast notification: "test_audio_upload.wav is ready for training!"
- ✅ Enable/disable toggle becomes ENABLED
- ✅ Statistics card updates: "Ready: 1 / 3"
- ✅ Blue processing alert disappears automatically

**Browser Network Tab** (for verification):
- Should see polling requests to `/api/v1/audio-library/list` every 2 seconds
- Polling should STOP after status changes to READY

---

### Test 3: Multiple Concurrent Uploads
**Objective**: Verify polling handles multiple files correctly

**Steps**:
1. Upload 2-3 audio files in quick succession
2. Observe polling behavior

**Expected Results**:
- ✅ Alert shows: "Processing X files..."
- ✅ All files show "Processing" badge initially
- ✅ Each file transitions to "Ready" independently
- ✅ Success toast appears for each file as it completes
- ✅ Polling stops only when ALL files are ready

---

### Test 4: Enable/Disable Functionality
**Objective**: Verify toggle only works for READY files

**Steps**:
1. Try to toggle the switch for a PENDING file (should be disabled)
2. Toggle the switch for a READY file
3. Verify API call succeeds

**Expected Results**:
- ✅ PENDING files: Toggle is disabled (grayed out)
- ✅ READY files: Toggle works normally
- ✅ Toggle state persists after page refresh

---

### Test 5: Refresh During Processing
**Objective**: Verify state persistence

**Steps**:
1. Upload a file
2. While it's processing (yellow badge), refresh the page (F5)
3. Observe behavior

**Expected Results**:
- ✅ Page reloads
- ✅ File still shows "Processing" badge
- ✅ **BUT**: Polling does NOT auto-resume (by design - user must manually refresh or wait)
- ✅ User can click "Refresh" button to update status manually

**Note**: This is expected behavior. WebSocket integration (future enhancement) will fix this.

---

### Test 6: Handle Stuck PENDING Files
**Objective**: Clean up the 2 stuck files

**Option A: Delete via UI** (Recommended)
1. Click trash icon for `so-fresh-315255.mp3`
2. Confirm deletion
3. Repeat for second file
4. Verify statistics update

**Option B: Manual Database Fix**
```bash
docker exec heimdall-backend psql -U heimdall_user -d heimdall -c "
UPDATE audio_library 
SET processing_status = 'FAILED' 
WHERE processing_status = 'PENDING' 
  AND total_chunks IS NULL
  AND uploaded_at < NOW() - INTERVAL '5 minutes';
"
```

**Expected Results**:
- ✅ Files removed from library
- ✅ Statistics update to reflect deletion
- ✅ Success message appears

---

## 📊 Test Results Template

Use this checklist as you test:

### Visual Elements
- [ ] Statistics cards display correctly
- [ ] Processing badges show correct colors (yellow/green/red)
- [ ] Spinner animation appears for PENDING files
- [ ] Chunk count appears for READY files ("X chunks")
- [ ] Processing alert appears/disappears correctly

### Functionality
- [ ] Upload works without errors
- [ ] Polling starts automatically after upload
- [ ] Status changes from PENDING → READY automatically
- [ ] Success toast appears when processing completes
- [ ] Polling stops when no files are pending
- [ ] Enable/disable toggle disabled for PENDING files
- [ ] Statistics update correctly

### Performance
- [ ] Polling interval is ~2 seconds (check Network tab)
- [ ] Preprocessing completes in 2-5 seconds for small files
- [ ] No memory leaks (check browser Task Manager during extended polling)
- [ ] No excessive API calls when no files are pending

### Browser Console
- [ ] No JavaScript errors
- [ ] No React warnings
- [ ] Polling logs visible (if debug mode enabled)

---

## 🐛 Troubleshooting

### Issue: Upload succeeds but file stays PENDING forever
**Cause**: Backend preprocessing task failed  
**Debug**:
```bash
# Check backend logs
docker logs heimdall-backend --tail 50

# Check Celery worker logs
docker logs heimdall-backend --tail 100 | grep -i "celery\|audio\|preprocessing"

# Check if audio chunks were created in MinIO
# (Backend should log chunk creation)
```

### Issue: Polling doesn't start after upload
**Cause**: Frontend not adding file ID to polling set  
**Debug**: Open browser console, check for errors  
**Fix**: Verify `uploadedSample.processing_status === 'PENDING'` in upload handler

### Issue: Badge shows "Ready" but no chunk count
**Cause**: `total_chunks` is null  
**Debug**: Check API response: `curl http://localhost:8001/api/v1/audio-library/list | jq`

### Issue: Network tab shows polling never stops
**Cause**: File stuck in PENDING, polling set never cleared  
**Fix**: Delete file or manually update DB status

---

## ✅ Success Criteria

All of the following must be true for test to pass:

1. **Upload Flow**: File uploads without errors
2. **Real-time Updates**: Status changes from PENDING → READY automatically
3. **Visual Feedback**: Badges, spinners, and chunk count display correctly
4. **Notifications**: Success toast and processing alert appear
5. **Statistics**: Cards update to reflect current state
6. **Polling Behavior**: Starts automatically, stops when done, 2s interval
7. **No Errors**: Browser console clean, no backend errors

---

## 🚀 After Testing

### If Tests Pass
1. Document results in this file (add screenshots if possible)
2. Mark "Stuck PENDING files" issue as resolved
3. Move to Phase 7 next steps (E2E testing with Playwright)
4. Consider WebSocket integration as enhancement

### If Tests Fail
1. Document failure mode in this file
2. Check troubleshooting section
3. Review backend logs for errors
4. Consider rolling back frontend changes if major issues

---

## 📝 Test Execution Log

**Tester**: _________  
**Date**: 2025-11-06  
**Browser**: _________  
**Result**: ⬜ PASS / ⬜ FAIL  

### Notes:
_Add observations, screenshots, or issues discovered during testing_

---

## 📚 Related Documentation

- **Implementation Details**: `docs/AUDIO_PREPROCESSING_FRONTEND.md`
- **Backend Integration Tests**: `test_audio_preprocessing_integration.py`
- **Backend Storage**: `services/backend/src/storage/audio_storage.py`
- **Frontend Component**: `frontend/src/pages/AudioLibrary.tsx`
- **API Client**: `frontend/src/services/api/audioLibrary.ts`

---

**Ready to test!** 🎉 Open `http://localhost:80` and navigate to Audio Library.
