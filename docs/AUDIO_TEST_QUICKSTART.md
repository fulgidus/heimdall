# 🚀 Quick Start: Audio Preprocessing Frontend Test

## 📍 Access Points
- **Frontend**: http://localhost:80
- **Backend API**: http://localhost:8001
- **Audio Library Page**: http://localhost:80 → Navigate to "Training" → "Audio Library"

## 🎯 Quick Test (5 minutes)

### Step 1: Open Audio Library
```
1. Open browser: http://localhost:80
2. Click: Training → Audio Library
3. ✅ Verify: Page loads, shows 2 existing files with "Processing" badges
```

### Step 2: Upload Test File
```
Test file ready: /home/fulgidus/Documents/Projects/heimdall/test_audio_upload.wav
Size: 431 KB | Duration: 5 seconds

1. Drag test_audio_upload.wav into upload area
2. Select category: "Voice"
3. Add tags: "test, verification"
4. Click "Upload"
5. ⏱️ WATCH: Badge should go yellow → green in 2-5 seconds
```

### Step 3: Verify Real-Time Updates
**Expected behavior (automatic, no user action needed):**
```
0s:   ✅ Success alert: "Successfully uploaded..."
0s:   ✅ Yellow "Processing" badge appears
0-2s: 🔄 Polling starts (check Network tab)
2-5s: ✅ Badge changes to green "Ready"
2-5s: ✅ Shows "X chunks" below badge
2-5s: ✅ Toast: "test_audio_upload.wav is ready for training!"
5s:   ✅ Enable/disable toggle becomes active
5s:   ✅ Statistics update: "Ready: 1 / 3"
5s:   🛑 Polling stops automatically
```

## 🔍 What to Check

### Visual Elements (All should be ✅)
- [ ] Yellow badge with hourglass icon for PENDING files
- [ ] Green badge with checkmark icon for READY files
- [ ] Spinner animation next to PENDING badge
- [ ] Chunk count text: "X chunks" for READY files
- [ ] Blue alert: "Processing X file(s)..."
- [ ] Statistics card: "Ready: X / Y"

### Browser Console (Should be clean)
- [ ] No JavaScript errors
- [ ] No React warnings
- [ ] Network tab shows polling at 2-second intervals

### Functionality
- [ ] Upload works without errors
- [ ] Status changes automatically (no refresh needed)
- [ ] Enable/disable toggle disabled during processing
- [ ] Polling stops when processing complete

## 🧹 Clean Up Stuck Files (Optional)

### Option 1: Delete via UI
```
1. Click trash icon for "so-fresh-315255.mp3"
2. Confirm deletion
3. Repeat for "Il Vecchio e il Mare..." file
```

### Option 2: Test with the stuck files first
```
Just verify they show "Processing" badge (yellow)
Leave them for now to test UI displays PENDING status correctly
```

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Upload fails | Check backend logs: `docker logs heimdall-backend --tail 50` |
| Badge stays yellow forever | Delete file and retry with new upload |
| Polling doesn't stop | Refresh page to clear polling state |
| No chunk count shown | Check API: `curl http://localhost:8001/api/v1/audio-library/list \| jq` |

## ✅ Test Result Quick Check

**PASS if ALL true:**
- ✅ Upload succeeds
- ✅ Badge changes yellow → green automatically
- ✅ Chunk count appears
- ✅ No browser console errors
- ✅ Polling starts and stops correctly

**FAIL if ANY true:**
- ❌ Upload errors
- ❌ Badge stays yellow forever
- ❌ JavaScript errors in console
- ❌ Polling never stops

## 📊 Test Results

**Date:** _______  
**Browser:** _______  
**Result:** ⬜ PASS / ⬜ FAIL

**Notes:**
```
[Add your observations here]
```

---

**Need detailed instructions?** See `docs/AUDIO_PREPROCESSING_TEST_PLAN.md`

**All systems operational!** Ready to test. 🎉
