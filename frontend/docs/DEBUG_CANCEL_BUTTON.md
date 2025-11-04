# Debug Guide: Cancel Button Issue

## Issue Description
The "Cancel" button on running training jobs is showing an error: "Cannot delete job in status 'running'. Cancel the job first."

This suggests the button is calling DELETE instead of POST /cancel.

## Code Changes Made

Added comprehensive logging to track the exact flow:

### 1. Component Level (`JobCard.tsx`)
- **Button click logging**: Each button logs when clicked, including button element and data-action attribute
- **Data attributes**: All buttons have `data-action` attribute (pause/cancel/resume/delete)
- `handleCancel()` logs when called with job ID and status
- `handleDelete()` logs when called with job ID and status
- Both handlers log success/failure and error details

### 2. Store Level (`trainingStore.ts`)
- `cancelJob()` logs the endpoint being called: `POST /v1/training/jobs/{jobId}/cancel`
- `deleteJob()` logs the endpoint being called: `DELETE /v1/training/jobs/{jobId}`
- Both log success/failure

### 3. HTTP Level (`api.ts`)
- Request interceptor logs all outgoing requests (method + URL)
- Response interceptor logs all responses (status + URL)
- Error interceptor logs all failures

## How to Reproduce and Debug

1. **Start the application**:
   ```bash
   cd frontend
   npm run dev
   ```

2. **Open browser console** (F12)

3. **Navigate to Training page** and start a training job

4. **Click the "Cancel" button** on a running job

5. **Check console logs** for this sequence:

   **Expected flow:**
   ```
   [JobCard] Cancel button clicked! { jobId: "abc12345", jobStatus: "running", dataAction: "cancel" }
   [JobCard] handleCancel called for job: { id: "abc12345", status: "running", ... }
   [trainingStore] cancelJob called: { jobId: "abc12345" }
   [trainingStore] Calling POST /v1/training/jobs/abc12345/cancel
   📤 API Request: { method: "POST", url: "/v1/training/jobs/abc12345/cancel", ... }
   📥 API Response: { status: 200, url: "/v1/training/jobs/abc12345/cancel", ... }
   [trainingStore] Cancel API call successful
   [JobCard] cancelJob successful
   ```

   **If bug occurs, you might see:**
   ```
   [JobCard] Delete button clicked! { jobId: "abc12345", jobStatus: "running", dataAction: "delete" }
   [JobCard] handleDelete called for job: { id: "abc12345", status: "running", ... }
   [trainingStore] deleteJob called: { jobId: "abc12345" }
   [trainingStore] Calling DELETE /v1/training/jobs/abc12345
   📤 API Request: { method: "DELETE", url: "/v1/training/jobs/abc12345", ... }
   ❌ API Error: { status: 400, message: "Cannot delete job in status 'running'. Cancel the job first." }
   ```
   
   **Or you might see a status mismatch:**
   ```
   [JobCard] Cancel button clicked! { jobId: "abc12345", jobStatus: "completed", dataAction: "cancel" }
   ```
   This would indicate the UI is showing wrong button for the current job status.

## What to Look For

1. **Which handler is called?**
   - If you see `[JobCard] handleDelete` instead of `[JobCard] handleCancel`, the wrong button/handler is being triggered

2. **What's the job status?**
   - Check if job status in logs matches what's shown in UI
   - If status is different, there's a state synchronization issue

3. **Which HTTP method is used?**
   - Should be `POST` for cancel
   - If you see `DELETE`, that's the bug

4. **Is there a race condition?**
   - Multiple rapid clicks?
   - Status change between render and click?

## Possible Root Causes

Based on the code review, the implementation looks correct. Possible issues:

1. **State Desynchronization**: Job status in UI doesn't match actual status, causing wrong button to render
2. **Event Handler Mixup**: Multiple event listeners attached to the same button
3. **React Rendering Bug**: Button renders twice with different handlers
4. **Store State Mutation**: Job status changes mid-click
5. **WebSocket Update Timing**: Real-time status update happens during button click

## Next Steps

After reproducing with these logs:
1. Share the console output showing the exact sequence
2. Note the job status at time of click
3. Check if multiple handlers are attached to the button
4. Verify if there are any WebSocket messages arriving at the same time

## Files Modified

- `/frontend/src/pages/Training/components/JobsTab/JobCard.tsx` - Added logging to handlers
- `/frontend/src/store/trainingStore.ts` - Added logging to store methods
- `/frontend/src/lib/api.ts` - Already had request/response interceptors

## Rollback Instructions

To remove debug logging after issue is resolved:
```bash
git checkout HEAD -- frontend/src/pages/Training/components/JobsTab/JobCard.tsx
git checkout HEAD -- frontend/src/store/trainingStore.ts
```
