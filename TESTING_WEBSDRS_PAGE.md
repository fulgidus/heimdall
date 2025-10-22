# Testing Guide: WebSDR Management Page Integration

## Overview
This guide explains how to test the newly implemented real-time WebSDR data integration on the `/websdrs` page.

## Changes Made

### Frontend Changes
1. **WebSDRManagement.tsx**
   - Replaced hardcoded WebSDR data with real API calls
   - Added `useWebSDRStore` hook for state management
   - Implemented automatic data loading on component mount
   - Added periodic health check refresh (every 30 seconds)
   - Added manual refresh button with loading indicator
   - Implemented error handling with alert messages
   - Added loading state UI when fetching data
   - Updated status display to handle 'online', 'offline', and 'unknown' states
   - Display "N/A" for uptime and avgSnr when data is not yet available

2. **Alert Component**
   - Created new `src/components/ui/alert.tsx` component for error messages
   - Follows the existing UI component pattern

### Backend Integration
The page now integrates with these API endpoints:
- `GET /api/v1/acquisition/websdrs` - Fetch list of configured WebSDRs
- `GET /api/v1/acquisition/websdrs/health` - Check health status of all WebSDRs

## Testing Instructions

### Prerequisites
1. Ensure all backend services are running:
   ```bash
   cd /home/runner/work/heimdall/heimdall
   make dev-up
   ```

2. Wait for services to be healthy:
   ```bash
   docker compose ps
   ```

3. Start the frontend development server:
   ```bash
   cd frontend
   npm run dev
   ```

### Test Cases

#### Test 1: Initial Page Load
1. Navigate to `http://localhost:3001/login`
2. Login with credentials (admin/admin)
3. Navigate to `http://localhost:3001/websdrs`
4. **Expected Result:**
   - Page shows a loading spinner initially
   - After loading, displays 7 WebSDR receivers from Northwestern Italy
   - Each receiver shows:
     - Name (e.g., "Aquila di Giaveno", "Montanaro", etc.)
     - URL
     - Location name
     - GPS coordinates
     - Status (online/offline/unknown)
     - Last check timestamp
     - Uptime (N/A initially)
     - Avg SNR (N/A initially)

#### Test 2: Health Status Display
1. Wait for initial health check to complete
2. **Expected Result:**
   - Green dot and "Online" for WebSDRs that respond successfully
   - Red dot and "Offline" for WebSDRs that fail health check
   - Yellow dot and "Unknown" for WebSDRs with pending health checks

#### Test 3: Status Summary Cards
1. Check the three summary cards at the top
2. **Expected Result:**
   - "Online Receivers" shows count like "6/7"
   - "Average Uptime" shows percentage (N/A initially)
   - "Network Status" shows "Healthy"

#### Test 4: Manual Refresh
1. Click the "Refresh" button in the top-right
2. **Expected Result:**
   - Button shows spinning icon and is disabled
   - Data is refreshed
   - Health status is updated
   - Button returns to normal state

#### Test 5: Auto-Refresh
1. Wait 30 seconds after page load
2. **Expected Result:**
   - Health status is automatically refreshed
   - Last check timestamp is updated
   - No page reload or interruption

#### Test 6: Error Handling
1. Stop the rf-acquisition service:
   ```bash
   docker compose stop rf-acquisition
   ```
2. Click "Refresh" button
3. **Expected Result:**
   - Red alert banner appears at the top
   - Error message explains the failure
   - Page remains functional
4. Restart the service:
   ```bash
   docker compose start rf-acquisition
   ```

#### Test 7: Network Diagnostics
1. Click "Test All Connections" button
2. **Expected Result:**
   - Same as manual refresh
   - All health checks are re-run

## Verification Checklist

- [ ] Page loads without console errors
- [ ] WebSDR data is fetched from backend (check Network tab)
- [ ] Health checks execute periodically
- [ ] Loading states display correctly
- [ ] Error states display correctly
- [ ] Status indicators (online/offline/unknown) work correctly
- [ ] Manual refresh works
- [ ] Auto-refresh works (30s interval)
- [ ] All 7 Italian WebSDRs are displayed
- [ ] GPS coordinates are accurate
- [ ] Table is sortable and responsive
- [ ] Mobile view works correctly

## Backend API Response Examples

### GET /api/v1/acquisition/websdrs
```json
[
  {
    "id": 1,
    "name": "Aquila di Giaveno",
    "url": "http://sdr1.ik1jns.it:8076/",
    "location_name": "Giaveno, Italy",
    "latitude": 45.02,
    "longitude": 7.29,
    "is_active": true,
    "timeout_seconds": 30,
    "retry_count": 3
  },
  ...
]
```

### GET /api/v1/acquisition/websdrs/health
```json
{
  "1": {
    "websdr_id": 1,
    "name": "Aquila di Giaveno",
    "status": "online",
    "last_check": "2025-10-22T17:30:00Z"
  },
  ...
}
```

## Known Issues / Limitations

1. **Uptime and Avg SNR**: Currently showing "N/A" because these metrics require historical data from the database. These will be populated once measurements are recorded.

2. **Health Check Timeout**: Health checks may take 30-60 seconds depending on WebSDR response times.

3. **Auto-Refresh Interval**: Set to 30 seconds. Can be adjusted in the useEffect hook if needed.

## Future Enhancements

- Add filtering and sorting to the table
- Add WebSDR edit functionality (currently only displays)
- Show historical uptime trends
- Add real-time SNR graphs
- Implement WebSocket for instant status updates
- Add ability to enable/disable specific WebSDRs
- Show more detailed connection diagnostics

## Troubleshooting

### Issue: "Cannot fetch WebSDRs" error
- **Solution**: Ensure rf-acquisition service is running and accessible
- Check: `curl http://localhost:8000/api/v1/acquisition/websdrs`

### Issue: All WebSDRs show "Unknown" status
- **Solution**: Health check may be in progress or Celery worker is not running
- Check: `docker compose logs rf-acquisition`

### Issue: Frontend cannot connect to backend
- **Solution**: Check API gateway is running on port 8000
- Check: `curl http://localhost:8000/health`

### Issue: Page shows loading spinner forever
- **Solution**: Check browser console for errors
- Check: Network tab in DevTools for failed requests
- Verify CORS settings in api-gateway

## Summary

The WebSDR management page now displays real-time data from the backend, with automatic health monitoring and manual refresh capabilities. The UI gracefully handles loading and error states, providing a professional user experience for monitoring the WebSDR network.
