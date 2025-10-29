# Sources Management Feature

## Overview
Complete CRUD (Create, Read, Update, Delete) system for managing radio frequency sources with interactive map visualization.

## Features Implemented

### Backend
1. **Database Schema**
   - Added `error_margin_meters` field to `known_sources` table
   - Migration file: `db/migrations/02-add-error-margin-to-known-sources.sql`
   - Default value: 50.0 meters
   - Validation: Must be greater than 0

2. **API Endpoints** (`/api/v1/sessions/known-sources`)
   - `GET /api/v1/sessions/known-sources` - List all sources
   - `GET /api/v1/sessions/known-sources/{source_id}` - Get specific source
   - `POST /api/v1/sessions/known-sources` - Create new source
   - `PUT /api/v1/sessions/known-sources/{source_id}` - Update source
   - `DELETE /api/v1/sessions/known-sources/{source_id}` - Delete source

3. **Data Models** (`services/data-ingestion-web/src/models/session.py`)
   - `KnownSource` - Complete source data model
   - `KnownSourceCreate` - Creation schema with validation
   - `KnownSourceUpdate` - Partial update schema

4. **Validation & Safety**
   - Prevents deletion of sources referenced by recording sessions
   - Returns 409 Conflict with count of referencing sessions
   - Validates coordinates (lat: -90 to 90, lon: -180 to 180)
   - Validates frequency (must be positive)
   - Validates error margin (must be greater than 0)
   - Prevents duplicate source names (unique constraint)

### Frontend
1. **Sources Management Page** (`frontend/src/pages/SourcesManagement.tsx`)
   - Full CRUD interface
   - Interactive Mapbox map visualization
   - Error margin circles on map
   - Drag-and-drop marker positioning
   - Click-to-place new sources
   - Real-time updates

2. **Features**
   - **Map Integration**:
     - Mapbox dark theme
     - Interactive markers (green = validated, orange = unvalidated)
     - Draggable markers update source location
     - Click map to set coordinates for new sources
     - Visual error margin circles around each source
     - Popups showing source details
     - Fly-to animation when selecting sources

   - **Form Validation**:
     - Required fields: name, frequency, latitude, longitude
     - Coordinate range validation
     - Positive error margin requirement
     - Real-time error display

   - **User Experience**:
     - Loading states during API calls
     - Success/error notifications (auto-dismiss after 5s)
     - Confirmation dialog for destructive actions
     - Empty state when no sources exist
     - Source count badge
     - Responsive design

3. **Navigation**
   - Added to main navigation menu under "RF Operations"
   - Route: `/sources`
   - Icon: broadcast icon

### Recording Session Fix
1. **API Contract Alignment**
   - Updated `RecordingSessionCreate` to include `known_source_id`
   - Fixed frequency conversion (MHz to Hz)
   - Proper field mapping: `frequency_hz` instead of `frequency_mhz`

2. **Component Updates** (`frontend/src/pages/RecordingSession.tsx`)
   - Correctly passes `known_source_id` to backend
   - Validates source selection before submission
   - Proper frequency unit conversion

## Database Migration

To apply the database migration:

```bash
# Option 1: Through docker-compose (when services are running)
docker-compose exec postgres psql -U heimdall_user -d heimdall -f /docker-entrypoint-initdb.d/migrations/02-add-error-margin-to-known-sources.sql

# Option 2: Directly via psql
psql -h localhost -U heimdall_user -d heimdall -f db/migrations/02-add-error-margin-to-known-sources.sql
```

## Usage Guide

### Creating a Source
1. Navigate to "Sources Management" from the sidebar
2. Click "Add Source" button
3. Fill in the form:
   - **Name**: Unique identifier for the source
   - **Description**: Optional notes
   - **Frequency (Hz)**: e.g., 144800000 for 144.8 MHz
   - **Latitude/Longitude**: Click map or enter manually
   - **Error Margin (meters)**: Uncertainty radius (default: 50m)
   - **Power (dBm)**: Optional transmission power
   - **Source Type**: beacon, repeater, station, or other
   - **Validated**: Check if source is confirmed
4. Click "Create"

### Updating a Source
1. Click "Edit" button on a source in the list
2. Modify fields
3. Click "Update"

Or:
- Drag marker on map to update location
- Auto-saves on drag end

### Deleting a Source
1. Click "Delete" button on a source
2. Confirm deletion
3. Note: Cannot delete sources referenced by recording sessions

### Using Sources in Recording Sessions
1. Navigate to "Recording Session"
2. Select source from "Known Source" dropdown
3. Source details populate automatically
4. Proceed with recording

## API Examples

### Create Source
```bash
curl -X POST http://localhost:8000/api/v1/sessions/known-sources \
  -H "Content-Type: application/json" \
  -d '{
    "name": "VK3RGL Beacon",
    "description": "2m beacon in Rome",
    "frequency_hz": 144800000,
    "latitude": 41.9028,
    "longitude": 12.4964,
    "power_dbm": 10.0,
    "source_type": "beacon",
    "is_validated": true,
    "error_margin_meters": 25.0
  }'
```

### Update Source
```bash
curl -X PUT http://localhost:8000/api/v1/sessions/known-sources/{source_id} \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 41.9050,
    "longitude": 12.5000,
    "error_margin_meters": 30.0
  }'
```

### Delete Source
```bash
curl -X DELETE http://localhost:8000/api/v1/sessions/known-sources/{source_id}
```

### List Sources
```bash
curl http://localhost:8000/api/v1/sessions/known-sources
```

## Edge Cases Handled

1. **Duplicate Names**: Returns 400 Bad Request
2. **Invalid Coordinates**: Returns 422 Unprocessable Entity
3. **Negative/Zero Error Margin**: Returns 422 Unprocessable Entity
4. **Source in Use**: Returns 409 Conflict when attempting to delete
5. **Network Errors**: Displays user-friendly error message
6. **Empty State**: Shows helpful empty state message
7. **Concurrent Updates**: Last write wins (optimistic concurrency)

## Testing

### Manual Testing Checklist
- [ ] Create source via form
- [ ] Create source by clicking map
- [ ] Update source by editing form
- [ ] Update source by dragging marker
- [ ] Delete unused source
- [ ] Attempt to delete source in use (should fail)
- [ ] Create source with duplicate name (should fail)
- [ ] Create source with invalid coordinates (should fail)
- [ ] Create source with negative error margin (should fail)
- [ ] View error margin circles on map
- [ ] Use source in recording session
- [ ] Verify source data persists after page reload

### Backend Tests
```bash
# Run data-ingestion-web tests
cd services/data-ingestion-web
pytest tests/ -v
```

### Frontend Tests
```bash
# Run frontend tests
cd frontend
npm run test

# Type checking
npm run type-check

# Linting
npm run lint
```

## Configuration

### Mapbox Token
Set in `.env` or `.env.local`:
```
VITE_MAPBOX_TOKEN=your_mapbox_token_here
```

Without a token, the map will not display. Get a free token at https://www.mapbox.com/

## Files Modified/Created

### Backend
- `db/migrations/02-add-error-margin-to-known-sources.sql` (new)
- `services/data-ingestion-web/src/models/session.py` (modified)
- `services/data-ingestion-web/src/routers/sessions.py` (modified)

### Frontend
- `frontend/src/pages/SourcesManagement.tsx` (new)
- `frontend/src/pages/RecordingSession.tsx` (modified)
- `frontend/src/services/api/session.ts` (modified)
- `frontend/src/store/sessionStore.ts` (modified)
- `frontend/src/App.tsx` (modified)
- `frontend/src/components/layout/DattaLayout.tsx` (modified)

### API Gateway
- No changes needed (already proxies `/api/v1/sessions/*` routes)

## Future Enhancements

1. **Import/Export**: Bulk import sources from CSV/JSON
2. **Source Categories**: Group sources by type or region
3. **Source History**: Track changes to source properties
4. **Validation Workflow**: Multi-step validation process
5. **Coverage Map**: Heatmap showing receiver coverage
6. **Source Search**: Filter and search sources
7. **Batch Operations**: Multi-select for bulk actions
8. **Source Templates**: Pre-defined source configurations

## Known Limitations

1. Requires Mapbox token for map functionality
2. Error circles are approximate (zoom-dependent calculation)
3. No undo functionality for deletions
4. No source versioning or audit trail
5. Cannot merge duplicate sources
6. No geofencing or boundary restrictions

## Support

For issues or questions:
- Check the FAQ: `docs/FAQ.md`
- Review Architecture: `docs/ARCHITECTURE.md`
- Contact: alessio.corsi@gmail.com
