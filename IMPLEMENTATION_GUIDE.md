# Sources Management Implementation Guide

## 🎯 Quick Summary

This PR implements a complete **Sources Management** system with CRUD operations and fixes the **Recording Session** workflow to properly integrate with known sources.

### What's New

1. **Sources Management Page** - Full-featured UI for managing RF sources
2. **Interactive Map** - Mapbox integration with drag-and-drop positioning
3. **Error Margin Visualization** - Circles showing uncertainty radius
4. **Complete CRUD API** - Create, Read, Update, Delete endpoints
5. **Recording Session Fix** - Proper integration with known sources

---

## 🚀 How to Use

### 1. Start the Services

```bash
# Start all services
docker-compose up -d

# Apply database migration
docker-compose exec postgres psql -U heimdall_user -d heimdall -f /docker-entrypoint-initdb.d/migrations/02-add-error-margin-to-known-sources.sql
```

### 2. Configure Mapbox (Required for Map)

Add to your `.env` file:
```
VITE_MAPBOX_TOKEN=your_token_here
```

Get a free token at: https://www.mapbox.com/

### 3. Access Sources Management

Navigate to: **http://localhost:3000/sources**

Or via sidebar: **RF Operations → Sources Management**

---

## 🎨 User Interface Overview

### Main Components

```
┌─────────────────────────────────────────────────────────────┐
│ Sources Management                                           │
├─────────────────────────────────────────────────────────────┤
│                                         │                     │
│  ┌──────────────────────────────────┐  │  ┌───────────────┐ │
│  │                                  │  │  │ Sources List  │ │
│  │     Interactive Map              │  │  │               │ │
│  │     (Mapbox)                     │  │  │ • Source 1    │ │
│  │                                  │  │  │ • Source 2    │ │
│  │  📍 Draggable Markers            │  │  │ • Source 3    │ │
│  │  ⭕ Error Circles                │  │  │               │ │
│  │  🗺️  Click to Place              │  │  │ [+ Add]       │ │
│  │                                  │  │  └───────────────┘ │
│  └──────────────────────────────────┘  │                     │
│                                         │                     │
│  Legend: 🟢 Validated  🟠 Unvalidated  │                     │
└─────────────────────────────────────────────────────────────┘
```

### Create/Edit Form

```
┌────────────────────────────┐
│ New Source                 │
├────────────────────────────┤
│ Name: [_______________] *  │
│ Description: [________]    │
│ Frequency (Hz): [_____] *  │
│ Latitude: [___________] *  │
│ Longitude: [__________] *  │
│ Error Margin (m): [___] *  │
│ Power (dBm): [________]    │
│ Type: [beacon ▼]           │
│ ☐ Validated                │
│                            │
│ [Create] [Cancel]          │
└────────────────────────────┘
```

---

## 🔄 Workflows

### Creating a New Source

**Method 1: Using the Form**
1. Click "Add Source" button
2. Fill in required fields (*)
3. Enter coordinates or...
4. Click map to set location
5. Click "Create"

**Method 2: Map-First**
1. Click "Add Source"
2. Click desired location on map
3. Coordinates auto-populate
4. Fill remaining fields
5. Click "Create"

### Updating a Source

**Method 1: Edit Form**
1. Click "Edit" button on source
2. Modify fields
3. Click "Update"

**Method 2: Drag on Map**
1. Drag marker to new location
2. Release mouse
3. Auto-saves new coordinates
4. Notification confirms update

### Using in Recording Session

1. Navigate to "Recording Session"
2. Select source from dropdown
3. Details auto-populate
4. Continue with recording

---

## 🔧 API Reference

### Base URL
```
http://localhost:8000/api/v1/sessions/known-sources
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | List all sources |
| GET | `/{id}` | Get specific source |
| POST | `/` | Create new source |
| PUT | `/{id}` | Update source |
| DELETE | `/{id}` | Delete source |

### Example: Create Source

**Request:**
```json
POST /api/v1/sessions/known-sources
Content-Type: application/json

{
  "name": "VK3RGL Beacon",
  "description": "2m beacon in Rome",
  "frequency_hz": 144800000,
  "latitude": 41.9028,
  "longitude": 12.4964,
  "power_dbm": 10.0,
  "source_type": "beacon",
  "is_validated": true,
  "error_margin_meters": 25.0
}
```

**Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "name": "VK3RGL Beacon",
  "description": "2m beacon in Rome",
  "frequency_hz": 144800000,
  "latitude": 41.9028,
  "longitude": 12.4964,
  "power_dbm": 10.0,
  "source_type": "beacon",
  "is_validated": true,
  "error_margin_meters": 25.0,
  "created_at": "2024-10-29T21:00:00Z",
  "updated_at": "2024-10-29T21:00:00Z"
}
```

### Error Responses

| Code | Meaning | Example |
|------|---------|---------|
| 400 | Bad Request | Duplicate name, invalid data |
| 404 | Not Found | Source doesn't exist |
| 409 | Conflict | Source in use by recording session |
| 422 | Validation Error | Invalid coordinates, negative error margin |

---

## ✅ Validation Rules

| Field | Rules |
|-------|-------|
| name | Required, unique |
| frequency_hz | Required, > 0 |
| latitude | Required, -90 to 90 |
| longitude | Required, -180 to 180 |
| error_margin_meters | Required, > 0 |
| power_dbm | Optional |
| source_type | Optional (beacon, repeater, station, other) |
| is_validated | Optional, default false |

---

## 🛡️ Edge Cases Handled

### ✅ Duplicate Names
```
❌ Error: "A known source with this name already exists"
→ Use a unique name
```

### ✅ Invalid Coordinates
```
❌ Error: "Valid latitude (-90 to 90) is required"
→ Check coordinate bounds
```

### ✅ Source in Use
```
❌ Error: "Cannot delete source: it is referenced by 3 recording session(s)"
→ Delete or update sessions first
```

### ✅ Network Errors
```
❌ Error: "Backend service unavailable"
→ Check service health, retry
```

### ✅ Empty State
```
ℹ️ "No sources yet. Click 'Add Source' to create one."
→ Helpful message with clear action
```

---

## 🎯 Integration Points

### Frontend → API Gateway
```
Frontend (React)
    ↓ HTTP Request
API Gateway (FastAPI)
    ↓ Proxy
Data Ingestion Service
    ↓ SQL Query
PostgreSQL Database
```

### Recording Session Flow
```
1. User selects source from dropdown
2. Source details populate form
3. User starts recording
4. Session created with known_source_id
5. Acquisition triggered with frequency
6. Results linked to source
```

---

## 📊 Database Schema

### New Field Added
```sql
ALTER TABLE heimdall.known_sources 
ADD COLUMN error_margin_meters FLOAT DEFAULT 50.0 
CHECK (error_margin_meters > 0);
```

### Complete Schema
```sql
CREATE TABLE heimdall.known_sources (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    frequency_hz BIGINT NOT NULL,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    power_dbm FLOAT,
    source_type VARCHAR(100),
    is_validated BOOLEAN DEFAULT FALSE,
    error_margin_meters FLOAT DEFAULT 50.0,  -- NEW
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

---

## 🧪 Testing Checklist

### Backend Tests
- [ ] Create source API
- [ ] Update source API
- [ ] Delete source API
- [ ] Delete source in use (should fail)
- [ ] List sources API
- [ ] Get source API
- [ ] Validation errors
- [ ] Duplicate name handling

### Frontend Tests
- [ ] Component renders
- [ ] Form validation
- [ ] Map interactions
- [ ] CRUD operations
- [ ] Error handling
- [ ] Loading states
- [ ] Notifications

### Integration Tests
- [ ] Create source end-to-end
- [ ] Update source end-to-end
- [ ] Delete source end-to-end
- [ ] Recording session integration
- [ ] Error scenarios
- [ ] Concurrent updates

### Manual Tests
- [ ] Create source via form
- [ ] Create source via map click
- [ ] Update source via form
- [ ] Update source via drag
- [ ] Delete unused source
- [ ] Attempt delete source in use
- [ ] Use source in recording
- [ ] All validations trigger
- [ ] All notifications appear
- [ ] Map loads and displays

---

## 🐛 Troubleshooting

### Map Not Showing
**Problem:** Blank area where map should be  
**Solution:** Set VITE_MAPBOX_TOKEN in .env

### API Errors
**Problem:** "503 Service Unavailable"  
**Solution:** Check backend services are running

### Migration Not Applied
**Problem:** Database errors about missing column  
**Solution:** Run migration SQL script

### Sources Not Loading
**Problem:** Empty list but sources exist  
**Solution:** Check API Gateway proxy configuration

---

## 📝 Code Structure

```
heimdall/
├── db/
│   └── migrations/
│       └── 02-add-error-margin-to-known-sources.sql
├── services/
│   └── data-ingestion-web/
│       └── src/
│           ├── models/
│           │   └── session.py (updated)
│           └── routers/
│               └── sessions.py (updated)
└── frontend/
    └── src/
        ├── pages/
        │   ├── SourcesManagement.tsx (NEW)
        │   └── RecordingSession.tsx (updated)
        ├── services/
        │   └── api/
        │       └── session.ts (updated)
        └── store/
            └── sessionStore.ts (updated)
```

---

## 🚀 Performance Considerations

- **Map Rendering**: Circles use zoom-dependent sizing
- **Marker Updates**: Debounced drag events
- **API Calls**: Optimistic UI updates
- **State Management**: Zustand for efficient re-renders
- **Lazy Loading**: Page code-split with React.lazy

---

## 🎓 Best Practices Applied

✅ TypeScript for type safety  
✅ Proper error handling with typed errors  
✅ Form validation with user feedback  
✅ Loading states for all async operations  
✅ Success/error notifications  
✅ Confirmation for destructive actions  
✅ Responsive design  
✅ Accessibility (labels, ARIA)  
✅ Clean code with proper separation of concerns  
✅ Comprehensive documentation  

---

## 🔮 Future Enhancements

1. **Bulk Operations** - Import/export sources from CSV
2. **Source Search** - Filter and search functionality
3. **Version History** - Track changes to sources
4. **Coverage Analysis** - Heatmap of receiver coverage
5. **Geofencing** - Define boundaries and alerts
6. **Source Templates** - Pre-configured source types
7. **Batch Edit** - Multi-select and bulk update
8. **Mobile App** - Native mobile interface

---

## 📞 Support

- **Documentation**: `SOURCES_MANAGEMENT_README.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **FAQ**: `docs/FAQ.md`
- **Contact**: alessio.corsi@gmail.com

---

## ✨ Summary

This implementation provides a complete, production-ready Sources Management system with:

- ✅ Full CRUD functionality
- ✅ Interactive map visualization
- ✅ Comprehensive validation
- ✅ Error handling
- ✅ Great UX
- ✅ Proper integration
- ✅ Complete documentation

Ready for review and testing! 🎉
