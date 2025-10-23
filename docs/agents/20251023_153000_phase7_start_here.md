# 🎯 PHASE 7 START GUIDE - Frontend Development

**Previous Phase**: ✅ Phase 6 (Inference Service) - COMPLETE  
**Current Phase**: 🚀 Phase 7 (Frontend) - READY TO START  
**Date**: 2025-10-24  

---

## 📋 QUICK START

### What You Have (From Phase 6)

✅ **Backend API ready**:
- `POST /api/v1/inference/predict` - Single prediction with <500ms SLA
- `POST /api/v1/inference/predict/batch` - Batch processing (1-100 samples)
- `GET /api/v1/inference/health` - Service health status
- `GET /model/info` - Model metadata and status
- `GET /model/performance` - Performance metrics
- `GET /api/v1/inference/predictions/{task_id}` - Async result polling

✅ **Data Format Specifications**:

**Single Prediction Request**:
```json
{
  "iq_data": [[1.0, 0.5], [1.1, 0.4], ...],
  "cache_enabled": true,
  "session_id": "sess-2025-10-24-001"
}
```

**Prediction Response**:
```json
{
  "position": {"lat": 45.123, "lon": 8.456},
  "uncertainty": {"sigma_x": 25.5, "sigma_y": 30.2, "theta": 45.0},
  "confidence": 0.95,
  "model_version": "v1",
  "inference_time_ms": 145.3,
  "cache_hit": false,
  "timestamp": "2025-10-24T15:30:00Z"
}
```

**Batch Request**:
```json
{
  "iq_samples": [
    {"sample_id": "s1", "iq_data": [[1.0, 0.5], ...]},
    {"sample_id": "s2", "iq_data": [[0.9, 0.6], ...]}
  ],
  "cache_enabled": true,
  "session_id": "batch-sess-001"
}
```

**Batch Response**:
```json
{
  "session_id": "batch-sess-001",
  "total_samples": 2,
  "successful": 2,
  "failed": 0,
  "success_rate": 1.0,
  "predictions": [...],
  "total_time_ms": 250.5,
  "samples_per_second": 7.98,
  "average_latency_ms": 125.25,
  "p95_latency_ms": 145.3,
  "cache_hit_rate": 0.5
}
```

---

## 🎨 FRONTEND TASKS (T7.1-T7.10)

### T7.1: React + TypeScript + Vite Setup
**Objective**: Create frontend project structure  
**Requirements**:
- React 18+ with TypeScript
- Vite for fast development
- ESLint + Prettier configured
- Environment variables for API endpoint

**Deliverables**:
```
frontend/
├── src/
│   ├── App.tsx
│   ├── main.tsx
│   ├── components/
│   ├── pages/
│   ├── services/
│   └── styles/
├── index.html
├── vite.config.ts
├── tsconfig.json
└── package.json
```

### T7.2: Mapbox Integration
**Objective**: Interactive map with WebSDR locations  
**Requirements**:
- Mapbox GL JS for interactive mapping
- 7 WebSDR receiver markers (from WEBSDRS.md)
- Map controls (zoom, pan, fullscreen)
- Mark WebSDR online/offline status

**Deliverables**:
- `components/Map.tsx` - Map component
- `services/websdrs.ts` - WebSDR data loader
- Pin icons for each receiver state

### T7.3: WebSDR Status Dashboard
**Objective**: Display WebSDR receiver status  
**Requirements**:
- List all 7 receivers with status
- Real-time health polling (every 5 seconds)
- Signal strength indicators
- Frequency display

**Deliverables**:
- `components/WebSDRStatus.tsx` - Status component
- `services/api.ts` - API communication layer
- Health check visualization

### T7.4: Real-time Localization Display
**Objective**: Show predicted location with uncertainty ellipse  
**Requirements**:
- Map marker for predicted location (lat, lon)
- Uncertainty ellipse overlay (sigma_x, sigma_y, theta)
- Color code confidence (green=high, yellow=medium, red=low)
- Update in real-time as new predictions arrive

**Deliverables**:
- `components/Localization.tsx` - Localization display
- `components/UncertaintyEllipse.tsx` - Ellipse visualization
- SVG or GeoJSON rendering

### T7.5: Recording Session Manager
**Objective**: UI for managing RF recording sessions  
**Requirements**:
- Start/stop recording button
- Recording duration counter
- Selected frequency input (MHz)
- Trigger RF acquisition endpoint (Phase 3)
- Display recording status

**Deliverables**:
- `pages/Recording.tsx` - Recording interface
- Session state management (Zustand/Context)
- API integration for acquisition

### T7.6: Spectrogram Visualization
**Objective**: Display signal spectrogram for validation  
**Requirements**:
- Fetches IQ data from recorded session
- Computes spectrogram (Welch or STFT)
- Canvas or SVG rendering
- Frequency/time axes labeled
- Colormap (Viridis recommended)

**Deliverables**:
- `components/Spectrogram.tsx` - Spectrogram display
- `services/spectrogram.ts` - Computation logic
- Interactive hover for value inspection

### T7.7: User Authentication
**Objective**: Operator authentication and role management  
**Requirements**:
- Login form with username/password
- JWT token storage (localStorage)
- Role-based access (Operator, Admin, Viewer)
- Logout functionality
- Protected routes

**Deliverables**:
- `pages/Login.tsx` - Login form
- `services/auth.ts` - Auth service
- `hooks/useAuth.ts` - Auth hook
- Route guards

### T7.8: Responsive Design
**Objective**: Mobile/tablet support for field operators  
**Requirements**:
- Mobile-first CSS
- Tailwind CSS or Material-UI
- Touch-friendly buttons/inputs
- Responsive map sizing
- Breakpoints: 320px, 768px, 1024px

**Deliverables**:
- Global styles with responsive breakpoints
- Mobile layout variants
- Touch event handlers where needed

### T7.9: WebSocket Real-time Updates
**Objective**: Live updates without polling  
**Requirements**:
- WebSocket connection to backend
- Subscribe to localization updates
- Automatic map refresh on new predictions
- Graceful reconnection on disconnect
- Message queuing if offline

**Deliverables**:
- `services/websocket.ts` - WebSocket client
- `hooks/useWebSocket.ts` - WebSocket hook
- Connection state management

### T7.10: End-to-End Tests
**Objective**: Playwright tests for user workflows  
**Requirements**:
- Login workflow
- Recording session lifecycle
- Prediction polling
- Map interactions
- Error scenarios

**Deliverables**:
- `e2e/tests/` directory
- `playwright.config.ts`
- GitHub Actions integration

---

## 🗺️ INTEGRATION CHECKLIST

**Before Starting Frontend**:
- [ ] Phase 6 backend deployed and running
- [ ] API endpoints accessible at `http://localhost:8003/` (or configured URL)
- [ ] Redis cache running (Port 6379)
- [ ] ONNX model loaded in inference service
- [ ] Database migrations complete

**Frontend Setup**:
- [ ] Create `frontend/` directory in project root
- [ ] Initialize React + Vite project
- [ ] Configure environment variables:
  ```env
  VITE_API_URL=http://localhost:8003
  VITE_MAPBOX_TOKEN=your-mapbox-token
  VITE_WS_URL=ws://localhost:8003/ws
  ```
- [ ] Install dependencies: `npm install`
- [ ] Start dev server: `npm run dev`

---

## 📊 API ENDPOINTS READY FOR INTEGRATION

### Prediction Endpoints
```
POST /api/v1/inference/predict
  Request: { iq_data, cache_enabled, session_id }
  Response: { position, uncertainty, confidence, ... }
  Status: 200 OK or 400/500 error

POST /api/v1/inference/predict/batch
  Request: { iq_samples[], cache_enabled, session_id }
  Response: { predictions[], total_time_ms, samples_per_second, ... }
  Status: 200 OK or 400/500 error

GET /api/v1/inference/health
  Response: { status: "healthy", timestamp, ... }
  Status: 200 OK or 503 SERVICE_UNAVAILABLE
```

### Model Management Endpoints
```
GET /model/info
  Response: { active_version, stage, accuracy, latency_p95_ms, ... }
  Status: 200 OK or 503 SERVICE_UNAVAILABLE

GET /model/performance
  Response: { inference_latency_ms, cache_hit_rate, throughput, ... }
  Status: 200 OK or 503 SERVICE_UNAVAILABLE

POST /model/reload
  Request: { version_id, stage, force }
  Response: { success, message, reload_time_ms, ... }
  Status: 200 OK or 503 SERVICE_UNAVAILABLE
```

### Recording Session Endpoints (Phase 3)
```
POST /api/v1/acquisition/acquire
  Request: { frequency_mhz, duration_seconds }
  Response: { task_id, session_id, status, ... }
  Status: 202 ACCEPTED

GET /api/v1/acquisition/status/{task_id}
  Response: { task_id, status, progress, result, ... }
  Status: 200 OK or 404 NOT_FOUND
```

---

## 🛠️ RECOMMENDED TECH STACK

### Frontend Framework
- **React 18+** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool (fast HMR)
- **Zustand** or **TanStack Query** - State management

### UI Components
- **Tailwind CSS** - Styling (mobile-first)
- **Radix UI** - Accessible components
- **Lucide Icons** - Icon library
- **ShadcN/ui** - Pre-built components

### Mapping
- **Mapbox GL JS** - Interactive maps
- **GeoJSON** - Geographic data format

### Real-time
- **Socket.io** or **ws** - WebSocket client
- **React-use** - Hooks library

### Testing
- **Playwright** - E2E testing
- **Vitest** - Unit testing
- **Testing Library** - Component testing

### Development
- **ESLint** - Code linting
- **Prettier** - Code formatting
- **Husky** - Git hooks

---

## 📐 WIREFRAME / USER FLOWS

### Main Dashboard
```
┌─────────────────────────────────┐
│ HEIMDALL RF LOCALIZATION        │ Header
├─────────────────────────────────┤
│ [Recording] [History] [Settings]│ Top Nav
├──────────────┬──────────────────┤
│              │                  │
│   MAP        │   WebSDR Status  │ Main Content
│   (7 pins)   │   (7 receivers)  │
│   + ellipse  │   + signals      │
│              │                  │
├──────────────┴──────────────────┤
│ Localization: 45.123°N 8.456°E  │ Bottom Status
│ Confidence: 95% | Cache: 82%    │
└─────────────────────────────────┘
```

### Recording Session
```
┌─────────────────────────────┐
│ RF RECORDING SESSION        │
├─────────────────────────────┤
│ Frequency: [147.5] MHz      │
│ Duration: [00:05:32]        │
│ Status: Recording...        │
│                             │
│ [Start] [Stop] [Analyze]    │
│                             │
│ Spectrogram:                │
│ ┌───────────────────────┐   │
│ │ Frequency Spectrum    │   │
│ │                       │   │
│ └───────────────────────┘   │
└─────────────────────────────┘
```

---

## 🚀 DEPLOYMENT PATHS

### Development
```bash
cd frontend
npm install
npm run dev
# Server at http://localhost:5173
```

### Production Build
```bash
npm run build
npm run preview
# Outputs to dist/
```

### Docker Deployment
```dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json .
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

---

## ⚠️ CRITICAL INTEGRATION NOTES

### API Error Handling
- All endpoints may return 503 SERVICE_UNAVAILABLE (model loading, cache failure)
- Implement retry logic with exponential backoff
- Display user-friendly error messages

### Real-time Considerations
- WebSocket may disconnect due to network issues
- Implement automatic reconnection with exponential backoff
- Queue events if offline
- Sync on reconnect

### Performance
- Memoize expensive computations (spectrogram)
- Lazy load components
- Debounce map interactions
- Use React.memo for static components

### Browser Compatibility
- Chrome/Edge: Full support
- Firefox: Full support
- Safari: Test WebSocket, Maps
- Mobile: Test touch events, responsive layout

---

## 📚 REFERENCE LINKS

- **API Documentation**: [PHASE6_COMPLETE_FINAL.md](PHASE6_COMPLETE_FINAL.md)
- **Backend Code**: `services/inference/src/routers/`
- **Data Formats**: `services/inference/src/utils/batch_predictor.py`
- **WebSDR Config**: `WEBSDRS.md`
- **Architecture**: `docs/architecture_diagrams.md`

---

## ✅ PHASE 7 SUCCESS CRITERIA

- [ ] All UI components functional and responsive
- [ ] Real-time map updates with <1s latency
- [ ] Batch prediction UI for multiple samples
- [ ] Error scenarios handled gracefully
- [ ] E2E tests covering main workflows
- [ ] Mobile layout tested on actual devices
- [ ] Performance: <500ms for all interactions
- [ ] Accessibility: WCAG 2.1 Level AA compliance

---

## 🎊 YOU ARE READY TO START PHASE 7!

All Phase 6 components are production-ready and waiting for frontend integration.

**Next**: Create React project and start T7.1 ✨

---

**Generated**: 2025-10-24  
**For**: Phase 7 Frontend Developer  
**From**: GitHub Copilot (Agent-Backend)  
**Project**: Heimdall SDR Radio Source Localization
