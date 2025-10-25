# Session Management UI - Visual Documentation

## Component Overview

The Session Management Interface consists of three main components:

### 1. SessionListEnhanced
**Purpose**: Display and filter recording sessions with approval workflow

**Key Features**:
- ✅ Dual filter system (Status + Approval)
- ✅ Pagination controls
- ✅ Session cards with metadata preview
- ✅ Click to open detail modal
- ✅ Auto-refresh capability
- ✅ Responsive grid layout

**Visual Elements**:
```
┌─────────────────────────────────────────────────────────────┐
│ Recording Sessions                          [Refresh Button] │
├─────────────────────────────────────────────────────────────┤
│ Filter: Status: [All] [Completed] [Pending] [Failed]        │
│         Approval: [All] [Pending] [Approved] [Rejected]     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Session Name           [COMPLETED] [PENDING]       │     │
│  │ Source: Test Source     Frequency: 145.000 MHz     │     │
│  │ Duration: 2m 0s        Created: 2025-01-01         │     │
│  │ Measurements: 42                            [👁️]   │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Another Session        [FAILED] [REJECTED]         │     │
│  │ Source: Source 2       Frequency: 432.100 MHz      │     │
│  │ Duration: 5m 30s       Created: 2025-01-02         │     │
│  │ Measurements: 0                             [👁️]   │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  Page 1 of 5 (42 total)       [◄ Previous] [Next ►]        │
└─────────────────────────────────────────────────────────────┘
```

**Badge Colors**:
- Status badges:
  - 🟡 PENDING (yellow)
  - 🔵 PROCESSING (blue)
  - 🟢 COMPLETED (green)
  - 🔴 FAILED (red)

- Approval badges:
  - 🟡 PENDING (yellow)
  - 🟢 APPROVED (green)
  - 🔴 REJECTED (red)

### 2. SessionDetailModal
**Purpose**: Display full session details with approval actions

**Key Features**:
- ✅ Complete session metadata
- ✅ Known source information
- ✅ Grid of 7 spectrogram placeholders
- ✅ SNR metrics with color coding
- ✅ Approve/Reject buttons (for pending sessions)
- ✅ Confirmation dialog for rejection

**Visual Layout**:
```
┌───────────────────────────────────────────────────────────────┐
│ Session Name               [PENDING APPROVAL]            [×]  │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  Session Information          │  Status                       │
│  ├─ Created: 2025-01-01      │  ├─ Processing: Completed     │
│  ├─ Duration: 2m 0s          │  ├─ Approval: Pending         │
│  ├─ Frequency: 145.000 MHz   │  └─ Task ID: task-123         │
│  └─ Measurements: 42          │                               │
│                                                                │
│  Known Source                                                 │
│  ├─ Name: Test Source                                         │
│  └─ Location: 45.000000, 7.500000                            │
│                                                                │
│  WebSDR Spectrograms (7 Receivers)                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                     │
│  │ Torino   │ │ Genova   │ │Alessand. │                     │
│  │SNR:25.3dB│ │SNR:18.7dB│ │SNR:22.1dB│                     │
│  │  [Good]  │ │[Marginal]│ │  [Good]  │                     │
│  └──────────┘ └──────────┘ └──────────┘                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │  Cuneo   │ │   Asti   │ │  Savona  │ │ Imperia  │       │
│  │SNR:15.4dB│ │SNR:19.8dB│ │SNR:12.3dB│ │SNR:16.5dB│       │
│  │[Marginal]│ │[Marginal]│ │  [Poor]  │ │[Marginal]│       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│                                                                │
├───────────────────────────────────────────────────────────────┤
│                        [Reject] [Approve for Training]        │
└───────────────────────────────────────────────────────────────┘
```

**SNR Color Coding**:
- 🟢 Green: SNR > 20 dB (Good quality)
- 🟡 Yellow: SNR 10-20 dB (Marginal quality)
- 🔴 Red: SNR < 10 dB (Poor quality)

### 3. SpectrogramViewer
**Purpose**: Display spectrogram visualization (placeholder for Phase 7)

**Current Implementation**:
```
┌────────────────────────────────────────┐
│ Torino Nord              SNR: 25.3 dB  │
│                         (Good)         │
│ [🔍+] [🔍-] [⛶]                       │
├────────────────────────────────────────┤
│                                        │
│           [Grid Pattern]               │
│                                        │
│      Spectrogram Preview               │
│   Session 1 - WebSDR 1                │
│                                        │
│  (Full implementation in Phase 8)      │
│                                        │
│ Frequency ↑                            │
│                         Time →         │
└────────────────────────────────────────┘
```

**Phase 8 Enhancements** (Deferred):
- Real spectrogram image from backend
- Zoom/pan functionality
- Frequency and time axis labels with values
- Signal region highlighting

## Approval Workflow

### Approve Flow:
1. User clicks session card → Opens detail modal
2. Reviews session metadata and spectrograms
3. Clicks "Approve for Training" button
4. Session approval_status → 'approved'
5. Modal closes, list refreshes
6. Session badge changes from 🟡 PENDING to 🟢 APPROVED

### Reject Flow:
1. User clicks session card → Opens detail modal
2. Reviews session metadata and spectrograms
3. Clicks "Reject" button
4. Confirmation dialog appears:
   ```
   ┌────────────────────────────────────┐
   │ Reject Session                     │
   ├────────────────────────────────────┤
   │ Are you sure you want to reject    │
   │ this session? It will not be used  │
   │ for training.                      │
   │                                    │
   │ Reason (optional):                 │
   │ ┌────────────────────────────────┐ │
   │ │ [Text area for comment]        │ │
   │ └────────────────────────────────┘ │
   │                                    │
   │     [Cancel] [Confirm Rejection]   │
   └────────────────────────────────────┘
   ```
5. User confirms rejection
6. Session approval_status → 'rejected'
7. Modal closes, list refreshes
8. Session badge changes from 🟡 PENDING to 🔴 REJECTED

## Usage Example

```tsx
import { SessionListEnhanced } from '@/components';

// In a page component
export default function DataIngestionPage() {
    return (
        <div>
            <h1>Session Management</h1>
            <SessionListEnhanced autoRefresh={true} />
        </div>
    );
}
```

## Integration Points

### Backend Endpoints Used:
- `GET /api/v1/sessions` - List sessions with pagination and filters
- `GET /api/v1/sessions/{id}` - Get session details
- `PATCH /api/v1/sessions/{id}/approval` - Update approval status

### State Management:
- Uses existing `sessionStore` (Zustand)
- Filters managed in store
- Pagination state in store
- Current session in store

### API Service:
- Uses existing `sessionService` from `@/services/api/session`
- No new API methods needed
- All endpoints already implemented

## Testing Coverage

**SessionDetailModal.test.tsx** (8 tests):
- ✅ Modal visibility control
- ✅ Session details rendering
- ✅ Metadata display
- ✅ Approval buttons for pending sessions
- ✅ Spectrogram grid display
- ✅ Close button functionality
- ✅ Reject confirmation dialog
- ✅ Approval action handling

**Integration with existing tests**:
- All 291 tests passing
- No regressions introduced
- Existing components unaffected

## Performance Characteristics

- **Auto-refresh**: 10 seconds (configurable)
- **Pagination**: 20 items per page (backend controlled)
- **Modal load time**: ~100ms (session fetch)
- **Filter application**: Instant (triggers backend query)

## Accessibility

- Keyboard navigation supported
- ARIA labels on interactive elements
- Focus management in modals
- Screen reader friendly badge text
- Color coding with text labels (not color-only)

## Browser Compatibility

- Modern browsers (Chrome, Firefox, Safari, Edge)
- React 19+ required
- Tailwind CSS for styling
- Lucide React for icons
- Radix UI for base components
