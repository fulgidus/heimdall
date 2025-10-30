# Recording Session UI Guide

## User Interface Overview

The Recording Session page has been redesigned with a 3-step wizard workflow that uses WebSocket for real-time updates.

## UI States and Workflow

### State 1: Idle (Initial State)

**What You See:**
```
┌─────────────────────────────────────────┐
│ Session Configuration                   │
├─────────────────────────────────────────┤
│ Step 1: Configure session and start    │
│ recording                               │
│                                         │
│ Session Name: [________________]        │
│ Frequency (MHz): [145.500]             │
│                                         │
│ Duration (seconds) - 15 samples        │
│ ├─────────●─────────────┤             │
│ 1s (1 sample) [15s = 15 samples] 30s  │
│ ℹ Each second is recorded as a         │
│   separate 1s sample                   │
│                                         │
│ Notes: [__________________________]    │
│        [__________________________]    │
│                                         │
│ [▶ Start Recording Session]            │
└─────────────────────────────────────────┘
```

**Action:** Click "Start Recording Session"

**What Happens:**
- WebSocket sends `session:start` command
- Backend creates session in database with "Unknown" source
- UI transitions to "Recording" state

---

### State 2: Recording (Awaiting Source Assignment)

**What You See:**
```
┌─────────────────────────────────────────┐
│ Session Configuration                   │
├─────────────────────────────────────────┤
│ [Session details from Step 1 - disabled]│
│                                         │
│ ⚠ Step 2: Assign a source (or select   │
│   "Unknown")                            │
│                                         │
│ Select Source:                          │
│ ┌───────────────────────────────────┐  │
│ │ Unknown Source                ▼  │  │
│ │─────────────────────────────────│  │
│ │ Unknown Source                  │  │
│ │ Beacon Station 1 - 145.500 MHz  │  │
│ │ Beacon Station 2 - 145.800 MHz  │  │
│ │ Repeater R0 - 145.000 MHz       │  │
│ └───────────────────────────────────┘  │
│                                         │
│ [✓ Assign Source]                      │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Session Status                          │
├─────────────────────────────────────────┤
│ ● Recording                             │
│ Recording session started. Now assign   │
│ a source.                               │
│                                         │
│ Session ID: a3f2e1d8...                │
└─────────────────────────────────────────┘
```

**Action:** Select a source and click "Assign Source"

**What Happens:**
- WebSocket sends `session:assign_source` command
- Backend updates session with selected source
- UI transitions to "Source Assigned" state

---

### State 3: Source Assigned (Ready for Acquisition)

**What You See:**
```
┌─────────────────────────────────────────┐
│ Session Configuration                   │
├─────────────────────────────────────────┤
│ [Session and source details - disabled] │
│                                         │
│ ✓ Step 3: Source assigned. Ready to    │
│   start acquisition!                    │
│                                         │
│ [🚀 Start Acquisition]                  │
│ [✗ Cancel]                              │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Session Status                          │
├─────────────────────────────────────────┤
│ ● Source Assigned                       │
│ Source assigned. Ready to start         │
│ acquisition.                            │
│                                         │
│ Session ID: a3f2e1d8...                │
└─────────────────────────────────────────┘
```

**Action:** Click "Start Acquisition"

**What Happens:**
- WebSocket sends `session:complete` command
- Backend starts `acquire_iq_chunked` Celery task
- UI transitions to "Acquiring" state
- Real-time progress updates begin

---

### State 4: Acquiring (In Progress with Real-time Updates)

**What You See:**
```
┌─────────────────────────────────────────┐
│ Session Configuration                   │
├─────────────────────────────────────────┤
│ [Session details - all disabled]        │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Session Status                          │
├─────────────────────────────────────────┤
│ ⟳ Acquiring Data                        │
│ Chunk 8/15 acquired                     │
│                                         │
│ Samples Acquired         8/15          │
│ ▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░  53%           │
│                                         │
│ Total Measurements      [56]            │
│                                         │
│ Session ID: a3f2e1d8...                │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ System Status                           │
├─────────────────────────────────────────┤
│ WebSDR Receivers         [7/7 Online]  │
│ Known Sources            [12]           │
│ System Ready             [Yes]          │
└─────────────────────────────────────────┘
```

**What Happens:**
- Every 1-2 seconds, a new chunk is acquired
- Progress bar animates with each update
- Measurements count increases (7 per chunk)
- WebSocket broadcasts `session:progress` events
- All connected clients see the same updates

---

### State 5: Complete (Acquisition Finished)

**What You See:**
```
┌─────────────────────────────────────────┐
│ Session Configuration                   │
├─────────────────────────────────────────┤
│ [Session details - all disabled]        │
│                                         │
│ [➕ New Session]                         │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Session Status                          │
├─────────────────────────────────────────┤
│ ✓ Complete                              │
│ Acquisition complete!                   │
│                                         │
│ ✓ Acquired 15 samples (105 total       │
│   measurements)                         │
│                                         │
│ Session ID: a3f2e1d8...                │
└─────────────────────────────────────────┘
```

**Action:** Click "New Session"

**What Happens:**
- Form resets to initial state
- Ready to start another recording session

---

## Key UI Elements

### Duration Slider (Step 1)

The slider has been updated to support 1-second granularity:

```
Old: 10s ────────●────────── 300s (10s steps)
     [Duration: 60s]

New: 1s ───────●──────── 30s (1s steps)
     [Duration: 15s = 15 samples]
```

**Important:** The label now shows "X samples" to emphasize that each second becomes a separate training sample.

### Source Selection (Step 2)

The source dropdown includes:
- "Unknown Source" (default/first option)
- All known sources with their frequencies

**Why "Unknown"?**
- For amateur radio stations with unknown locations
- For test recordings where source doesn't matter
- For exploratory data collection
- Can be updated later with actual source information

### Progress Display (Step 4)

Real-time updates show:
1. **Current chunk**: "Chunk 8/15"
2. **Progress bar**: Animated striped bar showing percentage
3. **Measurements count**: Total measurements received (chunks × WebSDRs)
4. **Status message**: Human-readable current action

### WebSocket Connection Indicator

(Could be added in future enhancement)
```
Connected    ● Green dot
Connecting   ⟳ Spinning icon
Disconnected ○ Gray dot
Error        ⚠ Warning icon
```

## Responsive Design

The UI adapts to different screen sizes:

**Desktop (≥992px):**
- 2-column layout: Form (left 8/12) + Status (right 4/12)
- Full-width controls
- Side-by-side buttons

**Tablet (768-991px):**
- 2-column layout maintained
- Slightly condensed spacing

**Mobile (<768px):**
- 1-column stacked layout
- Form first, status below
- Full-width buttons

## Color Scheme

The UI uses semantic colors for different states:

```
Recording        → Warning (Yellow/Orange)
Source Assigned  → Success (Green)
Acquiring        → Primary (Blue)
Complete         → Success (Green)
Error            → Danger (Red)
```

## Accessibility

- All form inputs have proper labels
- Progress bar has ARIA attributes
- Button states clearly indicated
- Color not sole indicator of state
- Keyboard navigation supported
- Screen reader friendly announcements

## Error Handling

If an error occurs at any stage:

```
┌─────────────────────────────────────────┐
│ Session Status                          │
├─────────────────────────────────────────┤
│ ⚠ Error                                 │
│ Error: WebSocket connection lost        │
│                                         │
│ [↻ Retry] [✗ Cancel]                   │
└─────────────────────────────────────────┘
```

## Keyboard Shortcuts (Potential Enhancement)

```
Ctrl/Cmd + Enter → Submit current step
Esc              → Cancel/Reset
Space            → Toggle selection (when focused)
Tab              → Navigate between fields
Shift + Tab      → Navigate backwards
```

## Animation Details

1. **Progress Bar**: Striped animation during acquisition
2. **Status Icon**: Spinning during "Acquiring" state
3. **Transitions**: Smooth fade between states (300ms)
4. **Button States**: Hover effects and active states

## Best Practices for Users

1. **Keep browser tab active**: WebSocket works best with active tab
2. **Wait for completion**: Don't close tab during acquisition
3. **Check WebSDR status**: Ensure receivers are online before starting
4. **Use descriptive names**: Makes it easier to identify sessions later
5. **Add notes**: Document any special conditions or observations

## Common User Flows

### Quick Recording (Known Source)
```
1. Enter session name
2. Set frequency (auto-filled from known source)
3. Adjust duration slider
4. Click "Start Recording Session"
5. Select known source from dropdown
6. Click "Assign Source"
7. Click "Start Acquisition"
8. Wait for completion
```
Time: ~30 seconds for 15-second recording

### Unknown Source Recording
```
1. Enter session name
2. Set frequency manually
3. Adjust duration slider
4. Click "Start Recording Session"
5. Keep "Unknown Source" selected
6. Click "Assign Source"
7. Click "Start Acquisition"
8. Wait for completion
```
Time: ~30 seconds for 15-second recording

### Multiple Back-to-Back Recordings
```
1. Complete first recording
2. Click "New Session"
3. Form resets to idle state
4. Repeat workflow
```

## Troubleshooting UI Issues

**Problem:** Button grayed out
- **Cause:** Required fields not filled
- **Solution:** Check session name and frequency

**Problem:** Progress bar not moving
- **Cause:** WebSocket connection lost
- **Solution:** Refresh page, check backend status

**Problem:** Can't select source
- **Cause:** No known sources in database
- **Solution:** "Unknown" is always available

**Problem:** Slider jumps in large increments
- **Cause:** Browser zoom affecting input
- **Solution:** Reset browser zoom to 100%
