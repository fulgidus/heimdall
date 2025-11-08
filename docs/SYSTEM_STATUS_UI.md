# System Status Page - UI Reference

## Overview
The System Status page now displays real-time health information for ALL system components, updating every second via WebSocket.

## Complete Page Layout

### System Overview Section (Top Cards)

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ System Status                                              [🔄 Refresh]       │
└───────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│    🖥️ CPU        │   💾 Database    │   📡 Radio       │   ❤️ Health      │
│                  │                  │                  │                  │
│  Microservices   │  Infrastructure  │ WebSDR Receivers │  System Health   │
│                  │                  │                  │                  │
│      3/3         │      5/5         │      7/7         │      Good        │
│    Healthy       │    Healthy       │     Online       │    Overall       │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

**Colors**:
- Microservices: Blue background
- Infrastructure: Gray background
- WebSDR: Green background
- System Health: Info blue background

**Updates**: Every 1 second from WebSocket events

---

### Microservices Health Section (Left Card)

```
┌─────────────────────────────────────────────────────────────────┐
│ Microservices Health                                            │
├─────────────────────────────────────────────────────────────────┤
│ Service             Status        Health                        │
├─────────────────────────────────────────────────────────────────┤
│ 🟢 Backend          ✅ healthy    ▓▓▓▓▓▓▓▓▓▓ (23.5ms)           │
│ 🟢 Training         ✅ healthy    ▓▓▓▓▓▓▓▓▓▓ (45.2ms)           │
│ 🟢 Inference        ✅ healthy    ▓▓▓▓▓▓▓▓▓▓ (12.8ms)           │
│                                                                 │
│    Model Info: v1.0.0, 89% accuracy, 1200 predictions          │
└─────────────────────────────────────────────────────────────────┘
```

**Status Colors**:
- Healthy: Green badge
- Degraded: Yellow/Warning badge
- Unhealthy: Red badge

**Information Shown**:
- Service name (capitalized)
- Status badge (healthy/unhealthy/degraded)
- Response time in milliseconds
- Progress bar visualization
- Model info for inference service

**Updates**: Every 1 second via WebSocket

---

### Infrastructure Components Section (Right Card) - NEW!

```
┌─────────────────────────────────────────────────────────────────┐
│ Infrastructure Components                                       │
├─────────────────────────────────────────────────────────────────┤
│ Component           Status        Details                       │
├─────────────────────────────────────────────────────────────────┤
│ 🗄️ PostgreSQL       ✅ healthy    Database connection OK        │
│   database                                                      │
│                                                                 │
│ ⚡ Redis            ✅ healthy    Cache connection OK           │
│   cache                                                         │
│                                                                 │
│ 📬 RabbitMQ         ✅ healthy    Message queue connection OK   │
│   queue                                                         │
│                                                                 │
│ 📦 MinIO            ✅ healthy    Object storage OK, bucket OK  │
│   storage                                                       │
│                                                                 │
│ ⚙️ Celery           ✅ healthy    2 worker(s) active            │
│   worker                        Workers: 2                      │
└─────────────────────────────────────────────────────────────────┘
```

**Component Icons**:
- Database (PostgreSQL): 🗄️ database icon
- Cache (Redis): ⚡ lightning icon
- Queue (RabbitMQ): 📬 queue icon
- Storage (MinIO): 📦 package icon
- Worker (Celery): ⚙️ CPU icon

**Status Colors**:
- Healthy: Green badge with checkmark
- Warning: Yellow badge with warning icon
- Unhealthy: Red badge with X icon
- Unknown: Gray badge with question mark

**Information Shown**:
- Component name (capitalized)
- Component type (database, cache, queue, storage, worker)
- Status badge
- Status message (e.g., "Database connection OK")
- Additional metrics (e.g., worker count for Celery)
- Error messages if unhealthy

**Updates**: Every 1 second via WebSocket - **THIS IS NEW!**

---

### WebSDR Receivers Section (Bottom Left)

```
┌─────────────────────────────────────────────────────────────────┐
│ WebSDR Receivers                                                │
├─────────────────────────────────────────────────────────────────┤
│ Location            Status        Response Time                 │
├─────────────────────────────────────────────────────────────────┤
│ 🟢 Torino           ✅ Online     123ms                         │
│   Italy                                                         │
│ 🟢 Milano           ✅ Online     156ms                         │
│   Italy                                                         │
│ 🔴 Roma             ❌ Offline    N/A                           │
│   Italy                                                         │
│ ... (7 total)                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Updates**: Every 60 seconds (separate WebSocket event channel)

---

### ML Model Status Section (Bottom Right)

```
┌─────────────────────────────────────────────────────────────────┐
│ ML Model Status                                                 │
├─────────────────────────────────────────────────────────────────┤
│ Version             1.0.0                                       │
│ Health Status       ✅ healthy                                  │
│                                                                 │
│ Accuracy            89.00%                                      │
│ Loaded At           2025-11-04 15:30:00                         │
│                                                                 │
│ Total Predictions   1234                                        │
│ Successful          1200                                        │
│ Failed              34                                          │
│                                                                 │
│ Last Prediction     2025-11-04 15:49:27                         │
└─────────────────────────────────────────────────────────────────┘
```

**Updates**: Every 30 seconds (REST API poll)

---

## Real-time Update Behavior

### WebSocket Event Flow

1. **Celery Beat** triggers `monitor_comprehensive_health` every 1 second
2. **Health Monitor** checks all 8 components (3 microservices + 5 infrastructure)
3. **Event Publisher** broadcasts `system:comprehensive_health` event to RabbitMQ
4. **RabbitMQ Consumer** receives event and broadcasts to WebSocket clients
5. **Frontend** receives WebSocket message and updates store
6. **React** re-renders affected components with new data

### Visual Feedback

**When a component becomes unhealthy**:
1. Badge changes from green ✅ to red ❌
2. Status text changes from "healthy" to "unhealthy"
3. Error message appears in Details column
4. System Overview "Overall" changes from "Good" to "Degraded"
5. Component count decreases (e.g., 5/5 → 4/5)

**Example: MinIO goes offline**
```
Before (healthy):
┌─────────────────────────────────────────────────────────┐
│ 📦 MinIO            ✅ healthy    Object storage OK     │
│   storage                                               │
└─────────────────────────────────────────────────────────┘

After (unhealthy, 1 second later):
┌─────────────────────────────────────────────────────────┐
│ 📦 MinIO            ❌ unhealthy  Connection failed     │
│   storage                        Error: Timeout         │
└─────────────────────────────────────────────────────────┘
```

### Update Frequency Summary

| Component | Update Method | Frequency |
|-----------|---------------|-----------|
| Microservices | WebSocket | 1 second |
| Infrastructure | WebSocket | 1 second |
| WebSDRs | WebSocket | 60 seconds |
| Model Info | REST API | 30 seconds |
| System Overview | Computed | Real-time (on any update) |

---

## Component Type Icons Reference

| Type | Icon | Component |
|------|------|-----------|
| database | 🗄️ | PostgreSQL, TimescaleDB |
| cache | ⚡ | Redis |
| queue | 📬 | RabbitMQ |
| storage | 📦 | MinIO |
| worker | ⚙️ | Celery workers |
| receiver | 📡 | WebSDR stations |
| service | 🖥️ | Microservices |

---

## Status Badge Colors

| Status | Badge Color | Icon | Example |
|--------|-------------|------|---------|
| healthy | Green (`bg-light-success`) | ✅ | Database connection OK |
| warning | Yellow (`bg-light-warning`) | ⚠️ | Bucket not found |
| unhealthy | Red (`bg-light-danger`) | ❌ | Connection timeout |
| degraded | Orange (`bg-light-warning`) | ⚠️ | Slow response time |
| unknown | Gray (`bg-light-secondary`) | ❓ | No data available |

---

## Browser Console Output (Debug)

When the page is working correctly, you should see in the browser console:

```javascript
[WebSocketContext] WebSocket connected
[useSystemWebSocket] Received system:comprehensive_health: {
  event: "system:comprehensive_health",
  timestamp: "2025-11-04T15:49:27.802923",
  data: {
    components: {
      backend: { status: "healthy", response_time_ms: 23.5, ... },
      training: { status: "healthy", response_time_ms: 45.2, ... },
      inference: { status: "healthy", response_time_ms: 12.8, ... },
      postgresql: { status: "healthy", type: "database", ... },
      redis: { status: "healthy", type: "cache", ... },
      rabbitmq: { status: "healthy", type: "queue", ... },
      minio: { status: "healthy", type: "storage", ... },
      celery: { status: "healthy", type: "worker", worker_count: 2, ... }
    }
  }
}
```

This message should appear **every 1 second**.

---

## Responsive Design

### Desktop (≥992px)
- 2 columns layout
- Microservices and Infrastructure side-by-side
- WebSDRs and ML Model side-by-side

### Tablet (768px - 991px)
- 1 column layout
- Full width cards stacked vertically

### Mobile (<768px)
- 1 column layout
- Simplified card headers
- Scrollable tables

---

## Accessibility

- **ARIA labels** on all status indicators
- **Keyboard navigation** supported
- **Screen reader** friendly status messages
- **Color contrast** meets WCAG AA standards
- **Focus indicators** on interactive elements

---

## Performance Considerations

### WebSocket Efficiency
- Single WebSocket connection shared across app
- Batched updates every 1 second (not per-component)
- Payload size: ~2-5KB compressed
- Zero polling - push-based updates only

### React Optimization
- Zustand store batches state updates
- Components only re-render when their data changes
- Virtual scrolling for large lists (if needed)
- Memoized computed values

### Backend Optimization
- Health checks run concurrently (asyncio.gather)
- Connection pooling for all dependencies
- Timeout protection (5 seconds max per check)
- Non-blocking event publishing

---

## Future Enhancements

Potential improvements (not in scope):
- [ ] Historical health charts (uptime over time)
- [ ] Alert thresholds and notifications
- [ ] Filtering and search for components
- [ ] Export health report as CSV/JSON
- [ ] Mobile app with push notifications
- [ ] Component-specific detail pages
- [ ] Health check scheduling configuration UI

---

## Summary

The System Status page now provides:
- ✅ **Complete visibility** into all system components
- ✅ **Real-time updates** every 1 second via WebSocket
- ✅ **No fake data** - everything is actual system state
- ✅ **Rich information** - response times, errors, metrics
- ✅ **Visual clarity** - icons, colors, badges
- ✅ **Infrastructure monitoring** - NEW feature showing PostgreSQL, Redis, RabbitMQ, MinIO, Celery
- ✅ **Microservices monitoring** - Enhanced with model info
- ✅ **WebSDR monitoring** - Existing functionality preserved
- ✅ **ML model status** - Existing functionality preserved

All requirements from the issue have been met.
