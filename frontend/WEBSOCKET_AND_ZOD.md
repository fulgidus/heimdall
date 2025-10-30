# WebSocket and Zod Validation Guide

## Overview

This document describes the WebSocket implementation and Zod runtime validation added to the Heimdall frontend.

## WebSocket Implementation

### Configuration

The frontend uses WebSocket for real-time updates from the backend. Configuration is in `src/store/dashboardStore.ts`:

```typescript
wsEnabled: true  // WebSocket enabled
wsUrl: ws://localhost:80/ws  // Default URL (configurable via VITE_SOCKET_URL)
```

### Architecture

```
Frontend → Envoy (port 80) → Backend (port 8001)
   |            |                    |
   |        /ws route            /ws endpoint
   |      WebSocket upgrade    WebSocket handler
```

### Real-time Events

The WebSocket connection receives these events:

- `services:health` - Service health status updates
- `websdrs:status` - WebSDR receiver status changes
- `signals:detected` - New signal detections
- `localizations:updated` - New localization results

### Connection Management

- **Auto-reconnect**: Exponential backoff (1s → 30s max)
- **Heartbeat**: Ping/pong every 30 seconds
- **State tracking**: DISCONNECTED, CONNECTING, CONNECTED, RECONNECTING

### Environment Variables

```bash
# .env.development or .env.production
VITE_SOCKET_URL=ws://localhost:80/ws
```

## Zod Runtime Validation

### Purpose

All API responses are validated at runtime using Zod schemas to:
- Catch malformed data early
- Provide clear error messages
- Ensure type safety beyond compile-time
- Prevent UI crashes from bad data

### Schema Location

All schemas are defined in `src/services/api/schemas.ts`:

```typescript
// Example: WebSDR configuration schema
export const WebSDRConfigSchema = z.object({
    id: z.string().uuid(),
    name: z.string(),
    latitude: z.number().min(-90).max(90),
    longitude: z.number().min(-180).max(180),
    // ... more fields
});

export type WebSDRConfig = z.infer<typeof WebSDRConfigSchema>;
```

### Usage in API Services

Every API service validates responses:

```typescript
// src/services/api/websdr.ts
export async function getWebSDRs(): Promise<WebSDRConfig[]> {
    const response = await api.get('/v1/acquisition/websdrs-all');
    
    // Validate with Zod - throws ZodError if invalid
    const validated = z.array(WebSDRConfigSchema).parse(response.data);
    
    return validated;
}
```

### Error Handling

When validation fails, Zod throws a detailed error:

```typescript
try {
    const websdrs = await getWebSDRs();
} catch (error) {
    if (error instanceof z.ZodError) {
        // Detailed validation errors
        console.error('Invalid API response:', error.issues);
        // error.issues contains:
        // - path: which field failed
        // - message: why it failed
        // - expected: what was expected
        // - received: what was received
    }
}
```

### Available Schemas

1. **WebSDR**: `WebSDRConfigSchema`, `WebSDRHealthStatusSchema`
2. **Acquisition**: `AcquisitionRequestSchema`, `AcquisitionTaskResponseSchema`, `AcquisitionStatusResponseSchema`
3. **Inference**: `ModelInfoSchema`, `ModelPerformanceMetricsSchema`, `PredictionResponseSchema`, `BatchPredictionResponseSchema`
4. **System**: `ServiceHealthSchema`, `SystemMetricsSchema`
5. **Session**: `RecordingSessionSchema`, `SessionListResponseSchema`, `KnownSourceSchema`, `SessionAnalyticsSchema`
6. **Localization**: `LocalizationResultSchema`, `UncertaintyEllipseSchema`
7. **Analytics**: `DashboardMetricsSchema`, `PredictionMetricsSchema`, `WebSDRPerformanceSchema`, `SystemPerformanceSchema`

### Validation Points

23 validation points across 6 service files:
- websdr.ts: 4 validations
- acquisition.ts: 2 validations
- inference.ts: 5 validations
- system.ts: 1 validation
- analytics.ts: 4 validations
- session.ts: 7 validations

## Troubleshooting

### WebSocket Connection Issues

**Problem**: WebSocket not connecting
**Solutions**:
1. Verify backend `/ws` endpoint is active
2. Check Envoy is running and routing `/ws`
3. Verify `VITE_SOCKET_URL` environment variable
4. Check browser console for connection errors

**Problem**: WebSocket disconnects frequently
**Solutions**:
1. Check network stability
2. Verify backend WebSocket handler is healthy
3. Review backend logs for errors

### Zod Validation Errors

**Problem**: API calls failing with ZodError
**Solutions**:
1. Check backend response format matches schema
2. Review `error.issues` for specific field errors
3. Update schema if backend API changed
4. Check backend logs for data corruption

**Problem**: Type mismatch errors
**Solutions**:
1. Ensure using latest schemas from `schemas.ts`
2. Re-run `npm run build` to check TypeScript errors
3. Update imports to use types from `schemas.ts`

## Best Practices

### WebSocket

1. **Always handle connection states**: Display connection status to users
2. **Implement fallback**: Polling as backup if WebSocket fails
3. **Rate limiting**: Don't spam reconnection attempts
4. **Clean up**: Disconnect WebSocket on component unmount

### Zod Validation

1. **Keep schemas in sync**: Update schemas when API changes
2. **Use specific types**: Avoid `z.unknown()` when possible
3. **Add constraints**: Use `.min()`, `.max()`, `.email()`, etc.
4. **Handle errors gracefully**: Show user-friendly messages, not raw Zod errors
5. **Test schemas**: Write unit tests for schema validation

## Migration Guide

If updating from old code without Zod:

```typescript
// Before
const response = await api.get<WebSDRConfig[]>('/websdrs');
return response.data;

// After
const response = await api.get('/websdrs');
const validated = z.array(WebSDRConfigSchema).parse(response.data);
return validated;
```

## Performance

- **Zod validation overhead**: ~0.1-1ms per validation (negligible)
- **WebSocket overhead**: Minimal, more efficient than polling
- **Memory usage**: WebSocket manager ~1-2MB, Zod schemas ~500KB

## Resources

- [Zod Documentation](https://zod.dev/)
- [WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
- [Envoy WebSocket](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/http/upgrades)
