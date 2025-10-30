# WebSDR Management - Zod Validation Error Fix

## Problem

When accessing the WebSDR Management page, the following error appeared:

```
Error! [ { "expected": "array", "code": "invalid_type", "path": [], "message": "Invalid input" } ]
```

This error indicates that the Zod schema validation for WebSDR API responses was failing because the API response data was not in the expected format.

## Root Causes

1. **Unhandled Response Wrapping**: The API response might be wrapped in an additional object structure (e.g., `{data: [...]}`), but the code assumed `response.data` would be the array directly.

2. **Null/Undefined Response**: The API might return `null` or `undefined` in some cases.

3. **Poor Error Messages**: Zod validation errors were not being caught and converted to readable error messages.

4. **Inconsistent URL Paths**: Some API endpoints used `/api/v1/...` while others used `/v1/...`, leading to routing issues.

## Solution

Updated `/frontend/src/services/api/websdr.ts` with the following improvements:

### 1. Added Response Data Validation

```typescript
// Handle case where response.data might not be an array
let data = response.data;
if (data && typeof data === 'object' && 'data' in data) {
    // If response is wrapped in {data: [...]}
    data = data.data;
}

if (!Array.isArray(data)) {
    console.error('❌ getWebSDRs(): Response is not an array. Received:', data);
    throw new Error(`Expected array of WebSDRs, got ${typeof data}: ${JSON.stringify(data)}`);
}
```

### 2. Added Try-Catch for Zod Errors

```typescript
try {
    const validated = z.array(WebSDRConfigSchema).parse(data);
    console.log('✅ WebSDRService.getWebSDRs(): ricevuti', validated.length, 'WebSDRs');
    return validated;
} catch (zodError) {
    console.error('❌ Zod validation error in getWebSDRs():', zodError);
    if (zodError instanceof z.ZodError) {
        throw new Error(`WebSDR validation error: ${zodError.message}`);
    }
    throw zodError;
}
```

### 3. Fixed URL Inconsistencies

- Changed `updateWebSDR` endpoint from `/api/v1/acquisition/websdrs/{id}` to `/v1/acquisition/websdrs/{id}`
- Changed `deleteWebSDR` endpoint from `/api/v1/acquisition/websdrs/{id}` to `/v1/acquisition/websdrs/{id}`

This ensures all WebSDR endpoints use consistent URL paths.

### 4. Applied Similar Fixes to Related Functions

- `checkWebSDRHealth()`: Added response validation and error handling
- `createWebSDR()`: Added response validation and error handling
- `updateWebSDR()`: Added response validation and error handling

## Files Modified

- `/frontend/src/services/api/websdr.ts`

## Impact

- **User Experience**: Error messages are now much more informative and easier to debug
- **Reliability**: The API service now gracefully handles various response formats
- **Console Logging**: Detailed debug logs help identify issues quickly
- **API Consistency**: All WebSDR endpoints now use consistent URL paths

## Testing

To verify the fix:

1. Navigate to the WebSDR Management page
2. Verify that the WebSDR list loads correctly
3. Check the browser console for the enhanced debug logs
4. Try creating, editing, and deleting WebSDR entries
5. Verify health checks work properly

## Related Issues

- This fix addresses the Zod validation error in WebSDR Management
- Improves API reliability and error reporting across all WebSDR operations

## Future Improvements

1. Consider implementing a global API response wrapper standard
2. Add request/response logging middleware for better debugging
3. Implement automatic retry logic for transient failures
4. Add more comprehensive error recovery strategies
