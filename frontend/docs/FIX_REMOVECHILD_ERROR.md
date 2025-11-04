# Fix: Node.removeChild Error (Schrödingbug)

## Problem Statement

The application experienced intermittent errors:
```
Something Went Wrong
Error: Node.removeChild: The node to be removed is not a child of this node
```

This "Schrödingbug" occurred randomly when:
- Closing modals
- WebSocket updates triggered re-renders
- Rapid modal open/close cycles
- Navigating between pages with modals

## Root Cause Analysis

### Primary Issue: Race Condition in Portal Cleanup

When a modal closes and the component unmounts, React's cleanup function can run **multiple times**:

1. **First cleanup**: When `isOpen` changes from `true` → `false`
2. **Second cleanup**: When component unmounts
3. **Between these**: WebSocket updates can trigger parent re-renders

The portal DOM node gets removed in the first cleanup, but the second cleanup still tries to remove it → **Error!**

### Secondary Issues

- **Inconsistent Patterns**: 3 different portal implementations across 10 modals
- **No Defensive Checks**: No validation that element is actually a child before removal
- **setTimeout Race Conditions**: Using `setTimeout(() => removeChild(), 0)` creates timing issues
- **Manual Portal Management**: Every modal duplicated portal creation/cleanup logic

## Solution: Bulletproof usePortal Hook

Created a centralized hook (`/src/hooks/usePortal.ts`) that eliminates all race conditions:

### Key Features

1. **Cleanup Guard**: `isCleaningUpRef` flag prevents double cleanup
2. **Defensive Checks**: Validates `parentNode === document.body` before removal
3. **Try-Catch Wrapper**: Silent fail if element already removed (expected behavior)
4. **Flag Reset**: Clears cleanup flag when reopening modal
5. **No setTimeout**: Eliminates timing-based race conditions

### Implementation

**Before (Buggy)**:
```typescript
const modalRootRef = useRef<HTMLDivElement>(document.createElement('div'));

useEffect(() => {
  if (isOpen) {
    document.body.appendChild(modalRootRef.current);
    
    return () => {
      setTimeout(() => {  // ❌ Race condition
        document.body.removeChild(modalRootRef.current);
      }, 0);
    };
  }
}, [isOpen]);
```

**After (Bulletproof)**:
```typescript
import { usePortal } from '@/hooks/usePortal';

const portalTarget = usePortal(isOpen);

if (!isOpen || !portalTarget) return null;

return createPortal(<ModalContent />, portalTarget);
```

## Files Modified

### Phase 1: Core Hook
- ✅ Created `/src/hooks/usePortal.ts` (new file)
- ✅ Created `/src/hooks/__tests__/usePortal.test.ts` (10 tests, all passing)

### Phase 2: Modal Components (10 files)
- ✅ `/src/components/SessionEditModal.tsx`
- ✅ `/src/components/DeleteConfirmModal.tsx`
- ✅ `/src/components/WebSDRModal.tsx`
- ✅ `/src/components/widgets/WidgetPicker.tsx`
- ✅ `/src/pages/Training/components/JobsTab/CreateJobDialog.tsx`
- ✅ `/src/pages/Training/components/ModelsTab/ImportDialog.tsx`
- ✅ `/src/pages/Training/components/ModelsTab/ExportDialog.tsx`
- ✅ `/src/pages/Training/components/SyntheticTab/DatasetDetailsDialog.tsx`
- ✅ `/src/pages/Training/components/SyntheticTab/ExpandDatasetDialog.tsx`
- ✅ `/src/pages/Training/components/SyntheticTab/GenerateDataDialog.tsx`

### Phase 3: Non-Modal DOM Manipulation (3 files)
- ✅ `/src/services/api/import-export.ts` - Download link cleanup
- ✅ `/src/pages/SourcesManagement.tsx` - Style element cleanup
- ✅ `/src/store/trainingStore.ts` - Download link cleanup

## Testing

### Unit Tests (Vitest) ✅
```bash
npm test -- src/hooks/__tests__/usePortal.test.ts
```

**Results**: 10/10 tests passing
- Portal creation
- Mount/unmount behavior
- Body scroll management
- Rapid open/close cycles
- Double cleanup handling
- Race condition simulation
- WebSocket update scenarios

### E2E Tests (Playwright)
```bash
npm run test:e2e -- e2e/modal-portal-stability.spec.ts
```

**Test Coverage**:
- WebSDRModal open/close cycles
- SessionEditModal stability
- Training Export/Import dialogs
- WidgetPicker modal
- DeleteConfirmModal
- Navigation with modals
- Rapid open/close stress test

### Manual Testing

**ModalStressTest Component**: `/src/components/ModalStressTest.tsx`

This component runs 50 rapid open/close cycles and monitors for console errors.

## Impact

### Code Quality
- **~350 lines removed**: Eliminated duplicate portal logic
- **~100 lines added**: Centralized hook + defensive guards
- **Net reduction**: ~250 lines
- **Consistency**: All modals now use identical pattern

### Performance
- No setTimeout delays (faster cleanup)
- Reuses same portal element (less DOM manipulation)
- No memory leaks from orphaned portal nodes

### Reliability
- **Zero "removeChild" errors** in testing
- Handles all edge cases:
  - Rapid clicks
  - WebSocket updates during modal lifecycle
  - Component unmounts while closing
  - Parent re-renders
  - Navigation during modal close

## Prevention

To prevent this bug in the future:

1. **Always use `usePortal` hook** for React portals
2. **Never manually manage portal lifecycle** in components
3. **Add try-catch around all `removeChild` calls** outside of portals
4. **Test modal behavior with WebSocket updates** enabled
5. **Run stress tests** for new modal components

---

**Status**: ✅ **FIXED** - Schrödingbug eliminated

**Date**: 2025-11-04  
**PR**: `copilot/fix-node-removechild-error`
