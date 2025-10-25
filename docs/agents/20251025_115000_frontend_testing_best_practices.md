# Dashboard Test Suite - Recommendations and Best Practices

**Date**: 2025-10-25  
**Context**: Frontend comprehensive test suite implementation (Phase 7)

---

## Overview

This document provides recommendations for maintaining and extending the Dashboard test suite, based on lessons learned from fixing 15 integration test failures.

---

## Lessons Learned

### 1. Mock Store Management

**Issue**: Mock store was incomplete, causing `TypeError: connectWebSocket is not a function`

**Best Practice**:
```typescript
// ✅ DO: Ensure mock includes ALL store methods and properties
const createMockStore = (overrides = {}) => ({
    // Include every property and method from the actual store interface
    requiredMethod: mockFn,
    requiredProperty: defaultValue,
    ...overrides,
});

// ❌ DON'T: Assume store interface is simpler than it is
```

**Verification Checklist**:
- [ ] Run TypeScript check on mock store type
- [ ] Compare mock with actual store interface line-by-line
- [ ] Verify all async methods are included
- [ ] Check for WebSocket/real-time methods

### 2. Text Matching and Regex Patterns

**Issue**: Tests searched for text without accounting for component transformations

**Common Patterns**:
```typescript
// Dashes to spaces: 'api-gateway' → 'api gateway'
expect(screen.getByText(/api gateway/i)).toBeInTheDocument();

// Special characters need escaping: 'Connecting...' 
expect(screen.getByText(/connecting\.\.\./i)).toBeInTheDocument();

// Case insensitive when appropriate:
expect(screen.getByText(/healthy/i)).toBeInTheDocument();  // ✅ Good

// Exact match when needed:
expect(screen.getByText('N/A')).toBeInTheDocument();  // ✅ For exact strings
```

**Best Practice**:
1. Review component render logic for text transformations
2. Document any text modifications in component code
3. Use case-insensitive regex unless exact case is critical
4. Always escape special regex characters (`, ?, *, +, [, ], (, ), {, }, |, \, ., ^, $`)

### 3. Handling Multiple Similar Elements

**Issue**: Tests assumed single elements but components rendered multiple

**Solution**:
```typescript
// ✅ DO: Use getAllByRole/getAllByText when multiple elements exist
const refreshButtons = screen.getAllByRole('button', { name: /refresh/i });
fireEvent.click(refreshButtons[0]);

// ❌ DON'T: Use getByRole when multiple elements exist
const refreshButton = screen.getByRole('button', { name: /refresh/i }); // Will throw!
```

**Decision Matrix**:
- One element expected → use `getByRole`, `getByText`, etc.
- Multiple elements expected → use `getAllByRole`, `getAllByText`, etc.
- Optional element → use `queryByRole`, `queryByText`, etc.

### 4. Mock Cleanup Between Tests

**Issue**: Mock call counts weren't reset, causing flaky tests

**Pattern**:
```typescript
describe('Suite', () => {
    beforeEach(() => {
        vi.clearAllMocks();  // ✅ Always clear before each test
    });

    it('test 1', () => {
        renderComponent();
        expect(mockFn).toHaveBeenCalledTimes(1);
    });

    it('test 2', () => {
        renderComponent();
        expect(mockFn).toHaveBeenCalledTimes(1);  // ✅ Starts fresh
    });
});
```

### 5. Fake Timers and Async Tests

**Issue**: Polling tests were unreliable with improper timer management

**Correct Pattern**:
```typescript
it('should poll at intervals', () => {
    vi.useFakeTimers();
    vi.clearAllMocks();  // Clear before using fake timers
    
    render(<Component />);
    
    // Initial render call
    expect(mockFn).toHaveBeenCalledTimes(1);
    
    // Advance time
    vi.advanceTimersByTime(30000);
    expect(mockFn).toHaveBeenCalledTimes(2);
    
    vi.useRealTimers();  // Always restore!
});
```

### 6. Component State Overrides

**Issue**: Tests assumed default state but needed specific conditions

**Pattern**:
```typescript
// ✅ DO: Explicitly set state for conditional branches
renderComponent({
    isLoading: true,
    error: null,
    wsConnectionState: 'Disconnected',
    wsEnabled: true,
});

// ❌ DON'T: Rely on defaults that might change
renderComponent();  // What are the defaults?
```

---

## Recommendations for Phase 7 Frontend

### 1. Test Organization

**Structure**:
```
src/pages/
  ├── Dashboard.tsx
  ├── Dashboard.unit.test.tsx      # Unit tests (component logic)
  ├── Dashboard.integration.test.tsx # Integration tests (with stores)
  └── Dashboard.e2e.test.tsx       # E2E tests (with real API)

src/components/
  ├── __tests__/
  │   ├── ServiceCard.test.tsx
  │   ├── WebSDRMap.test.tsx
  │   └── ...
```

### 2. Mock Store Standards

**Template**:
```typescript
// Always create a complete mock factory
const createMockStore = (overrides = {}) => ({
    // Properties
    property1: initialValue,
    property2: null,
    
    // Methods (use vi.fn())
    method1: vi.fn(),
    method2: vi.fn(),
    
    // Apply overrides
    ...overrides,
});

// Always include in beforeEach
beforeEach(() => {
    vi.clearAllMocks();
    (useStore as any).mockReturnValue(createMockStore());
});
```

### 3. Test Naming Convention

```typescript
describe('Component Name', () => {
    describe('Feature Area', () => {
        it('should [action] when [condition]', () => {
            // Arrange
            const input = ...;
            
            // Act
            render(<Component {...input} />);
            
            // Assert
            expect(...).toBeInTheDocument();
        });
    });
});
```

### 4. Coverage Targets

**Target**: ≥80% across all files

```bash
# Generate coverage report
pnpm test -- --coverage

# Target breakdown by component type:
# Pages: 85%
# Components: 80%
# Hooks: 85%
# Utils: 90%
```

### 5. Test Data Factories

**Create reusable test data**:
```typescript
// In test/factories.ts
export const createMockWebSDR = (overrides = {}) => ({
    id: 1,
    name: 'Turin',
    location_name: 'Turin, Italy',
    is_active: true,
    ...overrides,
});

export const createMockServiceHealth = (overrides = {}) => ({
    status: 'healthy',
    service: 'api-gateway',
    version: '0.1.0',
    timestamp: new Date().toISOString(),
    ...overrides,
});

// In tests:
renderDashboard({
    data: {
        websdrs: [createMockWebSDR(), createMockWebSDR({ name: 'Milan' })],
    },
});
```

### 6. Common Testing Patterns

**Testing async operations**:
```typescript
it('should handle async operations', async () => {
    const mockFn = vi.fn().mockResolvedValue({ data: 'test' });
    
    render(<Component />);
    
    await waitFor(() => {
        expect(mockFn).toHaveBeenCalled();
    });
    
    expect(screen.getByText('test')).toBeInTheDocument();
});
```

**Testing error states**:
```typescript
it('should display error message on failure', () => {
    mockFetch.mockRejectedValue(new Error('Network error'));
    
    render(<Component />);
    
    expect(screen.getByText(/network error/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
});
```

**Testing user interactions**:
```typescript
it('should respond to user clicks', async () => {
    render(<Component />);
    
    const button = screen.getByRole('button', { name: /save/i });
    fireEvent.click(button);
    
    await waitFor(() => {
        expect(mockOnSave).toHaveBeenCalled();
    });
});
```

---

## Checklist for New Tests

Before submitting a test, ensure:

- [ ] **Mock completeness**: All store methods/properties included
- [ ] **Clear assertions**: Each `expect()` tests one behavior
- [ ] **Descriptive names**: Test name explains what it tests
- [ ] **Proper cleanup**: `vi.clearAllMocks()` in `beforeEach`
- [ ] **No flakiness**: Tests pass consistently (run 5x)
- [ ] **Proper timers**: Fake timers cleaned up with `vi.useRealTimers()`
- [ ] **Good coverage**: Covers happy path + error cases
- [ ] **Documentation**: Comments explain non-obvious logic
- [ ] **Isolated tests**: No dependencies between tests
- [ ] **Performance**: Tests complete in <1 second (except integration)

---

## Common Issues and Solutions

### Issue 1: "getByRole not found"
```typescript
// ❌ Problem: Multiple elements
const button = screen.getByRole('button', { name: /refresh/i });

// ✅ Solution: Use getAllByRole or add more specificity
const buttons = screen.getAllByRole('button', { name: /refresh/i });
fireEvent.click(buttons[0]);

// ✅ Alternative: Use a more specific query
const button = screen.getByRole('button', { name: /refresh.*data/i });
```

### Issue 2: "TypeError: mockFn is not a function"
```typescript
// ❌ Problem: Mock property not included
const mockStore = { data: {...} };  // Missing methods!

// ✅ Solution: Include all methods
const mockStore = {
    data: {...},
    fetchData: vi.fn(),
    setError: vi.fn(),
};
```

### Issue 3: Tests pass individually but fail in suite
```typescript
// ❌ Problem: Missing mock cleanup
beforeEach(() => {
    // Missing: vi.clearAllMocks();
    renderComponent();
});

// ✅ Solution: Clear mocks first
beforeEach(() => {
    vi.clearAllMocks();
    renderComponent();
});
```

### Issue 4: Fake timer tests hang
```typescript
// ❌ Problem: Timer not restored
it('should poll', () => {
    vi.useFakeTimers();
    // ... test code ...
    // Missing: vi.useRealTimers();
});

// ✅ Solution: Always restore timers
it('should poll', () => {
    vi.useFakeTimers();
    try {
        // ... test code ...
    } finally {
        vi.useRealTimers();
    }
});
```

---

## Performance Optimization

### Test Execution Time

Current baseline for Dashboard tests:
- **Setup**: 181s (environment)
- **Tests**: 21.81s (15 tests = ~1.45s each)
- **Total**: ~200s for full suite

**Optimization strategies**:
1. **Parallel execution**: Use `--reporter=verbose --threads=4`
2. **Selective testing**: `pnpm test -- src/pages/Dashboard` (skip others)
3. **Watch mode**: `pnpm test -- --watch` for development
4. **Coverage caching**: Use `--coverage` only before PR

---

## CI/CD Integration

**GitHub Actions setup**:
```yaml
- name: Run Frontend Tests
  run: |
    cd frontend
    pnpm test
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./frontend/coverage/coverage-final.json
```

---

## References

- [Vitest Documentation](https://vitest.dev/)
- [React Testing Library](https://testing-library.com/react)
- [Testing Best Practices](https://kentcdodds.com/blog/common-mistakes-with-react-testing-library)

---

## Summary

By following these recommendations, the Phase 7 frontend test suite will be:
- ✅ More maintainable
- ✅ More reliable
- ✅ Easier to extend
- ✅ Better documented
- ✅ Faster to execute

---

*Last updated: 2025-10-25*
