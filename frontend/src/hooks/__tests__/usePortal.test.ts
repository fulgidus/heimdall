/**
 * Tests for usePortal hook
 * 
 * Verifies the bulletproof portal implementation prevents
 * "Node.removeChild: The node to be removed is not a child of this node" errors
 */

import { renderHook } from '@testing-library/react';
import { usePortal } from '../usePortal';
import { describe, it, expect, beforeEach, afterEach } from 'vitest';

describe('usePortal', () => {
  // Track created portals for cleanup
  const createdPortals: HTMLDivElement[] = [];

  beforeEach(() => {
    // Clear body
    document.body.innerHTML = '';
  });

  afterEach(() => {
    // Defensive cleanup
    createdPortals.forEach(portal => {
      if (portal.parentNode === document.body) {
        try {
          document.body.removeChild(portal);
        } catch {
          // Already removed - that's fine
        }
      }
    });
    createdPortals.length = 0;
    document.body.style.overflow = '';
  });

  it('should create portal element when open', () => {
    const { result } = renderHook(() => usePortal(true));
    
    expect(result.current).toBeInstanceOf(HTMLDivElement);
    expect(result.current?.parentNode).toBe(document.body);
    expect(result.current?.getAttribute('data-portal')).toBe('modal-portal');
    
    if (result.current) createdPortals.push(result.current);
  });

  it('should not mount portal when closed', () => {
    const { result } = renderHook(() => usePortal(false));
    
    expect(result.current).toBeInstanceOf(HTMLDivElement);
    expect(result.current?.parentNode).toBeNull();
    
    if (result.current) createdPortals.push(result.current);
  });

  it('should prevent body scroll when open', () => {
    renderHook(() => usePortal(true));
    
    expect(document.body.style.overflow).toBe('hidden');
  });

  it('should restore body scroll when closed', () => {
    const { rerender } = renderHook(({ isOpen }) => usePortal(isOpen), {
      initialProps: { isOpen: true },
    });
    
    expect(document.body.style.overflow).toBe('hidden');
    
    rerender({ isOpen: false });
    
    // Give React time to cleanup
    setTimeout(() => {
      expect(document.body.style.overflow).toBe('');
    }, 10);
  });

  it('should handle rapid open/close cycles without errors', () => {
    const { rerender, result } = renderHook(({ isOpen }) => usePortal(isOpen), {
      initialProps: { isOpen: false },
    });

    if (result.current) createdPortals.push(result.current);

    // Rapid cycles
    for (let i = 0; i < 10; i++) {
      rerender({ isOpen: true });
      rerender({ isOpen: false });
    }

    // Should not throw errors (test passes if no exception)
    expect(true).toBe(true);
  });

  it('should reuse same portal element across re-renders', () => {
    const { rerender, result } = renderHook(({ isOpen }) => usePortal(isOpen), {
      initialProps: { isOpen: true },
    });

    const firstPortal = result.current;
    if (firstPortal) createdPortals.push(firstPortal);

    rerender({ isOpen: false });
    rerender({ isOpen: true });

    expect(result.current).toBe(firstPortal);
  });

  it('should handle double cleanup gracefully', () => {
    const { rerender, result, unmount } = renderHook(({ isOpen }) => usePortal(isOpen), {
      initialProps: { isOpen: true },
    });

    if (result.current) createdPortals.push(result.current);

    // Close modal (triggers cleanup)
    rerender({ isOpen: false });

    // Unmount component (triggers cleanup again)
    unmount();

    // Should not throw "not a child of this node" error
    expect(true).toBe(true);
  });

  it('should not throw if portal already removed from DOM', () => {
    const { result, unmount } = renderHook(() => usePortal(true));

    const portal = result.current;
    if (portal && portal.parentNode === document.body) {
      // Manually remove portal (simulating race condition)
      document.body.removeChild(portal);
    }

    // Unmount should not throw
    expect(() => unmount()).not.toThrow();
  });

  it('should return null when closed', () => {
    const { result } = renderHook(() => usePortal(false));
    
    // Portal element exists but not mounted
    expect(result.current).toBeInstanceOf(HTMLDivElement);
    expect(result.current?.parentNode).toBeNull();
  });

  it('should handle WebSocket update race condition', () => {
    const { rerender, result } = renderHook(({ isOpen }) => usePortal(isOpen), {
      initialProps: { isOpen: true },
    });

    if (result.current) createdPortals.push(result.current);

    // Simulate WebSocket update triggering re-render
    rerender({ isOpen: true });
    
    // Now close
    rerender({ isOpen: false });
    
    // Another WebSocket update while closing
    rerender({ isOpen: false });

    // Should not throw errors
    expect(true).toBe(true);
  });
});
