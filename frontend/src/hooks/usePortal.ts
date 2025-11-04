/**
 * usePortal Hook
 * 
 * Bulletproof React portal management that prevents the infamous
 * "Node.removeChild: The node to be removed is not a child of this node" error.
 * 
 * ROOT CAUSE OF THE BUG:
 * When a modal closes and the component unmounts, React's cleanup function
 * can run multiple times in these scenarios:
 * 1. When isOpen changes from true → false (cleanup runs)
 * 2. When component unmounts (cleanup runs AGAIN)
 * 3. Between these events, WebSocket updates can trigger re-renders
 * 4. The portal might be removed before the second cleanup attempt
 * 
 * SOLUTION:
 * - Use isCleaningUpRef to track cleanup state
 * - Reset flag when reopening modal (in effect body, not cleanup)
 * - Defensive checks: portal exists AND is child of body
 * - Try-catch for removeChild (silent fail if already removed)
 * - No setTimeout (eliminates race conditions)
 * 
 * USAGE:
 * ```tsx
 * const portalTarget = usePortal(isOpen);
 * 
 * if (!isOpen || !portalTarget) return null;
 * 
 * return createPortal(
 *   <div className="modal">...</div>,
 *   portalTarget
 * );
 * ```
 * 
 * @param isOpen - Whether the portal should be mounted
 * @returns Portal container element or null
 */

import { useEffect, useRef } from 'react';

export function usePortal(isOpen: boolean): HTMLDivElement | null {
  const portalRef = useRef<HTMLDivElement | null>(null);
  const isMountedRef = useRef(false);
  const isCleaningUpRef = useRef(false);

  // Lazy initialization: create div once and reuse
  if (!portalRef.current) {
    portalRef.current = document.createElement('div');
    portalRef.current.setAttribute('data-portal', 'modal-portal');
  }

  useEffect(() => {
    const portal = portalRef.current;
    if (!portal) return;

    if (isOpen) {
      // CRITICAL: Reset cleanup flag when opening
      // This allows the portal to be removed again on next close
      isCleaningUpRef.current = false;

      // Mount portal if not already mounted
      if (!isMountedRef.current) {
        try {
          document.body.appendChild(portal);
          isMountedRef.current = true;
          // Prevent body scroll when modal is open (set once on mount)
          document.body.style.overflow = 'hidden';
        } catch (error) {
          console.error('Failed to mount portal:', error);
        }
      }
    }

    // Cleanup function (runs on dependency change or unmount)
    return () => {
      // GUARD: Prevent double cleanup
      if (isCleaningUpRef.current) {
        return;
      }
      isCleaningUpRef.current = true;

      // Restore body scroll
      document.body.style.overflow = '';

      // DEFENSIVE: Only remove if mounted AND actually in DOM
      if (isMountedRef.current && portal.parentNode === document.body) {
        try {
          document.body.removeChild(portal);
          isMountedRef.current = false;
        } catch (error) {
          // Silent fail - portal already removed by another cleanup
          // This is expected in race conditions and not an error
          if (process.env.NODE_ENV === 'development') {
            console.debug('Portal cleanup: already removed (expected in fast close/open cycles)');
          }
        }
      } else {
        // Portal not mounted or not in body - just update state
        isMountedRef.current = false;
      }
    };
  }, [isOpen]);

  return portalRef.current;
}
