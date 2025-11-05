/**
 * usePortal Hook
 * 
 * Bulletproof React portal management that prevents the infamous
 * "Node.removeChild: The node to be removed is not a child of this node" error.
 * 
 * ROOT CAUSE OF THE BUG:
 * When a modal closes and the component unmounts, React's cleanup functions
 * can run multiple times or be queued asynchronously:
 * 1. When isOpen changes from true → false (cleanup queued)
 * 2. When component unmounts (cleanup queued AGAIN)
 * 3. Rapid state changes can queue multiple cleanups
 * 4. Cleanup functions from previous effects may run after state has changed
 * 5. Using refs to track state creates race conditions because refs can be
 *    modified while cleanup functions are queued but not yet executed
 * 
 * SOLUTION - THE KEY INSIGHT:
 * NEVER trust cached state (refs, flags). ALWAYS check the actual DOM state.
 * - Check portal.parentNode === document.body before EVERY operation
 * - No isMountedRef (source of race conditions)
 * - No isCleaningUpRef (source of race conditions)
 * - The DOM itself is the source of truth
 * - appendChild only if not already appended (check parentNode)
 * - removeChild only if actually in DOM (check parentNode)
 * - Try-catch as final safety net (though should never trigger)
 * 
 * This eliminates ALL race conditions because we always check the current
 * DOM state at operation time, not cached state from when effect was scheduled.
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

  // Lazy initialization: create div once and reuse
  if (!portalRef.current) {
    portalRef.current = document.createElement('div');
    portalRef.current.setAttribute('data-portal', 'modal-portal');
  }

  useEffect(() => {
    const portal = portalRef.current;
    if (!portal) return;

    if (isOpen) {
      // Mount portal only if not already in DOM
      // Check actual DOM state, not flags - this is the source of truth
      if (portal.parentNode !== document.body) {
        try {
          document.body.appendChild(portal);
          document.body.style.overflow = 'hidden';
        } catch (error) {
          console.error('Failed to mount portal:', error);
        }
      } else {
        // Portal already mounted, just ensure body scroll is prevented
        document.body.style.overflow = 'hidden';
      }
    }

    // Cleanup function (runs on dependency change or unmount)
    return () => {
      // Restore body scroll
      document.body.style.overflow = '';

      // CRITICAL: Only remove if actually in DOM at cleanup time
      // Check the actual DOM state, not any cached flags
      // This eliminates ALL race conditions
      if (portal.parentNode === document.body) {
        try {
          document.body.removeChild(portal);
        } catch (error) {
          // Should theoretically never happen because we checked parentNode.
          // Could only occur if another cleanup removed the portal between
          // the check and removeChild call (extremely unlikely but possible
          // if multiple effects somehow share the same portal ref).
          if (process.env.NODE_ENV === 'development') {
            console.debug('Portal cleanup: removeChild failed despite parent check', error);
          }
        }
      }
    };
  }, [isOpen]);

  return portalRef.current;
}
