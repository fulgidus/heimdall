import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useMediaQuery, useIsMobile, useIsTablet, useIsDesktop } from '../useMediaQuery';

describe('useMediaQuery Hook', () => {
    let matchMediaMock: any;

    beforeEach(() => {
        matchMediaMock = {
            matches: false,
            media: '',
            addEventListener: vi.fn(),
            removeEventListener: vi.fn(),
            addListener: vi.fn(),
            removeListener: vi.fn(),
            dispatchEvent: vi.fn(),
            onchange: null,
        };

        window.matchMedia = vi.fn((query) => ({
            ...matchMediaMock,
            media: query,
        }));
    });

    it('returns false when media query does not match', () => {
        matchMediaMock.matches = false;

        const { result } = renderHook(() => useMediaQuery('(max-width: 768px)'));

        expect(result.current).toBe(false);
    });

    it('returns true when media query matches', () => {
        matchMediaMock.matches = true;

        const { result } = renderHook(() => useMediaQuery('(max-width: 768px)'));

        expect(result.current).toBe(true);
    });

    it('adds event listener for media query changes', () => {
        renderHook(() => useMediaQuery('(max-width: 768px)'));

        expect(matchMediaMock.addEventListener).toHaveBeenCalledWith('change', expect.any(Function));
    });

    it('removes event listener on unmount', () => {
        const { unmount } = renderHook(() => useMediaQuery('(max-width: 768px)'));

        unmount();

        expect(matchMediaMock.removeEventListener).toHaveBeenCalledWith(
            'change',
            expect.any(Function)
        );
    });

    it('updates when window is resized and query changes', () => {
        let changeListener: any;

        matchMediaMock.addEventListener = vi.fn((event, listener) => {
            changeListener = listener;
        });

        const { result } = renderHook(() => useMediaQuery('(max-width: 768px)'));

        // Initial state
        expect(result.current).toBe(false);

        // Simulate resize that matches query
        act(() => {
            matchMediaMock.matches = true;
            if (changeListener) {
                changeListener({ matches: true });
            }
        });

        // Verify the state was updated by the listener
        expect(result.current).toBe(true);
    });

    it('handles different media queries', () => {
        const queries = [
            '(max-width: 768px)',
            '(min-width: 1024px)',
            '(orientation: portrait)',
            '(prefers-color-scheme: dark)',
        ];

        queries.forEach((query) => {
            const { result } = renderHook(() => useMediaQuery(query));
            expect(typeof result.current).toBe('boolean');
        });
    });
});

describe('useIsMobile Hook', () => {
    let matchMediaMock: any;

    beforeEach(() => {
        matchMediaMock = {
            matches: false,
            media: '',
            addEventListener: vi.fn(),
            removeEventListener: vi.fn(),
            addListener: vi.fn(),
            removeListener: vi.fn(),
            dispatchEvent: vi.fn(),
            onchange: null,
        };

        window.matchMedia = vi.fn((query) => ({
            ...matchMediaMock,
            media: query,
        }));
    });

    it('returns true when viewport is mobile size', () => {
        matchMediaMock.matches = true;

        const { result } = renderHook(() => useIsMobile());

        expect(result.current).toBe(true);
    });

    it('returns false when viewport is not mobile size', () => {
        matchMediaMock.matches = false;

        const { result } = renderHook(() => useIsMobile());

        expect(result.current).toBe(false);
    });

    it('uses correct breakpoint for mobile (max-width: 768px)', () => {
        renderHook(() => useIsMobile());

        expect(window.matchMedia).toHaveBeenCalledWith('(max-width: 768px)');
    });
});

describe('useIsTablet Hook', () => {
    let matchMediaMock: any;

    beforeEach(() => {
        matchMediaMock = {
            matches: false,
            media: '',
            addEventListener: vi.fn(),
            removeEventListener: vi.fn(),
            addListener: vi.fn(),
            removeListener: vi.fn(),
            dispatchEvent: vi.fn(),
            onchange: null,
        };

        window.matchMedia = vi.fn((query) => ({
            ...matchMediaMock,
            media: query,
        }));
    });

    it('returns true when viewport is tablet size', () => {
        matchMediaMock.matches = true;

        const { result } = renderHook(() => useIsTablet());

        expect(result.current).toBe(true);
    });

    it('returns false when viewport is not tablet size', () => {
        matchMediaMock.matches = false;

        const { result } = renderHook(() => useIsTablet());

        expect(result.current).toBe(false);
    });

    it('uses correct breakpoint for tablet', () => {
        renderHook(() => useIsTablet());

        expect(window.matchMedia).toHaveBeenCalledWith('(min-width: 769px) and (max-width: 1024px)');
    });
});

describe('useIsDesktop Hook', () => {
    let matchMediaMock: any;

    beforeEach(() => {
        matchMediaMock = {
            matches: false,
            media: '',
            addEventListener: vi.fn(),
            removeEventListener: vi.fn(),
            addListener: vi.fn(),
            removeListener: vi.fn(),
            dispatchEvent: vi.fn(),
            onchange: null,
        };

        window.matchMedia = vi.fn((query) => ({
            ...matchMediaMock,
            media: query,
        }));
    });

    it('returns true when viewport is desktop size', () => {
        matchMediaMock.matches = true;

        const { result } = renderHook(() => useIsDesktop());

        expect(result.current).toBe(true);
    });

    it('returns false when viewport is not desktop size', () => {
        matchMediaMock.matches = false;

        const { result } = renderHook(() => useIsDesktop());

        expect(result.current).toBe(false);
    });

    it('uses correct breakpoint for desktop (min-width: 1025px)', () => {
        renderHook(() => useIsDesktop());

        expect(window.matchMedia).toHaveBeenCalledWith('(min-width: 1025px)');
    });
});
