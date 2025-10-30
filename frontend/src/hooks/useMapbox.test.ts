/**
 * useMapbox Hook Tests
 * 
 * Validates map initialization and prevents unnecessary recreations
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useMapbox } from './useMapbox';
import mapboxgl from 'mapbox-gl';

// Mock mapbox-gl
vi.mock('mapbox-gl', () => {
    const mockMap = {
        on: vi.fn(),
        addControl: vi.fn(),
        remove: vi.fn(),
    };

    return {
        default: {
            Map: vi.fn(() => mockMap),
            NavigationControl: vi.fn(),
            FullscreenControl: vi.fn(),
            ScaleControl: vi.fn(),
            accessToken: '',
        },
        Map: vi.fn(() => mockMap),
        NavigationControl: vi.fn(),
        FullscreenControl: vi.fn(),
        ScaleControl: vi.fn(),
    };
});

describe('useMapbox', () => {
    let mockContainer: HTMLDivElement;

    beforeEach(() => {
        mockContainer = document.createElement('div');
        vi.clearAllMocks();
        
        // Set a valid token to avoid error state
        vi.stubEnv('VITE_MAPBOX_TOKEN', 'pk.test_token_12345');
    });

    afterEach(() => {
        vi.clearAllMocks();
    });

    it('should initialize map when container is provided', () => {
        renderHook(() =>
            useMapbox({
                container: mockContainer,
            })
        );

        // Verify map constructor was called
        expect(mapboxgl.Map).toHaveBeenCalledTimes(1);
        expect(mapboxgl.Map).toHaveBeenCalledWith(
            expect.objectContaining({
                container: mockContainer,
            })
        );
    });

    it('should not initialize map when container is null', () => {
        const { result } = renderHook(() =>
            useMapbox({
                container: null,
            })
        );

        expect(mapboxgl.Map).not.toHaveBeenCalled();
        expect(result.current.map).toBeNull();
    });

    it('should not recreate map on re-render with same container', () => {
        const { rerender } = renderHook(
            ({ container }) => useMapbox({ container }),
            {
                initialProps: { container: mockContainer },
            }
        );

        expect(mapboxgl.Map).toHaveBeenCalledTimes(1);

        // Re-render with same container
        rerender({ container: mockContainer });

        // Map should not be recreated
        expect(mapboxgl.Map).toHaveBeenCalledTimes(1);
    });

    it('should set error when token is missing', () => {
        vi.stubEnv('VITE_MAPBOX_TOKEN', '');

        const { result } = renderHook(() =>
            useMapbox({
                container: mockContainer,
            })
        );

        expect(result.current.error).toBeTruthy();
        expect(result.current.error).toContain('token not configured');
        expect(mapboxgl.Map).not.toHaveBeenCalled();
    });

    it('should set error when token is placeholder', () => {
        vi.stubEnv('VITE_MAPBOX_TOKEN', 'your_mapbox_api_token_here');

        const { result } = renderHook(() =>
            useMapbox({
                container: mockContainer,
            })
        );

        expect(result.current.error).toBeTruthy();
        expect(result.current.error).toContain('token not configured');
        expect(mapboxgl.Map).not.toHaveBeenCalled();
    });

    it('should use custom access token from config', () => {
        const customToken = 'pk.custom_token_xyz';
        
        renderHook(() =>
            useMapbox({
                container: mockContainer,
                accessToken: customToken,
            })
        );

        expect(mapboxgl.accessToken).toBe(customToken);
    });

    it('should add navigation, fullscreen, and scale controls', () => {
        renderHook(() =>
            useMapbox({
                container: mockContainer,
            })
        );

        // Verify all three controls were instantiated
        expect(mapboxgl.NavigationControl).toHaveBeenCalledTimes(1);
        expect(mapboxgl.FullscreenControl).toHaveBeenCalledTimes(1);
        expect(mapboxgl.ScaleControl).toHaveBeenCalledTimes(1);
    });

    it('should cleanup map on unmount', () => {
        const mockMap = {
            on: vi.fn(),
            addControl: vi.fn(),
            remove: vi.fn(),
        };
        
        (mapboxgl.Map as any).mockImplementation(() => mockMap);

        const { unmount } = renderHook(() =>
            useMapbox({
                container: mockContainer,
            })
        );

        unmount();

        expect(mockMap.remove).toHaveBeenCalledTimes(1);
    });
});
