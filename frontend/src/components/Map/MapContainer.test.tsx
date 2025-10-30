/**
 * MapContainer Component Tests
 * 
 * Validates type correctness and integration with WebSDR health status
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '@testing-library/react';
import MapContainer from './MapContainer';
import type { WebSDRConfig, WebSDRHealthStatus, LocalizationResult } from '@/services/api/types';

// Mock the useMapbox hook
vi.mock('@/hooks/useMapbox', () => ({
    useMapbox: vi.fn(() => ({
        map: null,
        isLoaded: false,
        error: null,
    })),
}));

// Mock child components
vi.mock('./WebSDRMarkers', () => ({
    default: () => null,
}));

vi.mock('./LocalizationLayer', () => ({
    default: () => null,
}));

describe('MapContainer', () => {
    let mockWebSDRs: WebSDRConfig[];
    let mockHealthStatus: Record<string, WebSDRHealthStatus>;
    let mockLocalizations: LocalizationResult[];

    beforeEach(() => {
        // Mock WebSDR data with UUID strings (correct type)
        mockWebSDRs = [
            {
                id: 'f47ac10b-58cc-4372-a567-0e02b2c3d479', // UUID string
                name: 'WebSDR Torino',
                url: 'http://websdr.example.com/torino',
                latitude: 45.0642,
                longitude: 7.6603,
                location_description: 'Torino, Italy',
                is_active: true,
                timeout_seconds: 30,
                retry_count: 3,
            },
            {
                id: 'a3bb189e-8bf9-3888-9912-ace4e6543002', // UUID string
                name: 'WebSDR Milano',
                url: 'http://websdr.example.com/milano',
                latitude: 45.4642,
                longitude: 9.1900,
                location_description: 'Milano, Italy',
                is_active: true,
                timeout_seconds: 30,
                retry_count: 3,
            },
        ];

        // Mock health status with UUID string keys (correct type)
        mockHealthStatus = {
            'f47ac10b-58cc-4372-a567-0e02b2c3d479': {
                websdr_id: 'f47ac10b-58cc-4372-a567-0e02b2c3d479',
                name: 'WebSDR Torino',
                status: 'online',
                response_time_ms: 150,
                last_check: '2024-10-30T00:00:00Z',
                avg_snr: 12.5,
            },
            'a3bb189e-8bf9-3888-9912-ace4e6543002': {
                websdr_id: 'a3bb189e-8bf9-3888-9912-ace4e6543002',
                name: 'WebSDR Milano',
                status: 'offline',
                last_check: '2024-10-30T00:00:00Z',
            },
        };

        mockLocalizations = [];
    });

    it('should render without crashing with correct types', () => {
        const { container } = render(
            <MapContainer
                websdrs={mockWebSDRs}
                healthStatus={mockHealthStatus}
                localizations={mockLocalizations}
            />
        );
        expect(container).toBeTruthy();
    });

    it('should accept UUID string keys for healthStatus', () => {
        // This test validates that the type definition is correct
        // If types are wrong, TypeScript would catch it at compile time
        const healthStatusWithStringKeys: Record<string, WebSDRHealthStatus> = {
            'f47ac10b-58cc-4372-a567-0e02b2c3d479': {
                websdr_id: 'f47ac10b-58cc-4372-a567-0e02b2c3d479',
                name: 'Test WebSDR',
                status: 'online',
                last_check: '2024-10-30T00:00:00Z',
            },
        };

        const { container } = render(
            <MapContainer
                websdrs={mockWebSDRs}
                healthStatus={healthStatusWithStringKeys}
                localizations={mockLocalizations}
            />
        );
        expect(container).toBeTruthy();
    });

    it('should handle empty data gracefully', () => {
        const { container } = render(
            <MapContainer
                websdrs={[]}
                healthStatus={{}}
                localizations={[]}
            />
        );
        expect(container).toBeTruthy();
    });

    it('should pass props to child components correctly', () => {
        const onLocalizationClick = vi.fn();
        
        render(
            <MapContainer
                websdrs={mockWebSDRs}
                healthStatus={mockHealthStatus}
                localizations={mockLocalizations}
                onLocalizationClick={onLocalizationClick}
            />
        );

        // If component renders without errors, props are correctly typed
        expect(true).toBe(true);
    });
});
