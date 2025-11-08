import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render } from '@testing-library/react';
import WebSDRMarkers from './WebSDRMarkers';
import type { WebSDRConfig, WebSDRHealthStatus } from '@/services/api/types';
import mapboxgl from 'mapbox-gl';

// Mock auth store
vi.mock('@/store', () => ({
  useAuthStore: {
    getState: vi.fn(() => ({ token: null })),
  },
}));

// Mock mapbox-gl
vi.mock('mapbox-gl', () => {
  const createMockMarker = () => ({
    setLngLat: vi.fn().mockReturnThis(),
    setPopup: vi.fn().mockReturnThis(),
    addTo: vi.fn().mockReturnThis(),
    remove: vi.fn(),
  });

  const createMockPopup = () => ({
    setHTML: vi.fn().mockReturnThis(),
  });

  return {
    default: {
      Marker: vi.fn(() => createMockMarker()),
      Popup: vi.fn(() => createMockPopup()),
    },
    Marker: vi.fn(() => createMockMarker()),
    Popup: vi.fn(() => createMockPopup()),
  };
});

describe('WebSDRMarkers', () => {
  let mockMap: any;
  let mockWebSDRs: WebSDRConfig[];
  let mockHealthStatus: Record<number, WebSDRHealthStatus>;

  beforeEach(() => {
    // Create mock map instance
    mockMap = {
      on: vi.fn(),
      off: vi.fn(),
      remove: vi.fn(),
    };

    // Mock WebSDR data
    mockWebSDRs = [
      {
        id: 1,
        name: 'WebSDR Torino',
        url: 'http://websdr.example.com/torino',
        latitude: 45.0642,
        longitude: 7.6603,
        location_description: 'Torino, Italy',
        active: true,
        frequency_range_mhz: '144-146',
      },
      {
        id: 2,
        name: 'WebSDR Milano',
        url: 'http://websdr.example.com/milano',
        latitude: 45.4642,
        longitude: 9.19,
        location_description: 'Milano, Italy',
        active: true,
        frequency_range_mhz: '144-146',
      },
      {
        id: 3,
        name: 'WebSDR Genova',
        url: 'http://websdr.example.com/genova',
        latitude: 44.4056,
        longitude: 8.9463,
        location_description: 'Genova, Italy',
        active: false,
        frequency_range_mhz: '144-146',
      },
    ];

    // Mock health status
    mockHealthStatus = {
      1: {
        id: 1,
        websdr_id: 1,
        status: 'online' as const,
        checked_at: '2025-01-15T10:00:00Z',
        response_time_ms: 150,
        avg_snr: 12.5,
      },
      2: {
        id: 2,
        websdr_id: 2,
        status: 'offline' as const,
        checked_at: '2025-01-15T10:00:00Z',
        response_time_ms: null,
        avg_snr: null,
      },
    };

    // Clear all mocks
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Marker Creation', () => {
    it('should create markers for all WebSDRs', () => {
      render(<WebSDRMarkers map={mockMap} websdrs={mockWebSDRs} healthStatus={mockHealthStatus} />);

      // Should create 3 markers (one for each WebSDR)
      expect(mapboxgl.Marker).toHaveBeenCalledTimes(3);
    });

    it('should create markers with correct coordinates', () => {
      const { rerender } = render(
        <WebSDRMarkers map={mockMap} websdrs={mockWebSDRs} healthStatus={mockHealthStatus} />
      );

      const markerInstances = (mapboxgl.Marker as any).mock.results.map((r: any) => r.value);

      // Check that setLngLat was called with correct coordinates
      expect(markerInstances[0].setLngLat).toHaveBeenCalledWith([7.6603, 45.0642]);
      expect(markerInstances[1].setLngLat).toHaveBeenCalledWith([9.19, 45.4642]);
      expect(markerInstances[2].setLngLat).toHaveBeenCalledWith([8.9463, 44.4056]);
    });

    it('should add markers to the map', () => {
      render(<WebSDRMarkers map={mockMap} websdrs={mockWebSDRs} healthStatus={mockHealthStatus} />);

      const markerInstances = (mapboxgl.Marker as any).mock.results.map((r: any) => r.value);

      markerInstances.forEach((marker: any) => {
        expect(marker.addTo).toHaveBeenCalledWith(mockMap);
      });
    });
  });

  describe('Marker Styling', () => {
    it('should style online markers with green color', () => {
      const onlineWebSDRs = [mockWebSDRs[0]]; // Only online WebSDR
      const onlineHealthStatus = { 1: mockHealthStatus[1] };

      render(
        <WebSDRMarkers map={mockMap} websdrs={onlineWebSDRs} healthStatus={onlineHealthStatus} />
      );

      // Verify Marker constructor was called with an element
      const markerCall = (mapboxgl.Marker as any).mock.calls[0];
      const element = markerCall[0] as HTMLElement;

      expect(element.style.backgroundColor).toBe('rgb(16, 185, 129)'); // Green in RGB
    });

    it('should style offline markers with red color', () => {
      const offlineWebSDRs = [mockWebSDRs[1]]; // Offline WebSDR
      const offlineHealthStatus = { 2: mockHealthStatus[2] };

      render(
        <WebSDRMarkers map={mockMap} websdrs={offlineWebSDRs} healthStatus={offlineHealthStatus} />
      );

      const markerCall = (mapboxgl.Marker as any).mock.calls[0];
      const element = markerCall[0] as HTMLElement;

      expect(element.style.backgroundColor).toBe('rgb(239, 68, 68)'); // Red in RGB
    });

    it('should style unknown status markers with yellow color', () => {
      const unknownWebSDRs = [mockWebSDRs[2]]; // No health status = unknown
      const emptyHealthStatus = {};

      render(
        <WebSDRMarkers map={mockMap} websdrs={unknownWebSDRs} healthStatus={emptyHealthStatus} />
      );

      const markerCall = (mapboxgl.Marker as any).mock.calls[0];
      const element = markerCall[0] as HTMLElement;

      expect(element.style.backgroundColor).toBe('rgb(245, 158, 11)'); // Yellow in RGB
    });

    it('should add pulse animation to online markers', () => {
      const onlineWebSDRs = [mockWebSDRs[0]];
      const onlineHealthStatus = { 1: mockHealthStatus[1] };

      render(
        <WebSDRMarkers map={mockMap} websdrs={onlineWebSDRs} healthStatus={onlineHealthStatus} />
      );

      const markerCall = (mapboxgl.Marker as any).mock.calls[0];
      const element = markerCall[0] as HTMLElement;

      expect(element.style.animation).toBe('pulse 2s infinite');
    });

    it('should not add pulse animation to offline markers', () => {
      const offlineWebSDRs = [mockWebSDRs[1]];
      const offlineHealthStatus = { 2: mockHealthStatus[2] };

      render(
        <WebSDRMarkers map={mockMap} websdrs={offlineWebSDRs} healthStatus={offlineHealthStatus} />
      );

      const markerCall = (mapboxgl.Marker as any).mock.calls[0];
      const element = markerCall[0] as HTMLElement;

      expect(element.style.animation).not.toBe('pulse 2s infinite');
    });
  });

  describe('Popup Content', () => {
    it('should create popup with WebSDR information', () => {
      render(<WebSDRMarkers map={mockMap} websdrs={mockWebSDRs} healthStatus={mockHealthStatus} />);

      const popupInstances = (mapboxgl.Popup as any).mock.results.map((r: any) => r.value);

      // Check that popup HTML was set
      expect(popupInstances[0].setHTML).toHaveBeenCalled();
    });

    it('should include WebSDR name in popup', () => {
      render(
        <WebSDRMarkers map={mockMap} websdrs={[mockWebSDRs[0]]} healthStatus={mockHealthStatus} />
      );

      const popupInstances = (mapboxgl.Popup as any).mock.results.map((r: any) => r.value);
      const htmlContent = popupInstances[0].setHTML.mock.calls[0][0];

      expect(htmlContent).toContain('WebSDR Torino');
    });

    it('should include location name in popup', () => {
      render(
        <WebSDRMarkers map={mockMap} websdrs={[mockWebSDRs[0]]} healthStatus={mockHealthStatus} />
      );

      const popupInstances = (mapboxgl.Popup as any).mock.results.map((r: any) => r.value);
      const htmlContent = popupInstances[0].setHTML.mock.calls[0][0];

      expect(htmlContent).toContain('Torino, Italy');
    });

    it('should include coordinates in popup', () => {
      render(
        <WebSDRMarkers map={mockMap} websdrs={[mockWebSDRs[0]]} healthStatus={mockHealthStatus} />
      );

      const popupInstances = (mapboxgl.Popup as any).mock.results.map((r: any) => r.value);
      const htmlContent = popupInstances[0].setHTML.mock.calls[0][0];

      expect(htmlContent).toContain('45.0642');
      expect(htmlContent).toContain('7.6603');
    });

    it('should include response time when available', () => {
      render(
        <WebSDRMarkers map={mockMap} websdrs={[mockWebSDRs[0]]} healthStatus={mockHealthStatus} />
      );

      const popupInstances = (mapboxgl.Popup as any).mock.results.map((r: any) => r.value);
      const htmlContent = popupInstances[0].setHTML.mock.calls[0][0];

      expect(htmlContent).toContain('150ms');
    });

    it('should include SNR when available', () => {
      render(
        <WebSDRMarkers map={mockMap} websdrs={[mockWebSDRs[0]]} healthStatus={mockHealthStatus} />
      );

      const popupInstances = (mapboxgl.Popup as any).mock.results.map((r: any) => r.value);
      const htmlContent = popupInstances[0].setHTML.mock.calls[0][0];

      expect(htmlContent).toContain('12.5 dB');
    });

    it('should not include response time when null', () => {
      render(
        <WebSDRMarkers map={mockMap} websdrs={[mockWebSDRs[1]]} healthStatus={mockHealthStatus} />
      );

      const popupInstances = (mapboxgl.Popup as any).mock.results.map((r: any) => r.value);
      const htmlContent = popupInstances[0].setHTML.mock.calls[0][0];

      expect(htmlContent).not.toContain('Response:');
    });

    it('should display online status badge', () => {
      render(
        <WebSDRMarkers map={mockMap} websdrs={[mockWebSDRs[0]]} healthStatus={mockHealthStatus} />
      );

      const popupInstances = (mapboxgl.Popup as any).mock.results.map((r: any) => r.value);
      const htmlContent = popupInstances[0].setHTML.mock.calls[0][0];

      expect(htmlContent).toContain('bg-success');
      expect(htmlContent).toContain('Online');
    });

    it('should display offline status badge', () => {
      render(
        <WebSDRMarkers map={mockMap} websdrs={[mockWebSDRs[1]]} healthStatus={mockHealthStatus} />
      );

      const popupInstances = (mapboxgl.Popup as any).mock.results.map((r: any) => r.value);
      const htmlContent = popupInstances[0].setHTML.mock.calls[0][0];

      expect(htmlContent).toContain('bg-danger');
      expect(htmlContent).toContain('Offline');
    });

    it('should display unknown status badge when no health data', () => {
      render(<WebSDRMarkers map={mockMap} websdrs={[mockWebSDRs[2]]} healthStatus={{}} />);

      const popupInstances = (mapboxgl.Popup as any).mock.results.map((r: any) => r.value);
      const htmlContent = popupInstances[0].setHTML.mock.calls[0][0];

      expect(htmlContent).toContain('bg-warning');
      expect(htmlContent).toContain('Unknown');
    });
  });

  describe('Marker Updates', () => {
    it('should remove old markers when WebSDRs change', () => {
      const { rerender } = render(
        <WebSDRMarkers map={mockMap} websdrs={mockWebSDRs} healthStatus={mockHealthStatus} />
      );

      const firstMarkers = (mapboxgl.Marker as any).mock.results.map((r: any) => r.value);

      // Update with different WebSDRs
      const newWebSDRs = [mockWebSDRs[0]];
      rerender(
        <WebSDRMarkers map={mockMap} websdrs={newWebSDRs} healthStatus={mockHealthStatus} />
      );

      // Old markers should be removed
      firstMarkers.forEach((marker: any) => {
        expect(marker.remove).toHaveBeenCalled();
      });
    });

    it('should update marker colors when health status changes', () => {
      const { rerender } = render(
        <WebSDRMarkers map={mockMap} websdrs={[mockWebSDRs[0]]} healthStatus={mockHealthStatus} />
      );

      // Initial marker should be green (online)
      let markerCall = (mapboxgl.Marker as any).mock.calls[0];
      let element = markerCall[0] as HTMLElement;
      expect(element.style.backgroundColor).toBe('rgb(16, 185, 129)');

      // Change health status to offline
      const updatedHealthStatus = {
        1: {
          ...mockHealthStatus[1],
          status: 'offline' as const,
        },
      };

      vi.clearAllMocks();

      rerender(
        <WebSDRMarkers
          map={mockMap}
          websdrs={[mockWebSDRs[0]]}
          healthStatus={updatedHealthStatus}
        />
      );

      // New marker should be red (offline)
      markerCall = (mapboxgl.Marker as any).mock.calls[0];
      element = markerCall[0] as HTMLElement;
      expect(element.style.backgroundColor).toBe('rgb(239, 68, 68)');
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty WebSDR list', () => {
      render(<WebSDRMarkers map={mockMap} websdrs={[]} healthStatus={{}} />);

      // Should not create any markers
      expect(mapboxgl.Marker).not.toHaveBeenCalled();
    });

    it('should handle WebSDR with no health status', () => {
      render(<WebSDRMarkers map={mockMap} websdrs={[mockWebSDRs[0]]} healthStatus={{}} />);

      // Should create marker with unknown (yellow) color
      const markerCall = (mapboxgl.Marker as any).mock.calls[0];
      const element = markerCall[0] as HTMLElement;
      expect(element.style.backgroundColor).toBe('rgb(245, 158, 11)');
    });

    it('should handle WebSDR with null SNR', () => {
      const healthWithNullSNR = {
        1: {
          ...mockHealthStatus[1],
          avg_snr: null,
        },
      };

      render(
        <WebSDRMarkers map={mockMap} websdrs={[mockWebSDRs[0]]} healthStatus={healthWithNullSNR} />
      );

      const popupInstances = (mapboxgl.Popup as any).mock.results.map((r: any) => r.value);
      const htmlContent = popupInstances[0].setHTML.mock.calls[0][0];

      expect(htmlContent).not.toContain('Avg SNR:');
    });

    it('should handle WebSDR with zero SNR', () => {
      const healthWithZeroSNR = {
        1: {
          ...mockHealthStatus[1],
          avg_snr: 0,
        },
      };

      render(
        <WebSDRMarkers map={mockMap} websdrs={[mockWebSDRs[0]]} healthStatus={healthWithZeroSNR} />
      );

      const popupInstances = (mapboxgl.Popup as any).mock.results.map((r: any) => r.value);
      const htmlContent = popupInstances[0].setHTML.mock.calls[0][0];

      expect(htmlContent).toContain('0.0 dB');
    });
  });

  describe('Cleanup', () => {
    it('should remove all markers on unmount', () => {
      const { unmount } = render(
        <WebSDRMarkers map={mockMap} websdrs={mockWebSDRs} healthStatus={mockHealthStatus} />
      );

      const markerInstances = (mapboxgl.Marker as any).mock.results.map((r: any) => r.value);

      unmount();

      // All markers should be removed
      markerInstances.forEach((marker: any) => {
        expect(marker.remove).toHaveBeenCalled();
      });
    });
  });

  describe('Rendering', () => {
    it('should return null (no visual component)', () => {
      const { container } = render(
        <WebSDRMarkers map={mockMap} websdrs={mockWebSDRs} healthStatus={mockHealthStatus} />
      );

      // Component should not render any DOM elements
      expect(container.firstChild).toBeNull();
    });
  });
});
