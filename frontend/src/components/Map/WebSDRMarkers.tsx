/**
 * WebSDRMarkers Component
 *
 * Displays WebSDR receiver locations on the map with status-based colors
 */

import { useEffect, useRef } from 'react';
import mapboxgl from 'mapbox-gl';
import type { WebSDRConfig, WebSDRHealthStatus } from '@/services/api/types';

export interface WebSDRMarkersProps {
  map: mapboxgl.Map;
  websdrs: WebSDRConfig[];
  healthStatus: Record<string, WebSDRHealthStatus>; // UUID keys
}

/**
 * Get marker color based on WebSDR health status
 */
function getStatusColor(status?: 'online' | 'offline' | 'unknown'): string {
  switch (status) {
    case 'online':
      return '#10b981'; // Green
    case 'offline':
      return '#ef4444'; // Red
    case 'unknown':
    default:
      return '#f59e0b'; // Yellow
  }
}

/**
 * Create popup HTML for WebSDR receiver
 */
function createPopupHTML(websdr: WebSDRConfig, health?: WebSDRHealthStatus): string {
  const status = health?.status || 'unknown';
  const statusBadge =
    status === 'online'
      ? '<span class="badge bg-success">Online</span>'
      : status === 'offline'
        ? '<span class="badge bg-danger">Offline</span>'
        : '<span class="badge bg-warning">Unknown</span>';

  return `
        <div style="min-width: 200px;">
            <h6 class="mb-2">${websdr.name}</h6>
            <div class="mb-2">${statusBadge}</div>
            <table class="table table-sm table-borderless mb-0">
                <tbody>
                    <tr>
                        <td class="text-muted">Location:</td>
                        <td>${websdr.location_description || websdr.name}</td>
                    </tr>
                    <tr>
                        <td class="text-muted">Coordinates:</td>
                        <td>${websdr.latitude != null ? websdr.latitude.toFixed(4) : 'N/A'}, ${websdr.longitude != null ? websdr.longitude.toFixed(4) : 'N/A'}</td>
                    </tr>
                    ${
                      health?.response_time_ms != null
                        ? `<tr>
                            <td class="text-muted">Response:</td>
                            <td>${health.response_time_ms}ms</td>
                        </tr>`
                        : ''
                    }
                    ${
                      health?.avg_snr != null
                        ? `<tr>
                            <td class="text-muted">Avg SNR:</td>
                            <td>${health.avg_snr.toFixed(1)} dB</td>
                        </tr>`
                        : ''
                    }
                </tbody>
            </table>
        </div>
    `;
}

const WebSDRMarkers: React.FC<WebSDRMarkersProps> = ({ map, websdrs, healthStatus }) => {
  const markersRef = useRef<mapboxgl.Marker[]>([]);

  useEffect(() => {
    // Add markers for each WebSDR
    websdrs.forEach(websdr => {
      const health = healthStatus[websdr.id];
      const color = getStatusColor(health?.status);

      // Create custom marker element
      const el = document.createElement('div');
      el.className = 'websdr-marker';
      el.style.width = '24px';
      el.style.height = '24px';
      el.style.borderRadius = '50%';
      el.style.backgroundColor = color;
      el.style.border = '2px solid white';
      el.style.boxShadow = '0 2px 4px rgba(0,0,0,0.3)';
      el.style.cursor = 'pointer';
      el.title = websdr.name;

      // Add pulsing animation for online receivers
      if (health?.status === 'online') {
        el.style.animation = 'pulse 2s infinite';
      }

      // Create popup
      const popup = new mapboxgl.Popup({
        offset: 25,
        closeButton: true,
        closeOnClick: false,
      }).setHTML(createPopupHTML(websdr, health));

      // Create marker
      const marker = new mapboxgl.Marker(el)
        .setLngLat([websdr.longitude, websdr.latitude])
        .setPopup(popup)
        .addTo(map);

      markersRef.current.push(marker);
    });

    // Cleanup function
    return () => {
      markersRef.current.forEach(marker => marker.remove());
      markersRef.current = [];
    };
  }, [map, websdrs, healthStatus]);

  return null; // This component doesn't render anything directly
};

export default WebSDRMarkers;
