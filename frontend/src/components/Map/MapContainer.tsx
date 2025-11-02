/**
 * MapContainer Component
 *
 * Base Mapbox GL JS map component with WebSDR markers and localization display
 */

import React, { useRef } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import './Map.css';
import { useMapbox } from '@/hooks/useMapbox';
import WebSDRMarkers from './WebSDRMarkers';
import LocalizationLayer from './LocalizationLayer';
import type { WebSDRConfig, WebSDRHealthStatus } from '@/services/api/types';
import type { LocalizationResult } from '@/services/api/types';

export interface MapContainerProps {
  websdrs: WebSDRConfig[];
  healthStatus: Record<string, WebSDRHealthStatus>; // UUID string keys
  localizations: LocalizationResult[];
  onLocalizationClick?: (localization: LocalizationResult) => void;
  style?: React.CSSProperties;
  className?: string;
  mapStyle?: string; // Mapbox style URL
  fitBoundsOnLoad?: boolean; // Auto-fit bounds to show all SDRs on initial load
}

const MapContainer: React.FC<MapContainerProps> = ({
  websdrs,
  healthStatus,
  localizations,
  onLocalizationClick,
  style,
  className = '',
  mapStyle,
  fitBoundsOnLoad = false,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const [containerReady, setContainerReady] = React.useState(false);
  const hasFitBounds = useRef(false);

  // Wait for container to be ready
  React.useEffect(() => {
    if (mapContainerRef.current) {
      setContainerReady(true);
    }
  }, []);

  const { map, isLoaded, error } = useMapbox({
    container: containerReady && mapContainerRef.current ? mapContainerRef.current : null,
    style: mapStyle,
  });

  // Auto-fit bounds to show all SDR receivers on initial load
  React.useEffect(() => {
    if (!map || !isLoaded || !fitBoundsOnLoad || hasFitBounds.current || websdrs.length === 0) {
      return;
    }

    // Calculate bounds from WebSDR coordinates
    const bounds = new mapboxgl.LngLatBounds();
    websdrs.forEach(websdr => {
      bounds.extend([websdr.longitude, websdr.latitude]);
    });

    // Fit map to bounds with padding
    map.fitBounds(bounds, {
      padding: { top: 50, bottom: 50, left: 50, right: 50 },
      maxZoom: 10,
      duration: 1000,
    });

    hasFitBounds.current = true;
  }, [map, isLoaded, fitBoundsOnLoad, websdrs]);

  // Show error if map fails to load
  if (error) {
    return (
      <div
        className={`d-flex align-items-center justify-content-center bg-body-secondary ${className}`}
        style={style || { height: '500px' }}
      >
        <div className="alert alert-warning mb-0 m-3">
          <i className="ph ph-warning me-2"></i>
          <strong>Map Error:</strong> {error}
          <br />
          <small className="text-muted">
            Please configure VITE_MAPBOX_TOKEN in your .env file.
          </small>
        </div>
      </div>
    );
  }

  return (
    <div className={`position-relative ${className}`} style={style || { height: '500px' }}>
      {/* Map container */}
      <div ref={mapContainerRef} className="w-100 h-100" />

      {/* Loading overlay */}
      {!isLoaded && (
        <div className="position-absolute top-0 start-0 w-100 h-100 d-flex align-items-center justify-content-center bg-dark bg-opacity-50">
          <div className="spinner-border text-light" role="status">
            <span className="visually-hidden">Loading map...</span>
          </div>
        </div>
      )}

      {/* Render map layers when loaded */}
      {isLoaded && map && (
        <>
          <WebSDRMarkers map={map} websdrs={websdrs} healthStatus={healthStatus} />
          <LocalizationLayer
            map={map}
            localizations={localizations}
            onLocalizationClick={onLocalizationClick}
            totalWebSDRs={websdrs.length}
          />
        </>
      )}
    </div>
  );
};

export default MapContainer;
