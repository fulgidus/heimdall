/**
 * useMapbox Hook
 *
 * Custom React hook for initializing and managing Mapbox GL JS map instance.
 */

import { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';

export interface MapConfig {
  container: string | HTMLElement | null;
  style?: string;
  center?: [number, number];
  zoom?: number;
  accessToken?: string;
}

export interface UseMapboxResult {
  map: mapboxgl.Map | null;
  isLoaded: boolean;
  error: string | null;
}

const DEFAULT_STYLE = 'mapbox://styles/mapbox/satellite-v9';
const DEFAULT_CENTER: [number, number] = [9.0, 44.5]; // Northwestern Italy
const DEFAULT_ZOOM = 8;

/**
 * Initialize and manage a Mapbox GL JS map
 *
 * @param config - Map configuration
 * @returns Map instance, loading state, and error
 */
export function useMapbox(config: MapConfig): UseMapboxResult {
  const mapRef = useRef<mapboxgl.Map | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const initializationAttempted = useRef(false);

  useEffect(() => {
    // Don't initialize if container is not ready or already initialized
    if (!config.container || initializationAttempted.current) {
      return;
    }

    initializationAttempted.current = true;

    // Get access token from environment or config
    const token = config.accessToken || import.meta.env.VITE_MAPBOX_TOKEN;

    if (!token || token === 'your_mapbox_api_token_here') {
      setError('Mapbox access token not configured. Please set VITE_MAPBOX_TOKEN in .env file.');
      return;
    }

    // Set access token
    mapboxgl.accessToken = token;

    try {
      // Use provided style or default
      const mapStyle = config.style !== undefined ? config.style : DEFAULT_STYLE;

      // Initialize map
      const map = new mapboxgl.Map({
        container: config.container,
        style: mapStyle,
        center: config.center || DEFAULT_CENTER,
        zoom: config.zoom || DEFAULT_ZOOM,
        attributionControl: true,
      });

      // Add navigation controls
      map.addControl(new mapboxgl.NavigationControl(), 'top-right');

      // Add fullscreen control
      map.addControl(new mapboxgl.FullscreenControl(), 'top-right');

      // Add scale control
      map.addControl(new mapboxgl.ScaleControl(), 'bottom-left');

      // Handle load event
      map.on('load', () => {
        setIsLoaded(true);
        setError(null);
      });

      // Handle error event
      map.on('error', e => {
        console.error('Mapbox error:', e);
        setError(e.error?.message || 'Map error occurred');
      });

      mapRef.current = map;

      // Cleanup function
      return () => {
        if (mapRef.current) {
          mapRef.current.remove();
          mapRef.current = null;
        }
        setIsLoaded(false);
        initializationAttempted.current = false;
      };
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to initialize map';
      setError(errorMessage);
      console.error('Failed to initialize Mapbox:', err);
      initializationAttempted.current = false;
    }
    // Only depend on container and style - center, zoom, and accessToken are captured at initialization
    // and don't need to trigger map recreation
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.container, config.style]);

  return {
    map: mapRef.current,
    isLoaded,
    error,
  };
}

export default useMapbox;
