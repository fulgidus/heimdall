/**
 * MapContainer Component
 * 
 * Base Mapbox GL JS map component with WebSDR markers and localization display
 */

import React, { useRef } from 'react';
import 'mapbox-gl/dist/mapbox-gl.css';
import './Map.css';
import { useMapbox } from '@/hooks/useMapbox';
import WebSDRMarkers from './WebSDRMarkers';
import LocalizationLayer from './LocalizationLayer';
import type { WebSDRConfig, WebSDRHealthStatus } from '@/services/api/types';
import type { LocalizationResult } from '@/services/api/types';

export interface MapContainerProps {
    websdrs: WebSDRConfig[];
    healthStatus: Record<number, WebSDRHealthStatus>;
    localizations: LocalizationResult[];
    onLocalizationClick?: (localization: LocalizationResult) => void;
    style?: React.CSSProperties;
    className?: string;
}

const MapContainer: React.FC<MapContainerProps> = ({
    websdrs,
    healthStatus,
    localizations,
    onLocalizationClick,
    style,
    className = '',
}) => {
    const mapContainerRef = useRef<HTMLDivElement>(null);
    const { map, isLoaded, error } = useMapbox({
        container: mapContainerRef.current!,
    });

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
                    <WebSDRMarkers
                        map={map}
                        websdrs={websdrs}
                        healthStatus={healthStatus}
                    />
                    <LocalizationLayer
                        map={map}
                        localizations={localizations}
                        onLocalizationClick={onLocalizationClick}
                    />
                </>
            )}
        </div>
    );
};

export default MapContainer;
