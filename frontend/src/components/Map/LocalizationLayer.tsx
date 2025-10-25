/**
 * LocalizationLayer Component
 * 
 * Displays localization results with uncertainty ellipses on the map
 */

import { useEffect, useRef, useCallback } from 'react';
import mapboxgl from 'mapbox-gl';
import type { LocalizationResult } from '@/services/api/types';
import { createCircularUncertainty, createConfidenceEllipses, getConfidenceColor, getConfidenceOpacity } from '@/utils/ellipse';

export interface LocalizationLayerProps {
    map: mapboxgl.Map;
    localizations: LocalizationResult[];
    onLocalizationClick?: (localization: LocalizationResult) => void;
    maxPoints?: number; // Maximum number of points to display (default: 100)
}

const LAYER_IDS = {
    ELLIPSE_3SIGMA: 'localization-ellipse-3sigma',
    ELLIPSE_2SIGMA: 'localization-ellipse-2sigma',
    ELLIPSE_1SIGMA: 'localization-ellipse-1sigma',
    POINTS: 'localization-points',
};

const SOURCE_IDS = {
    ELLIPSES: 'localization-ellipses',
    POINTS: 'localization-points',
};

/**
 * Create popup HTML for localization result
 */
function createPopupHTML(localization: LocalizationResult): string {
    return `
        <div style="min-width: 250px;">
            <h6 class="mb-2">Localization Result</h6>
            <table class="table table-sm table-borderless mb-0">
                <tbody>
                    <tr>
                        <td class="text-muted">Timestamp:</td>
                        <td>${new Date(localization.timestamp).toLocaleString()}</td>
                    </tr>
                    <tr>
                        <td class="text-muted">Latitude:</td>
                        <td>${localization.latitude.toFixed(6)}</td>
                    </tr>
                    <tr>
                        <td class="text-muted">Longitude:</td>
                        <td>${localization.longitude.toFixed(6)}</td>
                    </tr>
                    <tr>
                        <td class="text-muted">Uncertainty:</td>
                        <td>±${localization.uncertainty_m.toFixed(1)}m</td>
                    </tr>
                    <tr>
                        <td class="text-muted">Confidence:</td>
                        <td>${(localization.confidence * 100).toFixed(1)}%</td>
                    </tr>
                    <tr>
                        <td class="text-muted">Frequency:</td>
                        <td>${localization.source_frequency_mhz.toFixed(3)} MHz</td>
                    </tr>
                    <tr>
                        <td class="text-muted">SNR:</td>
                        <td>${localization.snr_avg_db.toFixed(1)} dB</td>
                    </tr>
                    <tr>
                        <td class="text-muted">Receivers:</td>
                        <td>${localization.websdr_count}/7</td>
                    </tr>
                </tbody>
            </table>
        </div>
    `;
}

const LocalizationLayer: React.FC<LocalizationLayerProps> = ({
    map,
    localizations,
    onLocalizationClick,
    maxPoints = 100,
}) => {
    const popupsRef = useRef<mapboxgl.Popup[]>([]);
    const layersInitialized = useRef(false);

    /**
     * Initialize map layers for ellipses and points
     */
    const initializeLayers = useCallback(() => {
        // Add ellipses source
        if (!map.getSource(SOURCE_IDS.ELLIPSES)) {
            map.addSource(SOURCE_IDS.ELLIPSES, {
                type: 'geojson',
                data: {
                    type: 'FeatureCollection',
                    features: [],
                },
            });
        }

        // Add points source
        if (!map.getSource(SOURCE_IDS.POINTS)) {
            map.addSource(SOURCE_IDS.POINTS, {
                type: 'geojson',
                data: {
                    type: 'FeatureCollection',
                    features: [],
                },
            });
        }

        // Add ellipse layers (3σ, 2σ, 1σ - in order from largest to smallest)
        if (!map.getLayer(LAYER_IDS.ELLIPSE_3SIGMA)) {
            map.addLayer({
                id: LAYER_IDS.ELLIPSE_3SIGMA,
                type: 'fill',
                source: SOURCE_IDS.ELLIPSES,
                filter: ['==', ['get', 'confidenceLevel'], 3],
                paint: {
                    'fill-color': getConfidenceColor(3),
                    'fill-opacity': getConfidenceOpacity(3),
                },
            });
        }

        if (!map.getLayer(LAYER_IDS.ELLIPSE_2SIGMA)) {
            map.addLayer({
                id: LAYER_IDS.ELLIPSE_2SIGMA,
                type: 'fill',
                source: SOURCE_IDS.ELLIPSES,
                filter: ['==', ['get', 'confidenceLevel'], 2],
                paint: {
                    'fill-color': getConfidenceColor(2),
                    'fill-opacity': getConfidenceOpacity(2),
                },
            });
        }

        if (!map.getLayer(LAYER_IDS.ELLIPSE_1SIGMA)) {
            map.addLayer({
                id: LAYER_IDS.ELLIPSE_1SIGMA,
                type: 'fill',
                source: SOURCE_IDS.ELLIPSES,
                filter: ['==', ['get', 'confidenceLevel'], 1],
                paint: {
                    'fill-color': getConfidenceColor(1),
                    'fill-opacity': getConfidenceOpacity(1),
                },
            });
        }

        // Add points layer
        if (!map.getLayer(LAYER_IDS.POINTS)) {
            map.addLayer({
                id: LAYER_IDS.POINTS,
                type: 'circle',
                source: SOURCE_IDS.POINTS,
                paint: {
                    'circle-radius': 8,
                    'circle-color': '#ef4444', // Red for localization points
                    'circle-stroke-width': 2,
                    'circle-stroke-color': '#ffffff',
                },
            });
        }

        // Add click handler for points
        map.on('click', LAYER_IDS.POINTS, (e) => {
            if (e.features && e.features.length > 0) {
                const feature = e.features[0];
                const localizationId = feature.properties?.id;
                const localization = localizations.find((loc) => loc.id === localizationId);

                if (localization) {
                    // Show popup
                    const popup = new mapboxgl.Popup({
                        closeButton: true,
                        closeOnClick: false,
                    })
                        .setLngLat([localization.longitude, localization.latitude])
                        .setHTML(createPopupHTML(localization))
                        .addTo(map);

                    popupsRef.current.push(popup);

                    // Call callback
                    if (onLocalizationClick) {
                        onLocalizationClick(localization);
                    }
                }
            }
        });

        // Change cursor on hover
        map.on('mouseenter', LAYER_IDS.POINTS, () => {
            map.getCanvas().style.cursor = 'pointer';
        });

        map.on('mouseleave', LAYER_IDS.POINTS, () => {
            map.getCanvas().style.cursor = '';
        });
    }, [map, localizations, onLocalizationClick]);

    /**
     * Update localization data on the map
     */
    const updateLocalizationData = useCallback(() => {
        // Limit to maxPoints most recent
        const recentLocalizations = localizations.slice(0, maxPoints);

        // Create ellipse features
        const ellipseFeatures = recentLocalizations.flatMap((loc) => {
            const ellipseParams = createCircularUncertainty(
                loc.latitude,
                loc.longitude,
                loc.uncertainty_m
            );
            return createConfidenceEllipses(ellipseParams, [1, 2, 3]).map((feature) => ({
                ...feature,
                properties: {
                    ...feature.properties,
                    id: loc.id,
                },
            }));
        });

        // Create point features
        const pointFeatures = recentLocalizations.map((loc) => ({
            type: 'Feature' as const,
            geometry: {
                type: 'Point' as const,
                coordinates: [loc.longitude, loc.latitude],
            },
            properties: {
                id: loc.id,
                timestamp: loc.timestamp,
                confidence: loc.confidence,
                uncertainty: loc.uncertainty_m,
            },
        }));

        // Update sources
        const ellipsesSource = map.getSource(SOURCE_IDS.ELLIPSES) as mapboxgl.GeoJSONSource;
        if (ellipsesSource) {
            ellipsesSource.setData({
                type: 'FeatureCollection',
                features: ellipseFeatures,
            });
        }

        const pointsSource = map.getSource(SOURCE_IDS.POINTS) as mapboxgl.GeoJSONSource;
        if (pointsSource) {
            pointsSource.setData({
                type: 'FeatureCollection',
                features: pointFeatures,
            });
        }

        // Auto-center on most recent localization if available
        if (recentLocalizations.length > 0) {
            const latest = recentLocalizations[0];
            // Only fly to if not already close
            const currentCenter = map.getCenter();
            const distance = Math.sqrt(
                Math.pow(currentCenter.lat - latest.latitude, 2) +
                    Math.pow(currentCenter.lng - latest.longitude, 2)
            );

            // If more than 0.1 degrees away, fly to the new location
            if (distance > 0.1) {
                map.flyTo({
                    center: [latest.longitude, latest.latitude],
                    zoom: Math.max(map.getZoom(), 10),
                    duration: 1000,
                });
            }
        }
    }, [map, localizations, maxPoints]);

    useEffect(() => {
        // Initialize layers once
        if (!layersInitialized.current) {
            initializeLayers();
            layersInitialized.current = true;
        }

        // Update data
        updateLocalizationData();

        // Cleanup popups on unmount
        return () => {
            popupsRef.current.forEach((popup) => popup.remove());
            popupsRef.current = [];
        };
    }, [initializeLayers, updateLocalizationData]);

    return null; // This component doesn't render anything directly
};

export default LocalizationLayer;
