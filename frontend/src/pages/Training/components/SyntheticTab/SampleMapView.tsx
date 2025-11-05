/**
 * SampleMapView Component
 * 
 * Displays a Mapbox map with:
 * - TX position (red marker)
 * - RX positions (blue markers)
 * - Path lines from TX to each RX with SNR tooltips
 * - Interactive tooltips with propagation details
 */

import React, { useEffect, useRef } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import type { SyntheticSample, ReceiverMetadata } from '../../types';

interface SampleMapViewProps {
    sample: SyntheticSample;
    selectedRxId?: string | null;
    onRxSelect?: (rxId: string) => void;
    height?: number;
}

// Set Mapbox access token (from environment or placeholder)
mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN || '';

export const SampleMapView: React.FC<SampleMapViewProps> = ({
    sample,
    selectedRxId,
    onRxSelect,
    height = 500
}) => {
    const mapContainerRef = useRef<HTMLDivElement>(null);
    const mapRef = useRef<mapboxgl.Map | null>(null);
    const markersRef = useRef<mapboxgl.Marker[]>([]);

    useEffect(() => {
        if (!mapContainerRef.current) return;

        // Initialize map
        const map = new mapboxgl.Map({
            container: mapContainerRef.current,
            style: 'mapbox://styles/mapbox/satellite-streets-v12',
            center: [sample.tx_lon, sample.tx_lat],
            zoom: 8
        });

        mapRef.current = map;

        map.on('load', () => {
            // Add TX marker (red)
            const txMarker = new mapboxgl.Marker({ color: '#dc3545' })
                .setLngLat([sample.tx_lon, sample.tx_lat])
                .setPopup(
                    new mapboxgl.Popup({ offset: 25 })
                        .setHTML(`
                            <div class="p-2">
                                <strong>Transmitter</strong><br/>
                                Position: ${sample.tx_lat.toFixed(6)}, ${sample.tx_lon.toFixed(6)}<br/>
                                Power: ${sample.tx_power_dbm.toFixed(1)} dBm<br/>
                                Frequency: ${(sample.frequency_hz / 1e6).toFixed(3)} MHz
                            </div>
                        `)
                )
                .addTo(map);

            markersRef.current.push(txMarker);

            // Add RX markers and path lines
            const receivers = sample.receivers as ReceiverMetadata[];
            receivers.forEach((rx) => {
                // Determine signal presence
                const hasSignal = rx.signal_present !== undefined ? rx.signal_present : (rx.snr_db > -20);
                const isSelected = selectedRxId === rx.rx_id;
                
                // Add path line from TX to RX
                const pathId = `path-${rx.rx_id}`;
                map.addSource(pathId, {
                    type: 'geojson',
                    data: {
                        type: 'Feature',
                        properties: {},
                        geometry: {
                            type: 'LineString',
                            coordinates: [
                                [sample.tx_lon, sample.tx_lat],
                                [rx.lon, rx.lat]
                            ]
                        }
                    }
                });

                // Style path line (green if signal present, red if absent, yellow if selected)
                map.addLayer({
                    id: pathId,
                    type: 'line',
                    source: pathId,
                    layout: {
                        'line-join': 'round',
                        'line-cap': 'round'
                    },
                    paint: {
                        'line-color': isSelected ? '#ffc107' : (hasSignal ? '#28a745' : '#dc3545'),
                        'line-width': isSelected ? 3 : 2,
                        'line-opacity': 0.6
                    }
                });

                // Create circular RX marker with color based on signal presence
                const markerEl = document.createElement('div');
                markerEl.style.width = '20px';
                markerEl.style.height = '20px';
                markerEl.style.borderRadius = '50%';
                markerEl.style.border = '2px solid white';
                markerEl.style.boxShadow = '0 2px 4px rgba(0,0,0,0.3)';
                markerEl.style.cursor = 'pointer';
                markerEl.style.backgroundColor = isSelected ? '#ffc107' : (hasSignal ? '#28a745' : '#dc3545');
                
                // Build popup content with conditional fields
                let popupContent = `
                    <div class="p-2" style="max-width: 300px;">
                        <strong>Receiver ${rx.rx_id}</strong><br/>
                        Position: ${rx.lat.toFixed(6)}, ${rx.lon.toFixed(6)}<br/>
                        Distance: ${rx.distance_km.toFixed(1)} km<br/>
                        SNR: ${rx.snr_db.toFixed(1)} dB<br/>
                        Signal: <strong>${hasSignal ? 'Present' : 'Absent'}</strong><br/>
                `;
                
                if (rx.rx_power_dbm !== undefined) {
                    popupContent += `Power: ${rx.rx_power_dbm.toFixed(1)} dBm<br/>`;
                }
                
                // Add propagation details if available
                const hasPropagation = rx.fspl_db !== undefined || rx.terrain_loss_db !== undefined;
                if (hasPropagation) {
                    popupContent += `<hr class="my-1"/><small class="text-muted"><strong>Propagation:</strong><br/>`;
                    if (rx.fspl_db !== undefined) popupContent += `FSPL: ${rx.fspl_db.toFixed(1)} dB<br/>`;
                    if (rx.terrain_loss_db !== undefined) popupContent += `Terrain: ${rx.terrain_loss_db.toFixed(1)} dB<br/>`;
                    if (rx.knife_edge_loss_db !== undefined) popupContent += `Knife-edge: ${rx.knife_edge_loss_db.toFixed(1)} dB<br/>`;
                    if (rx.atmospheric_absorption_db !== undefined) popupContent += `Atmospheric: ${rx.atmospheric_absorption_db.toFixed(1)} dB<br/>`;
                    if (rx.tropospheric_effect_db !== undefined) popupContent += `Tropospheric: ${rx.tropospheric_effect_db.toFixed(1)} dB<br/>`;
                    if (rx.sporadic_e_enhancement_db !== undefined) popupContent += `Sporadic-E: ${rx.sporadic_e_enhancement_db.toFixed(1)} dB<br/>`;
                    if (rx.polarization_loss_db !== undefined) popupContent += `Polarization: ${rx.polarization_loss_db.toFixed(1)} dB<br/>`;
                    popupContent += `</small>`;
                }
                
                popupContent += `</div>`;
                
                const rxMarker = new mapboxgl.Marker({ element: markerEl })
                    .setLngLat([rx.lon, rx.lat])
                    .setPopup(new mapboxgl.Popup({ offset: 25 }).setHTML(popupContent))
                    .addTo(map);

                // Add click handler for RX selection
                if (onRxSelect) {
                    markerEl.addEventListener('click', () => {
                        onRxSelect(rx.rx_id);
                    });
                }

                markersRef.current.push(rxMarker);
            });

            // Fit map to show all markers
            const bounds = new mapboxgl.LngLatBounds();
            bounds.extend([sample.tx_lon, sample.tx_lat]);
            receivers.forEach(rx => bounds.extend([rx.lon, rx.lat]));
            map.fitBounds(bounds, { padding: 50 });
        });

        // Cleanup
        return () => {
            markersRef.current.forEach(marker => marker.remove());
            markersRef.current = [];
            map.remove();
            mapRef.current = null;
        };
    }, [sample, selectedRxId, onRxSelect]);

    // Update marker and line colors when selection changes
    useEffect(() => {
        const map = mapRef.current;
        if (!map || !map.isStyleLoaded()) return;

        const receivers = sample.receivers as ReceiverMetadata[];
        
        // Update path lines and markers
        receivers.forEach((rx, idx) => {
            const pathId = `path-${rx.rx_id}`;
            const isSelected = selectedRxId === rx.rx_id;
            const hasSignal = rx.signal_present !== undefined ? rx.signal_present : (rx.snr_db > -20);

            // Update path line color
            if (map.getLayer(pathId)) {
                map.setPaintProperty(pathId, 'line-color', isSelected ? '#ffc107' : (hasSignal ? '#28a745' : '#dc3545'));
                map.setPaintProperty(pathId, 'line-width', isSelected ? 3 : 2);
            }
            
            // Update marker color (skip TX marker at index 0)
            const marker = markersRef.current[idx + 1];
            if (marker) {
                const markerEl = marker.getElement();
                markerEl.style.backgroundColor = isSelected ? '#ffc107' : (hasSignal ? '#28a745' : '#dc3545');
            }
        });
    }, [selectedRxId, sample.receivers]);

    return (
        <div
            ref={mapContainerRef}
            style={{
                width: '100%',
                height: `${height}px`,
                borderRadius: '4px',
                overflow: 'hidden'
            }}
        />
    );
};
