import React, { useEffect, useState, useRef } from 'react';
import { useSessionStore } from '../store/sessionStore';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import type { KnownSource, KnownSourceCreate, KnownSourceUpdate } from '../services/api/session';

// Set Mapbox access token
mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN || '';

// Add pulse animation for temporary marker
const style = document.createElement('style');
style.textContent = `
  @keyframes pulse {
    0%, 100% {
      opacity: 1;
      transform: scale(1);
    }
    50% {
      opacity: 0.7;
      transform: scale(1.1);
    }
  }
`;
document.head.appendChild(style);

interface SourceFormData {
    name: string;
    description: string;
    frequency_hz: string;
    latitude: string;
    longitude: string;
    power_dbm: string;
    source_type: string;
    is_validated: boolean;
    error_margin_meters: string;
}

const SourcesManagement: React.FC = () => {
    const {
        knownSources,
        fetchKnownSources,
        createKnownSource,
        updateKnownSource,
        deleteKnownSource,
        isLoading,
        error: storeError,
        clearError,
    } = useSessionStore();

    const mapContainer = useRef<HTMLDivElement>(null);
    const map = useRef<mapboxgl.Map | null>(null);
    const markers = useRef<{ [key: string]: mapboxgl.Marker }>({});
    const tempMarker = useRef<mapboxgl.Marker | null>(null);
    const [mapLoaded, setMapLoaded] = useState(false);

    const [isFormVisible, setIsFormVisible] = useState(false);
    const [editingSource, setEditingSource] = useState<KnownSource | null>(null);
    const [selectedSource, setSelectedSource] = useState<KnownSource | null>(null);
    const [formData, setFormData] = useState<SourceFormData>({
        name: '',
        description: '',
        frequency_hz: '',
        latitude: '',
        longitude: '',
        power_dbm: '',
        source_type: 'beacon',
        is_validated: false,
        error_margin_meters: '50',
    });
    const [formErrors, setFormErrors] = useState<Partial<SourceFormData>>({});
    const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
    const [notification, setNotification] = useState<{ type: 'success' | 'error'; message: string } | null>(null);
    const [mapError, setMapError] = useState<string | null>(null);

    // Initialize map
    useEffect(() => {
        if (!mapContainer.current || map.current) return;

        try {
            map.current = new mapboxgl.Map({
                container: mapContainer.current,
                style: 'mapbox://styles/mapbox/dark-v11',
                center: [12.4964, 41.9028], // Rome, Italy (center of coverage area)
                zoom: 5,
            });

            map.current.on('load', () => {
                setMapLoaded(true);

                // Add click handler for setting new source location
                map.current?.on('click', (e) => {
                    if (isFormVisible && !editingSource) {
                        // Remove previous temp marker
                        if (tempMarker.current) {
                            tempMarker.current.remove();
                        }

                        // Update form data
                        setFormData((prev) => ({
                            ...prev,
                            latitude: e.lngLat.lat.toFixed(6),
                            longitude: e.lngLat.lng.toFixed(6),
                        }));

                        // Create temporary marker
                        const el = document.createElement('div');
                        el.style.width = '30px';
                        el.style.height = '30px';
                        el.style.borderRadius = '50%';
                        el.style.backgroundColor = '#3b82f6';
                        el.style.border = '3px solid white';
                        el.style.boxShadow = '0 2px 8px rgba(59, 130, 246, 0.5)';
                        el.style.animation = 'pulse 2s infinite';

                        tempMarker.current = new mapboxgl.Marker({ element: el })
                            .setLngLat([e.lngLat.lng, e.lngLat.lat])
                            .addTo(map.current!);
                    }
                });
            });

            map.current?.on('error', (err) => {
                console.error('Mapbox error:', err);
                setMapError('Failed to initialize map. WebGL may not be supported in your browser.');
                setMapLoaded(false);
            });
        } catch (error) {
            console.error('Mapbox initialization error:', error);
            setMapError('Failed to initialize map. WebGL may not be supported in your browser.');
            setMapLoaded(false);
        }

        return () => {
            map.current?.remove();
            map.current = null;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Fetch sources on mount
    useEffect(() => {
        fetchKnownSources();
    }, [fetchKnownSources]);

    // Update markers when sources change
    useEffect(() => {
        if (!map.current || !mapLoaded) return;

        const updateMarkers = async () => {
            // Remove old markers
            Object.values(markers.current).forEach((marker) => marker.remove());
            markers.current = {};

            // Add new markers
            knownSources.forEach((source) => {
                // Skip sources without coordinates
                if (source.latitude == null || source.longitude == null) {
                    return;
                }

                // Create marker element
                const el = document.createElement('div');
                el.className = 'source-marker';
                el.style.width = '30px';
                el.style.height = '30px';
                el.style.borderRadius = '50%';
                el.style.backgroundColor = source.is_validated ? '#10b981' : '#f59e0b';
                el.style.border = '3px solid white';
                el.style.cursor = 'pointer';
                el.style.boxShadow = '0 2px 4px rgba(0,0,0,0.3)';

                // Create marker
                const marker = new mapboxgl.Marker({
                    element: el,
                    draggable: true,
                })
                    .setLngLat([source.longitude, source.latitude])
                    .addTo(map.current!);

                // Add popup with conditional frequency display
                const frequencyHtml = source.frequency_hz
                    ? `<p style="margin: 4px 0; font-size: 12px;">
                           <strong>Frequency:</strong> ${(source.frequency_hz / 1e6).toFixed(3)} MHz
                       </p>`
                    : '';

                const popup = new mapboxgl.Popup({ offset: 25 }).setHTML(`
                    <div style="padding: 8px;">
                        <h6 style="margin: 0 0 8px 0; font-weight: bold;">${source.name}</h6>
                        ${frequencyHtml}
                        <p style="margin: 4px 0; font-size: 12px;">
                            <strong>Location:</strong> ${source.latitude.toFixed(4)}, ${source.longitude.toFixed(4)}
                        </p>
                        <p style="margin: 4px 0; font-size: 12px;">
                            <strong>Error Margin:</strong> ${source.error_margin_meters}m
                        </p>
                    </div>
                `);

                marker.setPopup(popup);

                // Handle marker click
                el.addEventListener('click', () => {
                    setSelectedSource(source);
                    if (source.longitude != null && source.latitude != null) {
                        map.current?.flyTo({
                            center: [source.longitude, source.latitude],
                            zoom: 10,
                        });
                    }
                });

                // Handle marker drag
                marker.on('dragend', async () => {
                    const lngLat = marker.getLngLat();
                    try {
                        await updateKnownSource(source.id, {
                            latitude: lngLat.lat,
                            longitude: lngLat.lng,
                        });
                        showNotification('success', 'Source location updated');
                        await fetchKnownSources();
                    } catch (error) {
                        console.error('Failed to update source location:', error);
                        showNotification('error', 'Failed to update source location');
                        if (source.longitude != null && source.latitude != null) {
                            marker.setLngLat([source.longitude, source.latitude]);
                        }
                    }
                });

                markers.current[source.id] = marker;

                // Add error margin circle
                if (map.current?.getSource(`source-circle-${source.id}`)) {
                    map.current.removeLayer(`source-circle-${source.id}`);
                    map.current.removeSource(`source-circle-${source.id}`);
                }

                map.current?.addSource(`source-circle-${source.id}`, {
                    type: 'geojson',
                    data: {
                        type: 'Feature',
                        geometry: {
                            type: 'Point',
                            coordinates: [source.longitude, source.latitude],
                        },
                        properties: {},
                    },
                });

                map.current?.addLayer({
                    id: `source-circle-${source.id}`,
                    type: 'circle',
                    source: `source-circle-${source.id}`,
                    paint: {
                        'circle-radius': {
                            stops: [
                                [0, 0],
                                [20, metersToPixels(source.error_margin_meters, source.latitude, 20)],
                            ],
                            base: 2,
                        },
                        'circle-color': source.is_validated ? '#10b981' : '#f59e0b',
                        'circle-opacity': 0.2,
                        'circle-stroke-width': 2,
                        'circle-stroke-color': source.is_validated ? '#10b981' : '#f59e0b',
                        'circle-stroke-opacity': 0.5,
                    },
                });
            });
        };

        updateMarkers();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [knownSources, mapLoaded]);

    // Utility function to convert meters to pixels at a given latitude and zoom
    const metersToPixels = (meters: number, latitude: number, zoom: number) => {
        return meters / (156543.03392 * Math.cos((latitude * Math.PI) / 180) / Math.pow(2, zoom));
    };

    const showNotification = (type: 'success' | 'error', message: string) => {
        setNotification({ type, message });
        setTimeout(() => setNotification(null), 5000);
    };

    const validateForm = (): boolean => {
        const errors: Partial<SourceFormData> = {};

        if (!formData.name.trim()) {
            errors.name = 'Name is required';
        }

        // Frequency is optional, but if provided must be valid
        if (formData.frequency_hz) {
            const freq = parseFloat(formData.frequency_hz);
            if (isNaN(freq) || freq <= 0) {
                errors.frequency_hz = 'Frequency must be a positive number';
            }
        }

        // Coordinates are optional, but if one is provided, both must be valid
        const hasLat = formData.latitude.trim() !== '';
        const hasLon = formData.longitude.trim() !== '';

        if (hasLat || hasLon) {
            if (!hasLat) {
                errors.latitude = 'Latitude required when longitude is provided';
            } else {
                const lat = parseFloat(formData.latitude);
                if (isNaN(lat) || lat < -90 || lat > 90) {
                    errors.latitude = 'Latitude must be between -90 and 90';
                }
            }

            if (!hasLon) {
                errors.longitude = 'Longitude required when latitude is provided';
            } else {
                const lon = parseFloat(formData.longitude);
                if (isNaN(lon) || lon < -180 || lon > 180) {
                    errors.longitude = 'Longitude must be between -180 and 180';
                }
            }
        }

        // Error margin validation
        if (formData.error_margin_meters) {
            const errorMargin = parseFloat(formData.error_margin_meters);
            if (isNaN(errorMargin) || errorMargin <= 0) {
                errors.error_margin_meters = 'Error margin must be greater than 0';
            }
        }

        setFormErrors(errors);
        return Object.keys(errors).length === 0;
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        clearError();

        if (!validateForm()) {
            return;
        }

        try {
            const sourceData: KnownSourceCreate | KnownSourceUpdate = {
                name: formData.name.trim(),
                description: formData.description.trim() || undefined,
                frequency_hz: formData.frequency_hz ? parseInt(formData.frequency_hz) : undefined,
                latitude: formData.latitude ? parseFloat(formData.latitude) : undefined,
                longitude: formData.longitude ? parseFloat(formData.longitude) : undefined,
                power_dbm: formData.power_dbm ? parseFloat(formData.power_dbm) : undefined,
                source_type: formData.source_type || undefined,
                is_validated: formData.is_validated,
                error_margin_meters: formData.error_margin_meters ? parseFloat(formData.error_margin_meters) : undefined,
            };

            if (editingSource) {
                await updateKnownSource(editingSource.id, sourceData as KnownSourceUpdate);
                showNotification('success', 'Source updated successfully');
            } else {
                await createKnownSource(sourceData as KnownSourceCreate);
                showNotification('success', 'Source created successfully');
            }

            // Remove temp marker on successful creation
            if (tempMarker.current) {
                tempMarker.current.remove();
                tempMarker.current = null;
            }

            handleCancelForm();
            await fetchKnownSources();
        } catch (error: unknown) {
            const errorMessage =
                (error as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail ||
                (error as { message?: string })?.message ||
                'Operation failed';
            showNotification('error', errorMessage);
        }
    };

    const handleEdit = (source: KnownSource) => {
        setEditingSource(source);
        setFormData({
            name: source.name,
            description: source.description || '',
            frequency_hz: source.frequency_hz?.toString() || '',
            latitude: source.latitude?.toString() || '',
            longitude: source.longitude?.toString() || '',
            power_dbm: source.power_dbm?.toString() || '',
            source_type: source.source_type || 'beacon',
            is_validated: source.is_validated,
            error_margin_meters: source.error_margin_meters.toString(),
        });
        setIsFormVisible(true);
        setSelectedSource(null);
    };

    const handleDelete = async (sourceId: string) => {
        try {
            await deleteKnownSource(sourceId);
            showNotification('success', 'Source deleted successfully');
            setDeleteConfirm(null);
            setSelectedSource(null);
            await fetchKnownSources();
        } catch (error: unknown) {
            const errorMessage =
                (error as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail ||
                (error as { message?: string })?.message ||
                'Failed to delete source';
            showNotification('error', errorMessage);
        }
    };

    const handleCancelForm = () => {
        setIsFormVisible(false);
        setEditingSource(null);
        setFormData({
            name: '',
            description: '',
            frequency_hz: '',
            latitude: '',
            longitude: '',
            power_dbm: '',
            source_type: 'beacon',
            is_validated: false,
            error_margin_meters: '50',
        });
        setFormErrors({});

        // Remove temp marker
        if (tempMarker.current) {
            tempMarker.current.remove();
            tempMarker.current = null;
        }
    };

    return (
        <>
            {/* Breadcrumb */}
            <div className="page-header">
                <div className="page-block">
                    <div className="row align-items-center">
                        <div className="col-md-12">
                            <ul className="breadcrumb">
                                <li className="breadcrumb-item"><a href="/dashboard">Home</a></li>
                                <li className="breadcrumb-item"><a href="#">RF Operations</a></li>
                                <li className="breadcrumb-item" aria-current="page">Sources Management</li>
                            </ul>
                        </div>
                        <div className="col-md-12">
                            <div className="page-header-title">
                                <h2 className="mb-0">Sources Management</h2>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Notification */}
            {notification && (
                <div className={`alert alert-${notification.type === 'success' ? 'success' : 'danger'} alert-dismissible fade show`} role="alert">
                    <strong>{notification.type === 'success' ? 'Success!' : 'Error!'}</strong> {notification.message}
                    <button type="button" className="btn-close" onClick={() => setNotification(null)}></button>
                </div>
            )}

            {/* Error Alert */}
            {storeError && (
                <div className="alert alert-danger alert-dismissible fade show" role="alert">
                    <strong>Error!</strong> {storeError}
                    <button type="button" className="btn-close" onClick={clearError}></button>
                </div>
            )}

            <div className="row">
                {/* Map View */}
                <div className="col-lg-8">
                    <div className="card">
                        <div className="card-header">
                            <h5 className="mb-0">Map View</h5>
                        </div>
                        <div className="card-body p-0">
                            {mapError ? (
                                <div className="alert alert-warning m-3 mb-0" role="alert">
                                    <i className="ph ph-warning-circle me-2"></i>
                                    {mapError}
                                    <p className="mt-2 mb-0 small">
                                        Use the table below to manage RF sources manually.
                                    </p>
                                </div>
                            ) : (
                                <div ref={mapContainer} style={{ height: '600px', width: '100%' }} />
                            )}
                        </div>
                        <div className="card-footer">
                            <div className="d-flex gap-3 align-items-center flex-wrap">
                                <div className="d-flex align-items-center gap-2">
                                    <div style={{ width: '20px', height: '20px', borderRadius: '50%', backgroundColor: '#10b981', border: '2px solid white' }}></div>
                                    <span className="f-12">Validated</span>
                                </div>
                                <div className="d-flex align-items-center gap-2">
                                    <div style={{ width: '20px', height: '20px', borderRadius: '50%', backgroundColor: '#f59e0b', border: '2px solid white' }}></div>
                                    <span className="f-12">Unvalidated</span>
                                </div>
                                {isFormVisible && !editingSource && (
                                    <div className="d-flex align-items-center gap-2">
                                        <div style={{ width: '20px', height: '20px', borderRadius: '50%', backgroundColor: '#3b82f6', border: '2px solid white' }}></div>
                                        <span className="f-12">New (temp)</span>
                                    </div>
                                )}
                                <div className="ms-auto text-muted f-12">
                                    <i className="ph ph-info me-1"></i>
                                    {isFormVisible && !editingSource
                                        ? 'Click map to set location for new source'
                                        : 'Drag markers to update position'
                                    }
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Sources List / Form */}
                <div className="col-lg-4">
                    {isFormVisible ? (
                        <div className="card">
                            <div className="card-header">
                                <h5 className="mb-0">{editingSource ? 'Edit Source' : 'New Source'}</h5>
                            </div>
                            <div className="card-body">
                                <form onSubmit={handleSubmit}>
                                    <div className="mb-3">
                                        <label className="form-label">Name *</label>
                                        <input
                                            type="text"
                                            className={`form-control ${formErrors.name ? 'is-invalid' : ''}`}
                                            value={formData.name}
                                            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                        />
                                        {formErrors.name && <div className="invalid-feedback">{formErrors.name}</div>}
                                    </div>

                                    <div className="mb-3">
                                        <label className="form-label">Description</label>
                                        <textarea
                                            className="form-control"
                                            rows={2}
                                            value={formData.description}
                                            onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                                        />
                                    </div>

                                    <div className="mb-3">
                                        <label className="form-label">Frequency (Hz)</label>
                                        <input
                                            type="number"
                                            className={`form-control ${formErrors.frequency_hz ? 'is-invalid' : ''}`}
                                            value={formData.frequency_hz}
                                            onChange={(e) => setFormData({ ...formData, frequency_hz: e.target.value })}
                                            placeholder="Optional - e.g., 144800000"
                                        />
                                        {formErrors.frequency_hz && <div className="invalid-feedback">{formErrors.frequency_hz}</div>}
                                        <small className="text-muted">Optional - Can be unknown for amateur stations</small>
                                    </div>

                                    <div className="mb-3">
                                        <label className="form-label">Latitude</label>
                                        <input
                                            type="number"
                                            step="0.000001"
                                            className={`form-control ${formErrors.latitude ? 'is-invalid' : ''}`}
                                            value={formData.latitude}
                                            onChange={(e) => setFormData({ ...formData, latitude: e.target.value })}
                                            placeholder="Click map to set"
                                        />
                                        {formErrors.latitude && <div className="invalid-feedback">{formErrors.latitude}</div>}
                                        <small className="text-muted">Optional - Click map to select location</small>
                                    </div>

                                    <div className="mb-3">
                                        <label className="form-label">Longitude</label>
                                        <input
                                            type="number"
                                            step="0.000001"
                                            className={`form-control ${formErrors.longitude ? 'is-invalid' : ''}`}
                                            value={formData.longitude}
                                            onChange={(e) => setFormData({ ...formData, longitude: e.target.value })}
                                            placeholder="Click map to set"
                                        />
                                        {formErrors.longitude && <div className="invalid-feedback">{formErrors.longitude}</div>}
                                        <small className="text-muted">Optional - Click map to select location</small>
                                    </div>

                                    <div className="mb-3">
                                        <label className="form-label">Error Margin (meters)</label>
                                        <input
                                            type="number"
                                            step="0.1"
                                            className={`form-control ${formErrors.error_margin_meters ? 'is-invalid' : ''}`}
                                            value={formData.error_margin_meters}
                                            onChange={(e) => setFormData({ ...formData, error_margin_meters: e.target.value })}
                                            placeholder="50"
                                        />
                                        {formErrors.error_margin_meters && <div className="invalid-feedback">{formErrors.error_margin_meters}</div>}
                                        <small className="text-muted">Uncertainty radius (default: 50m)</small>
                                    </div>

                                    <div className="mb-3">
                                        <label className="form-label">Power (dBm)</label>
                                        <input
                                            type="number"
                                            step="0.1"
                                            className="form-control"
                                            value={formData.power_dbm}
                                            onChange={(e) => setFormData({ ...formData, power_dbm: e.target.value })}
                                        />
                                    </div>

                                    <div className="mb-3">
                                        <label className="form-label">Source Type</label>
                                        <select
                                            className="form-select"
                                            value={formData.source_type}
                                            onChange={(e) => setFormData({ ...formData, source_type: e.target.value })}
                                        >
                                            <option value="beacon">Beacon</option>
                                            <option value="repeater">Repeater</option>
                                            <option value="station">Station</option>
                                            <option value="other">Other</option>
                                        </select>
                                    </div>

                                    <div className="mb-3 form-check">
                                        <input
                                            type="checkbox"
                                            className="form-check-input"
                                            id="isValidated"
                                            checked={formData.is_validated}
                                            onChange={(e) => setFormData({ ...formData, is_validated: e.target.checked })}
                                        />
                                        <label className="form-check-label" htmlFor="isValidated">
                                            Validated
                                        </label>
                                    </div>

                                    <div className="d-flex gap-2">
                                        <button type="submit" className="btn btn-primary" disabled={isLoading}>
                                            {isLoading ? (
                                                <>
                                                    <span className="spinner-border spinner-border-sm me-2"></span>
                                                    Saving...
                                                </>
                                            ) : (
                                                <>
                                                    <i className="ph ph-check me-2"></i>
                                                    {editingSource ? 'Update' : 'Create'}
                                                </>
                                            )}
                                        </button>
                                        <button type="button" className="btn btn-secondary" onClick={handleCancelForm} disabled={isLoading}>
                                            Cancel
                                        </button>
                                    </div>
                                </form>
                            </div>
                        </div>
                    ) : (
                        <div className="card">
                            <div className="card-header d-flex justify-content-between align-items-center">
                                <h5 className="mb-0">Sources List</h5>
                                <button className="btn btn-primary btn-sm" onClick={() => setIsFormVisible(true)}>
                                    <i className="ph ph-plus me-1"></i>
                                    Add Source
                                </button>
                            </div>
                            <div className="card-body" style={{ maxHeight: '600px', overflowY: 'auto' }}>
                                {isLoading && knownSources.length === 0 ? (
                                    <div className="text-center py-4">
                                        <div className="spinner-border text-primary" role="status">
                                            <span className="visually-hidden">Loading...</span>
                                        </div>
                                    </div>
                                ) : knownSources.length === 0 ? (
                                    <div className="text-center py-4">
                                        <i className="ph ph-radio-button display-4 text-muted"></i>
                                        <p className="text-muted mt-3">No sources yet. Click "Add Source" to create one.</p>
                                    </div>
                                ) : (
                                    <div className="list-group list-group-flush">
                                        {knownSources.map((source) => (
                                            <div
                                                key={source.id}
                                                className={`list-group-item list-group-item-action ${selectedSource?.id === source.id ? 'active' : ''}`}
                                                onClick={() => {
                                                    setSelectedSource(source);
                                                    if (source.latitude != null && source.longitude != null) {
                                                        map.current?.flyTo({
                                                            center: [source.longitude, source.latitude],
                                                            zoom: 10,
                                                        });
                                                    }
                                                }}
                                                style={{ cursor: 'pointer' }}
                                            >
                                                <div className="d-flex w-100 justify-content-between align-items-start">
                                                    <div className="flex-grow-1">
                                                        <h6 className="mb-1">
                                                            {source.name}
                                                            {source.is_validated && (
                                                                <span className="badge bg-success ms-2">Validated</span>
                                                            )}
                                                            {!source.latitude && !source.longitude && (
                                                                <span className="badge bg-secondary ms-2">No Location</span>
                                                            )}
                                                        </h6>
                                                        {source.frequency_hz && (
                                                            <p className="mb-1 f-12">
                                                                <i className="ph ph-radio-button me-1"></i>
                                                                {(source.frequency_hz / 1e6).toFixed(3)} MHz
                                                            </p>
                                                        )}
                                                        {source.latitude != null && source.longitude != null && (
                                                            <p className="mb-1 f-12">
                                                                <i className="ph ph-map-pin me-1"></i>
                                                                {source.latitude.toFixed(4)}, {source.longitude.toFixed(4)}
                                                            </p>
                                                        )}
                                                        <p className="mb-1 f-12">
                                                            <i className="ph ph-circle me-1"></i>
                                                            ±{source.error_margin_meters}m
                                                        </p>
                                                        {source.description && (
                                                            <p className="mb-1 f-12 text-muted">{source.description}</p>
                                                        )}
                                                    </div>
                                                    <div className="btn-group-vertical" role="group">
                                                        <button
                                                            className="btn btn-sm btn-outline-primary"
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                handleEdit(source);
                                                            }}
                                                        >
                                                            <i className="ph ph-pencil"></i>
                                                        </button>
                                                        <button
                                                            className="btn btn-sm btn-outline-danger"
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                setDeleteConfirm(source.id);
                                                            }}
                                                        >
                                                            <i className="ph ph-trash"></i>
                                                        </button>
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                            <div className="card-footer">
                                <span className="badge bg-light-primary">{knownSources.length} Sources</span>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Delete Confirmation Modal */}
            {deleteConfirm && (
                <div className="modal fade show d-block" style={{ backgroundColor: 'rgba(0,0,0,0.5)' }}>
                    <div className="modal-dialog modal-dialog-centered">
                        <div className="modal-content">
                            <div className="modal-header">
                                <h5 className="modal-title">Confirm Deletion</h5>
                                <button type="button" className="btn-close" onClick={() => setDeleteConfirm(null)}></button>
                            </div>
                            <div className="modal-body">
                                <p>Are you sure you want to delete this source?</p>
                                <p className="text-danger mb-0">
                                    <i className="ph ph-warning me-2"></i>
                                    This action cannot be undone. Sources referenced by recording sessions cannot be deleted.
                                </p>
                            </div>
                            <div className="modal-footer">
                                <button type="button" className="btn btn-secondary" onClick={() => setDeleteConfirm(null)}>
                                    Cancel
                                </button>
                                <button
                                    type="button"
                                    className="btn btn-danger"
                                    onClick={() => handleDelete(deleteConfirm)}
                                >
                                    <i className="ph ph-trash me-2"></i>
                                    Delete
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
};

export default SourcesManagement;
