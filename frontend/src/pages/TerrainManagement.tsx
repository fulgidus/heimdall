/**
 * Terrain Management Page
 * 
 * Manages SRTM terrain tiles:
 * - Coverage status visualization
 * - Mapbox map with tile boundaries
 * - Tile download and deletion
 */

import React, { useEffect, useState, useRef } from 'react';
import { Container, Row, Col, Card, Button, Table, Badge, ProgressBar, Alert } from 'react-bootstrap';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { useTerrainStore } from '../store/terrainStore';
import { useWebSocket } from '../contexts/WebSocketContext';
import type { TerrainTile } from '../services/api/terrain';

// Mapbox token from environment
const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN;

const TerrainManagement: React.FC = () => {
  const {
    tiles,
    coverage,
    loading,
    downloading,
    error,
    fetchTiles,
    fetchCoverage,
    downloadWebSDRRegion,
    deleteTile,
    clearError
  } = useTerrainStore();

  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [selectedTile, setSelectedTile] = useState<string | null>(null);
  const [downloadResult, setDownloadResult] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<{current: number; total: number; tile_name: string} | null>(null);
  
  const { subscribe, isConnected } = useWebSocket();

  // Initialize map and load data
  useEffect(() => {
    fetchTiles();
    fetchCoverage();
  }, [fetchTiles, fetchCoverage]);

  // Subscribe to WebSocket terrain tile progress events
  useEffect(() => {
    if (!isConnected) return;

    const unsubscribe = subscribe('terrain:tile_progress', (data: any) => {
      console.log('[TerrainManagement] Terrain progress:', data);
      
      if (data.data) {
        const { tile_name, status, current, total, file_size } = data.data;
        
        // Update progress state
        setDownloadProgress({ current, total, tile_name });
        
        // When a tile completes (ready or failed), refresh the tiles list
        if (status === 'ready' || status === 'failed') {
          fetchTiles();
        }
        
        // Log progress
        console.log(`Tile ${tile_name}: ${status} (${current}/${total})${file_size ? ` - ${(file_size / 1024 / 1024).toFixed(2)} MB` : ''}`);
      }
    });

    return () => {
      unsubscribe();
    };
  }, [isConnected, subscribe, fetchTiles]);

  // Initialize Mapbox map
  useEffect(() => {
    if (!mapContainer.current || map.current) return;

    if (!MAPBOX_TOKEN) {
      console.error('Mapbox token not configured');
      return;
    }

    mapboxgl.accessToken = MAPBOX_TOKEN;

    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/satellite-streets-v12',
      center: [8.5, 45.0], // Center on NW Italy
      zoom: 7
    });

    map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');

    return () => {
      map.current?.remove();
      map.current = null;
    };
  }, []);

  // Update map with tile boundaries and coverage
  useEffect(() => {
    if (!map.current) return;

    // Wait for map to load
    if (!map.current.isStyleLoaded()) {
      map.current.once('load', () => updateMapLayers());
    } else {
      updateMapLayers();
    }
  }, [tiles, coverage, selectedTile]);

  const updateMapLayers = () => {
    if (!map.current) return;

    // Remove existing layers and sources
    const layersToRemove = ['tiles-fill-layer', 'tiles-border-layer', 'missing-tiles-layer', 'bbox-layer'];
    const sourcesToRemove = ['tiles', 'missing-tiles', 'bbox'];
    
    layersToRemove.forEach(layer => {
      if (map.current?.getLayer(layer)) {
        map.current.removeLayer(layer);
      }
    });
    
    sourcesToRemove.forEach(source => {
      if (map.current?.getSource(source)) {
        map.current.removeSource(source);
      }
    });

    // 1. Add bounding box if coverage exists
    if (coverage && coverage.region) {
      const { lat_min, lat_max, lon_min, lon_max } = coverage.region;
      
      map.current.addSource('bbox', {
        type: 'geojson',
        data: {
          type: 'Feature',
          properties: {},
          geometry: {
            type: 'Polygon',
            coordinates: [[
              [lon_min, lat_min],
              [lon_max, lat_min],
              [lon_max, lat_max],
              [lon_min, lat_max],
              [lon_min, lat_min]
            ]]
          }
        }
      });

      map.current.addLayer({
        id: 'bbox-layer',
        type: 'line',
        source: 'bbox',
        paint: {
          'line-color': '#007bff',
          'line-width': 3,
          'line-dasharray': [2, 2]
        }
      });
    }

    // 2. Add missing tiles (future tiles to download)
    if (coverage && coverage.region.tiles_missing.length > 0) {
      const missingFeatures = coverage.region.tiles_missing.map(tileName => {
        // Parse tile name (e.g., N44E007)
        const match = tileName.match(/([NS])(\d+)([EW])(\d+)/);
        if (!match) return null;

        const latSign = match[1] === 'N' ? 1 : -1;
        const lat = parseInt(match[2]) * latSign;
        const lonSign = match[3] === 'E' ? 1 : -1;
        const lon = parseInt(match[4]) * lonSign;

        return {
          type: 'Feature' as const,
          properties: {
            tile_name: tileName,
            status: 'missing'
          },
          geometry: {
            type: 'Polygon' as const,
            coordinates: [[
              [lon, lat],
              [lon + 1, lat],
              [lon + 1, lat + 1],
              [lon, lat + 1],
              [lon, lat]
            ]]
          }
        };
      }).filter(Boolean);

      if (missingFeatures.length > 0) {
        map.current.addSource('missing-tiles', {
          type: 'geojson',
          data: {
            type: 'FeatureCollection',
            features: missingFeatures as any[]
          }
        });

        map.current.addLayer({
          id: 'missing-tiles-layer',
          type: 'line',
          source: 'missing-tiles',
          paint: {
            'line-color': '#dc3545',
            'line-width': 2,
            'line-dasharray': [4, 4]
          }
        });
      }
    }

    // 3. Add existing tiles with fill and border
    if (tiles.length > 0) {
      const features = tiles.map(tile => {
        const color = getTileColor(tile.status, tile.tile_name === selectedTile);
        const fillColor = getTileFillColor(tile.status);
        
        return {
          type: 'Feature' as const,
          properties: {
            tile_name: tile.tile_name,
            status: tile.status,
            color: color,
            fillColor: fillColor,
            isSelected: tile.tile_name === selectedTile
          },
          geometry: {
            type: 'Polygon' as const,
            coordinates: [[
              [tile.lon_min, tile.lat_min],
              [tile.lon_max, tile.lat_min],
              [tile.lon_max, tile.lat_max],
              [tile.lon_min, tile.lat_max],
              [tile.lon_min, tile.lat_min]
            ]]
          }
        };
      });

      map.current.addSource('tiles', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: features
        }
      });

      // Add fill layer (semi-transparent)
      map.current.addLayer({
        id: 'tiles-fill-layer',
        type: 'fill',
        source: 'tiles',
        paint: {
          'fill-color': ['get', 'fillColor'],
          'fill-opacity': 0.3
        }
      });

      // Add border layer
      map.current.addLayer({
        id: 'tiles-border-layer',
        type: 'line',
        source: 'tiles',
        paint: {
          'line-color': ['get', 'color'],
          'line-width': [
            'case',
            ['get', 'isSelected'],
            4,
            2
          ]
        }
      });

      // Add click handler
      map.current.on('click', 'tiles-fill-layer', (e) => {
        if (e.features && e.features.length > 0) {
          const feature = e.features[0];
          const tileName = feature.properties?.tile_name;
          setSelectedTile(tileName === selectedTile ? null : tileName);
        }
      });

      // Change cursor on hover
      map.current.on('mouseenter', 'tiles-fill-layer', () => {
        if (map.current) {
          map.current.getCanvas().style.cursor = 'pointer';
        }
      });

      map.current.on('mouseleave', 'tiles-fill-layer', () => {
        if (map.current) {
          map.current.getCanvas().style.cursor = '';
        }
      });
    }
  };

  const getTileColor = (status: string, isSelected: boolean): string => {
    if (isSelected) return '#ff00ff'; // Magenta for selected
    
    switch (status) {
      case 'ready':
        return '#28a745'; // Green
      case 'downloading':
        return '#ffc107'; // Yellow
      case 'failed':
        return '#dc3545'; // Red
      default:
        return '#6c757d'; // Gray
    }
  };

  const getTileFillColor = (status: string): string => {
    switch (status) {
      case 'ready':
        return '#28a745'; // Green
      case 'downloading':
        return '#ffc107'; // Yellow
      case 'failed':
        return '#dc3545'; // Red
      default:
        return '#6c757d'; // Gray
    }
  };

  const getStatusBadge = (status: string) => {
    const variants: Record<string, string> = {
      ready: 'success',
      downloading: 'warning',
      failed: 'danger',
      pending: 'secondary'
    };

    return <Badge bg={variants[status] || 'secondary'}>{status}</Badge>;
  };

  const formatFileSize = (bytes?: number): string => {
    if (!bytes) return 'N/A';
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(2)} MB`;
  };

  const formatDate = (dateStr?: string): string => {
    if (!dateStr) return 'N/A';
    return new Date(dateStr).toLocaleString();
  };

  const handleDownload = async () => {
    setDownloadResult(null);
    setDownloadProgress(null);
    try {
      await downloadWebSDRRegion();
      // Result will be updated via WebSocket events
      // When all tiles complete, reset progress
      setDownloadProgress(null);
      setDownloadResult('Download completed! Check the tiles table for details.');
      // Refresh data
      await fetchTiles();
      await fetchCoverage();
    } catch (error: any) {
      console.error('Download error:', error);
      setDownloadProgress(null);
    }
  };

  const handleDelete = async (tileName: string) => {
    if (!confirm(`Delete tile ${tileName}?`)) return;
    
    try {
      await deleteTile(tileName);
      setSelectedTile(null);
    } catch (error: any) {
      console.error('Delete error:', error);
    }
  };

  return (
    <Container fluid className="p-4">
      <Row className="mb-4">
        <Col>
          <h2>Terrain Management</h2>
          <p className="text-muted">
            Manage SRTM terrain tiles for realistic RF propagation simulation
          </p>
        </Col>
      </Row>

      {error && (
        <Alert variant="danger" dismissible onClose={clearError}>
          {error}
        </Alert>
      )}

      {downloadResult && (
        <Alert variant="info" dismissible onClose={() => setDownloadResult(null)}>
          {downloadResult}
        </Alert>
      )}

      {/* Download Progress */}
      {downloadProgress && downloading && (
        <Alert variant="info">
          <div className="d-flex justify-content-between align-items-center mb-2">
            <span>
              <strong>Downloading:</strong> {downloadProgress.tile_name}
            </span>
            <span>
              {downloadProgress.current} / {downloadProgress.total} tiles
            </span>
          </div>
          <ProgressBar
            now={(downloadProgress.current / downloadProgress.total) * 100}
            label={`${Math.round((downloadProgress.current / downloadProgress.total) * 100)}%`}
            animated
            striped
          />
        </Alert>
      )}

      {/* Coverage Status Card */}
      <Row className="mb-4">
        <Col>
          <Card>
            <Card.Header>
              <h5>Coverage Status</h5>
            </Card.Header>
            <Card.Body>
              {loading && !coverage ? (
                <p>Loading coverage status...</p>
              ) : coverage ? (
                <>
                  <Row className="mb-3">
                    <Col md={6}>
                      <p className="mb-1">
                        <strong>Region:</strong> {coverage.region.lat_min.toFixed(2)}°N - {coverage.region.lat_max.toFixed(2)}°N, {' '}
                        {coverage.region.lon_min.toFixed(2)}°E - {coverage.region.lon_max.toFixed(2)}°E
                      </p>
                      <p className="mb-1">
                        <strong>Total Tiles:</strong> {coverage.total_tiles}
                      </p>
                    </Col>
                    <Col md={6}>
                      <p className="mb-1">
                        <Badge bg="success" className="me-2">Ready: {coverage.ready_count}</Badge>
                        <Badge bg="warning" className="me-2">Downloading: {coverage.downloading_count}</Badge>
                        <Badge bg="danger" className="me-2">Failed: {coverage.failed_count}</Badge>
                        <Badge bg="secondary">Missing: {coverage.missing_count}</Badge>
                      </p>
                    </Col>
                  </Row>

                  <ProgressBar className="mb-3" style={{ height: '30px' }}>
                    <ProgressBar
                      variant="success"
                      now={(coverage.ready_count / coverage.total_tiles) * 100}
                      label={`${coverage.ready_count} ready`}
                    />
                    <ProgressBar
                      variant="warning"
                      now={(coverage.downloading_count / coverage.total_tiles) * 100}
                      label={`${coverage.downloading_count} downloading`}
                    />
                    <ProgressBar
                      variant="danger"
                      now={(coverage.failed_count / coverage.total_tiles) * 100}
                      label={`${coverage.failed_count} failed`}
                    />
                  </ProgressBar>

                  <Button
                    variant="primary"
                    onClick={handleDownload}
                    disabled={downloading || coverage.missing_count === 0}
                  >
                    {downloading ? 'Downloading...' : `Download WebSDR Region (${coverage.missing_count} tiles)`}
                  </Button>
                </>
              ) : (
                <p>No coverage data available</p>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Map */}
      <Row className="mb-4">
        <Col>
          <Card>
            <Card.Header>
              <h5>Tile Coverage Map</h5>
            </Card.Header>
            <Card.Body>
              <div
                ref={mapContainer}
                style={{ width: '100%', height: '500px' }}
              />
              {!MAPBOX_TOKEN && (
                <Alert variant="warning" className="mt-3">
                  Mapbox token not configured. Set VITE_MAPBOX_TOKEN in .env
                </Alert>
              )}
              <div className="mt-3">
                <small className="text-muted">
                  <strong>Tiles:</strong>{' '}
                  <span style={{ color: '#28a745' }}>■</span> Ready{' '}
                  <span style={{ color: '#ffc107' }} className="ms-3">■</span> Downloading{' '}
                  <span style={{ color: '#dc3545' }} className="ms-3">■</span> Failed{' '}
                  <span style={{ color: '#6c757d' }} className="ms-3">■</span> Pending{' '}
                  <span style={{ color: '#ff00ff' }} className="ms-3">■</span> Selected
                  <span className="ms-4">|</span>
                  <strong className="ms-3">Regions:</strong>{' '}
                  <span style={{ 
                    display: 'inline-block',
                    width: '20px',
                    height: '2px',
                    backgroundColor: '#007bff',
                    verticalAlign: 'middle',
                    borderTop: '2px dashed #007bff',
                    marginRight: '5px'
                  }}></span> Bounding Box{' '}
                  <span style={{ 
                    display: 'inline-block',
                    width: '20px',
                    height: '2px',
                    backgroundColor: '#dc3545',
                    verticalAlign: 'middle',
                    borderTop: '2px dashed #dc3545',
                    marginLeft: '10px',
                    marginRight: '5px'
                  }}></span> Missing Tiles
                </small>
              </div>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Tiles Table */}
      <Row>
        <Col>
          <Card>
            <Card.Header>
              <h5>Terrain Tiles</h5>
            </Card.Header>
            <Card.Body>
              {loading && !tiles.length ? (
                <p>Loading tiles...</p>
              ) : tiles.length === 0 ? (
                <p>No tiles downloaded yet</p>
              ) : (
                <Table striped bordered hover responsive>
                  <thead>
                    <tr>
                      <th>Tile Name</th>
                      <th>Bounds</th>
                      <th>Status</th>
                      <th>Size</th>
                      <th>Downloaded At</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tiles.map(tile => (
                      <tr
                        key={tile.tile_name}
                        className={tile.tile_name === selectedTile ? 'table-active' : ''}
                        style={{ cursor: 'pointer' }}
                        onClick={() => setSelectedTile(tile.tile_name === selectedTile ? null : tile.tile_name)}
                      >
                        <td><strong>{tile.tile_name}</strong></td>
                        <td>
                          {tile.lat_min}°-{tile.lat_max}°N, {tile.lon_min}°-{tile.lon_max}°E
                        </td>
                        <td>{getStatusBadge(tile.status)}</td>
                        <td>{formatFileSize(tile.file_size_bytes)}</td>
                        <td>{formatDate(tile.downloaded_at)}</td>
                        <td>
                          <Button
                            variant="danger"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDelete(tile.tile_name);
                            }}
                            disabled={tile.status === 'downloading'}
                          >
                            Delete
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default TerrainManagement;
