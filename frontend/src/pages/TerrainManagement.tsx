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

  // Initialize map and load data
  useEffect(() => {
    fetchTiles();
    fetchCoverage();
  }, []);

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

  // Update map with tile boundaries
  useEffect(() => {
    if (!map.current || !tiles.length) return;

    // Wait for map to load
    if (!map.current.isStyleLoaded()) {
      map.current.once('load', () => updateMapTiles());
    } else {
      updateMapTiles();
    }
  }, [tiles, selectedTile]);

  const updateMapTiles = () => {
    if (!map.current) return;

    // Remove existing layers and sources
    if (map.current.getLayer('tiles-layer')) {
      map.current.removeLayer('tiles-layer');
    }
    if (map.current.getSource('tiles')) {
      map.current.removeSource('tiles');
    }

    // Create GeoJSON features for tiles
    const features = tiles.map(tile => {
      const color = getTileColor(tile.status, tile.tile_name === selectedTile);
      
      return {
        type: 'Feature' as const,
        properties: {
          tile_name: tile.tile_name,
          status: tile.status,
          color: color
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

    // Add source and layer
    map.current.addSource('tiles', {
      type: 'geojson',
      data: {
        type: 'FeatureCollection',
        features: features
      }
    });

    map.current.addLayer({
      id: 'tiles-layer',
      type: 'line',
      source: 'tiles',
      paint: {
        'line-color': ['get', 'color'],
        'line-width': 3
      }
    });

    // Add click handler
    map.current.on('click', 'tiles-layer', (e) => {
      if (e.features && e.features.length > 0) {
        const feature = e.features[0];
        const tileName = feature.properties?.tile_name;
        setSelectedTile(tileName === selectedTile ? null : tileName);
      }
    });

    // Change cursor on hover
    map.current.on('mouseenter', 'tiles-layer', () => {
      if (map.current) {
        map.current.getCanvas().style.cursor = 'pointer';
      }
    });

    map.current.on('mouseleave', 'tiles-layer', () => {
      if (map.current) {
        map.current.getCanvas().style.cursor = '';
      }
    });
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
    try {
      const result = await downloadWebSDRRegion();
      setDownloadResult(`Downloaded ${result.successful} tiles successfully, ${result.failed} failed.`);
      // Refresh data
      await fetchTiles();
      await fetchCoverage();
    } catch (error: any) {
      console.error('Download error:', error);
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
                  <span style={{ color: '#28a745' }}>■</span> Ready
                  <span style={{ color: '#ffc107' }} className="ms-3">■</span> Downloading
                  <span style={{ color: '#dc3545' }} className="ms-3">■</span> Failed
                  <span style={{ color: '#6c757d' }} className="ms-3">■</span> Pending
                  <span style={{ color: '#ff00ff' }} className="ms-3">■</span> Selected
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
