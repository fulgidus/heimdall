/**
 * Terrain/SRTM API Service
 *
 * Handles all terrain-related API calls:
 * - List terrain tiles
 * - Download SRTM tiles
 * - Query elevation
 * - Get coverage status
 * - Delete tiles
 */

import api from '@/lib/api';

// Type definitions
export interface TerrainTile {
  id?: string;
  tile_name: string;
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
  minio_bucket: string;
  minio_path: string;
  file_size_bytes?: number;
  status: 'pending' | 'downloading' | 'ready' | 'failed';
  error_message?: string;
  checksum_sha256?: string;
  source_url?: string;
  downloaded_at?: string;
  created_at?: string;
  updated_at?: string;
}

export interface TerrainTilesList {
  tiles: TerrainTile[];
  total: number;
  ready: number;
  downloading: number;
  failed: number;
  pending: number;
}

export interface BoundsRequest {
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
}

export interface DownloadRequest {
  bounds?: BoundsRequest;
}

export interface TileDownloadResult {
  tile_name: string;
  success: boolean;
  error?: string;
  file_size?: number;
  checksum?: string;
}

export interface DownloadResponse {
  successful: number;
  failed: number;
  total: number;
  tiles: TileDownloadResult[];
}

export interface CoverageRegion {
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
  tiles_needed: string[];
  tiles_ready: string[];
  tiles_downloading: string[];
  tiles_failed: string[];
  tiles_missing: string[];
}

export interface CoverageStatus {
  region: CoverageRegion;
  total_tiles: number;
  ready_count: number;
  downloading_count: number;
  failed_count: number;
  missing_count: number;
  coverage_percent: number;
}

export interface ElevationResponse {
  lat: number;
  lon: number;
  elevation_m: number;
  tile_name: string;
  source: 'srtm' | 'simplified';
}

/**
 * Terrain API client
 */
export const terrainApi = {
  /**
   * List all terrain tiles
   */
  async listTiles(): Promise<TerrainTilesList> {
    const response = await api.get('/v1/terrain/tiles');
    return response.data;
  },

  /**
   * Download terrain tiles
   * @param request Optional bounds, otherwise auto-detect from WebSDR stations
   */
  async downloadTiles(request?: DownloadRequest): Promise<DownloadResponse> {
    const response = await api.post('/v1/terrain/download', request || {});
    return response.data;
  },

  /**
   * Get coverage status for WebSDR region
   */
  async getCoverage(): Promise<CoverageStatus> {
    const response = await api.get('/v1/terrain/coverage');
    return response.data;
  },

  /**
   * Query elevation at specific coordinates
   */
  async queryElevation(lat: number, lon: number): Promise<ElevationResponse> {
    const response = await api.get('/v1/terrain/elevation', {
      params: { lat, lon }
    });
    return response.data;
  },

  /**
   * Delete a terrain tile
   */
  async deleteTile(tileName: string): Promise<{ success: boolean; message: string }> {
    const response = await api.delete(`/v1/terrain/tiles/${tileName}`);
    return response.data;
  }
};
