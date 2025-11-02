/**
 * Zustand store for terrain/SRTM data management
 */

import { create } from 'zustand';
import { terrainApi, TerrainTile, CoverageStatus } from '../services/api/terrain';

interface TerrainState {
  // Data
  tiles: TerrainTile[];
  coverage: CoverageStatus | null;
  
  // UI state
  loading: boolean;
  downloading: boolean;
  error: string | null;
  
  // Actions
  fetchTiles: () => Promise<void>;
  fetchCoverage: () => Promise<void>;
  downloadWebSDRRegion: () => Promise<void>;
  deleteTile: (tileName: string) => Promise<void>;
  queryElevation: (lat: number, lon: number) => Promise<{ elevation_m: number; source: string }>;
  clearError: () => void;
}

export const useTerrainStore = create<TerrainState>((set, get) => ({
  // Initial state
  tiles: [],
  coverage: null,
  loading: false,
  downloading: false,
  error: null,
  
  // Fetch all tiles
  fetchTiles: async () => {
    set({ loading: true, error: null });
    try {
      const data = await terrainApi.listTiles();
      set({ tiles: data.tiles, loading: false });
    } catch (error: any) {
      set({ 
        error: error.response?.data?.detail || 'Failed to fetch tiles',
        loading: false 
      });
    }
  },
  
  // Fetch coverage status
  fetchCoverage: async () => {
    set({ loading: true, error: null });
    try {
      const data = await terrainApi.getCoverage();
      set({ coverage: data, loading: false });
    } catch (error: any) {
      set({ 
        error: error.response?.data?.detail || 'Failed to fetch coverage',
        loading: false 
      });
    }
  },
  
  // Download tiles for WebSDR region (auto-detect bounds)
  downloadWebSDRRegion: async () => {
    set({ downloading: true, error: null });
    try {
      const result = await terrainApi.downloadTiles();
      
      // Refresh tiles and coverage after download
      await get().fetchTiles();
      await get().fetchCoverage();
      
      set({ downloading: false });
      
      // Return result for UI feedback
      return result;
    } catch (error: any) {
      set({ 
        error: error.response?.data?.detail || 'Failed to download tiles',
        downloading: false 
      });
      throw error;
    }
  },
  
  // Delete a specific tile
  deleteTile: async (tileName: string) => {
    set({ loading: true, error: null });
    try {
      await terrainApi.deleteTile(tileName);
      
      // Remove tile from local state
      set(state => ({
        tiles: state.tiles.filter(t => t.tile_name !== tileName),
        loading: false
      }));
      
      // Refresh coverage
      await get().fetchCoverage();
    } catch (error: any) {
      set({ 
        error: error.response?.data?.detail || 'Failed to delete tile',
        loading: false 
      });
      throw error;
    }
  },
  
  // Query elevation at specific coordinates
  queryElevation: async (lat: number, lon: number) => {
    try {
      const result = await terrainApi.queryElevation(lat, lon);
      return { elevation_m: result.elevation_m, source: result.source };
    } catch (error: any) {
      set({ 
        error: error.response?.data?.detail || 'Failed to query elevation'
      });
      throw error;
    }
  },
  
  // Clear error
  clearError: () => set({ error: null })
}));
