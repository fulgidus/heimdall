/**
 * Settings Store
 * Manages user settings state using Zustand
 */

import { create } from 'zustand';
import { getUserSettings, updateUserSettings, resetUserSettings, UserSettings, UserSettingsUpdate } from '../services/api/settings';

interface SettingsStore {
  settings: UserSettings | null;
  isLoading: boolean;
  error: string | null;
  isSaving: boolean;
  
  // Actions
  fetchSettings: () => Promise<void>;
  updateSettings: (settings: UserSettingsUpdate) => Promise<void>;
  resetSettings: () => Promise<void>;
  clearError: () => void;
}

export const useSettingsStore = create<SettingsStore>((set, get) => ({
  settings: null,
  isLoading: false,
  error: null,
  isSaving: false,

  fetchSettings: async () => {
    set({ isLoading: true, error: null });
    try {
      const settings = await getUserSettings();
      set({ settings, isLoading: false });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch settings';
      set({ error: errorMessage, isLoading: false });
      throw error;
    }
  },

  updateSettings: async (settingsUpdate: UserSettingsUpdate) => {
    set({ isSaving: true, error: null });
    try {
      const settings = await updateUserSettings(settingsUpdate);
      set({ settings, isSaving: false });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to update settings';
      set({ error: errorMessage, isSaving: false });
      throw error;
    }
  },

  resetSettings: async () => {
    set({ isSaving: true, error: null });
    try {
      await resetUserSettings();
      // After reset, fetch the new default settings
      await get().fetchSettings();
      set({ isSaving: false });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to reset settings';
      set({ error: errorMessage, isSaving: false });
      throw error;
    }
  },

  clearError: () => set({ error: null }),
}));
