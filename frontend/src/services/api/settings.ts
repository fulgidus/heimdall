/**
 * Settings API Service
 * Handles user settings operations
 */

import api from '../../lib/api';

export interface UserSettings {
  id: string;
  user_id: string;
  // General settings
  theme: 'dark' | 'light' | 'auto';
  language: 'en' | 'it';
  timezone: string;
  auto_refresh: boolean;
  refresh_interval: number;
  // API settings
  api_timeout: number;
  retry_attempts: number;
  enable_caching: boolean;
  // Notification settings
  email_notifications: boolean;
  system_alerts: boolean;
  performance_warnings: boolean;
  webhook_url: string | null;
  // Advanced settings
  debug_mode: boolean;
  log_level: 'error' | 'warn' | 'info' | 'debug';
  max_concurrent_requests: number;
  // Metadata
  created_at: string;
  updated_at: string;
}

export interface UserSettingsUpdate {
  // General settings
  theme?: 'dark' | 'light' | 'auto';
  language?: 'en' | 'it';
  timezone?: string;
  auto_refresh?: boolean;
  refresh_interval?: number;
  // API settings
  api_timeout?: number;
  retry_attempts?: number;
  enable_caching?: boolean;
  // Notification settings
  email_notifications?: boolean;
  system_alerts?: boolean;
  performance_warnings?: boolean;
  webhook_url?: string | null;
  // Advanced settings
  debug_mode?: boolean;
  log_level?: 'error' | 'warn' | 'info' | 'debug';
  max_concurrent_requests?: number;
}

/**
 * Get current user's settings
 */
export const getUserSettings = async (): Promise<UserSettings> => {
  const response = await api.get<UserSettings>('/settings/');
  return response.data;
};

/**
 * Update user settings
 */
export const updateUserSettings = async (settings: UserSettingsUpdate): Promise<UserSettings> => {
  const response = await api.put<UserSettings>('/settings/', settings);
  return response.data;
};

/**
 * Reset settings to defaults
 */
export const resetUserSettings = async (): Promise<void> => {
  await api.delete('/settings/');
};
