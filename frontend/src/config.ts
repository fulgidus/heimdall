export interface Config {
  apiUrl: string;
  environment: 'development' | 'staging' | 'production';
  mapboxToken: string;
  logLevel: 'debug' | 'info' | 'warn' | 'error';
  enableAnalytics: boolean;
  enableFeatures: {
    realtimeUpdates: boolean;
    userAuth: boolean;
    advancedFilters: boolean;
  };
}

const getConfig = (): Config => {
  const env = import.meta.env.MODE;
  
  return {
    apiUrl: import.meta.env.VITE_API_URL || 'http://localhost:8000',
    environment: (env as any) || 'development',
    mapboxToken: import.meta.env.VITE_MAPBOX_TOKEN || '',
    logLevel: (import.meta.env.VITE_LOG_LEVEL as any) || 'info',
    enableAnalytics: import.meta.env.VITE_ANALYTICS === 'true',
    enableFeatures: {
      realtimeUpdates: import.meta.env.VITE_FEATURE_REALTIME === 'true',
      userAuth: import.meta.env.VITE_FEATURE_AUTH === 'true',
      advancedFilters: import.meta.env.VITE_FEATURE_FILTERS === 'true',
    },
  };
};

export const config = getConfig();
export default config;
