import '@testing-library/jest-dom';
import { beforeEach, beforeAll, vi } from 'vitest';
import React from 'react';
import {
  createMockDashboardStore,
  createMockWebSDRStore,
  createMockSessionStore,
  createMockAuthStore,
  createMockSystemStore,
} from './mockStoreFactories';

// ============================================
// MAKE REACT AVAILABLE GLOBALLY FOR JSX IN TESTS
// ============================================
// React 19 with new JSX transform doesn't require React import in source files,
// but test files using JSX syntax need React available globally
declare global {
  // Extend globalThis to include React for test JSX
  var React: typeof React;
}
globalThis.React = React;

// ============================================
// SETUP MOCKS - MUST BE BEFORE COMPONENT IMPORTS
// ============================================

const mockDashboardStore = createMockDashboardStore();
const mockWebSDRStore = createMockWebSDRStore();
const mockSessionStore = createMockSessionStore();
const mockAuthStore = createMockAuthStore();
const mockSystemStore = createMockSystemStore();

// Mock all stores GLOBALLY
vi.mock('../store', () => ({
  useDashboardStore: () => ({
    ...mockDashboardStore.getState(),
    ...mockDashboardStore,
  }),
  useWebSDRStore: () => ({
    ...mockWebSDRStore.getState(),
    ...mockWebSDRStore,
  }),
  useSessionStore: () => ({
    ...mockSessionStore.getState(),
    ...mockSessionStore,
  }),
  useAuthStore: () => ({
    ...mockAuthStore.getState(),
    ...mockAuthStore,
  }),
  useSystemStore: () => ({
    ...mockSystemStore.getState(),
    ...mockSystemStore,
  }),
}));

// Export the mocks so they can be used in individual tests
vi.mock('../store/dashboardStore', () => ({
  useDashboardStore: () => ({
    ...mockDashboardStore.getState(),
    ...mockDashboardStore,
  }),
}));

vi.mock('../store/authStore', () => ({
  useAuthStore: () => ({
    ...mockAuthStore.getState(),
    ...mockAuthStore,
  }),
}));

vi.mock('../store/sessionStore', () => ({
  useSessionStore: () => ({
    ...mockSessionStore.getState(),
    ...mockSessionStore,
  }),
}));

vi.mock('../store/websdrStore', () => ({
  useWebSDRStore: () => ({
    ...mockWebSDRStore.getState(),
    ...mockWebSDRStore,
  }),
}));

vi.mock('../store/systemStore', () => ({
  useSystemStore: () => ({
    ...mockSystemStore.getState(),
    ...mockSystemStore,
  }),
}));

// ============================================
// API SERVICE MOCKS
// ============================================

// Mock inference service
vi.mock('../services/api/inference', () => ({
  inferenceService: {
    getModelInfo: vi.fn(() =>
      Promise.resolve({
        active_version: '1.0.0',
        stage: 'production',
        model_name: 'localization-net',
        accuracy: 0.95,
        latency_p95_ms: 150,
        cache_hit_rate: 0.8,
        loaded_at: new Date().toISOString(),
        uptime_seconds: 3600,
        last_prediction_at: new Date().toISOString(),
        predictions_total: 1000,
        predictions_successful: 950,
        predictions_failed: 50,
        is_ready: true,
        health_status: 'healthy',
      })
    ),
    getModelPerformance: vi.fn(() =>
      Promise.resolve({
        inference_latency_ms: 150,
        p50_latency_ms: 120,
        p95_latency_ms: 200,
        p99_latency_ms: 250,
        throughput_samples_per_second: 100,
        cache_hit_rate: 0.8,
        success_rate: 0.95,
        predictions_total: 1000,
        requests_total: 1050,
        errors_total: 50,
        uptime_seconds: 3600,
        timestamp: new Date().toISOString(),
      })
    ),
  },
}));

// ============================================
// WEBSOCKET CONTEXT MOCK
// ============================================

// Mock WebSocket context
vi.mock('../contexts/WebSocketContext', () => ({
  WebSocketProvider: ({ children }: { children: React.ReactNode }) => children,
  useWebSocket: () => ({
    manager: {
      send: vi.fn(),
      isConnected: () => true,
      connect: vi.fn(),
      disconnect: vi.fn(),
      subscribe: vi.fn(() => vi.fn()),
      unsubscribe: vi.fn(),
    },
    connectionState: 'CONNECTED',
    isConnected: true,
    connect: vi.fn(),
    disconnect: vi.fn(),
    subscribe: vi.fn(() => vi.fn()),
    unsubscribe: vi.fn(),
  }),
}));

// ============================================
// WINDOW & ENV MOCKS
// ============================================

// Mock window.matchMedia
beforeAll(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => true,
    }),
  });
});

// Mock environment variables
Object.defineProperty(import.meta, 'env', {
  value: {
    VITE_ADMIN_EMAIL: 'admin@heimdall.local',
    VITE_ADMIN_PASSWORD: 'Admin123!@#',
    VITE_API_URL: 'http://localhost:8000/api',
  },
  writable: true,
});

// Clear localStorage before each test
beforeEach(() => {
  localStorage.clear();
  vi.clearAllMocks();
});
