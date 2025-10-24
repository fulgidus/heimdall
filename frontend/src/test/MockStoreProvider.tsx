import React, { type ReactNode } from 'react';
import { create } from 'zustand';

// Create mock stores using Zustand's create() function

export const createMockDashboardStore = () => create(() => ({
    data: {
        modelInfo: {
            accuracy: 0.28,
            predictions_total: 150,
            predictions_successful: 135,
        },
        servicesHealth: {
            'api-gateway': { status: 'healthy', latency_ms: 10 },
            'rf-acquisition': { status: 'healthy', latency_ms: 50 },
            training: { status: 'healthy', latency_ms: 30 },
            inference: { status: 'healthy', latency_ms: 45 },
        },
    },
    metrics: {
        accuracy_history: [0.25, 0.26, 0.27, 0.28],
        latency_avg_ms: 150,
    },
    isLoading: false,
    error: null,
    fetchDashboardData: () => { },
}));

export const createMockWebSDRStore = () => create(() => ({
    websdrs: Array(7).fill(null).map((_, i) => ({
        id: i + 1,
        name: `WebSDR ${i + 1}`,
        url: `http://localhost:800${i}`,
        country: 'Italy',
        location: `Receiver ${i + 1}`,
        is_active: true,
    })),
    healthStatus: Object.fromEntries(
        Array(7).fill(null).map((_, i) => [
            `${i + 1}`,
            { status: 'online', response_time_ms: 150 + i * 10 },
        ])
    ),
    statistics: {
        online_count: 7,
        total_count: 7,
        active_count: 7,
        avg_response_time_ms: 151,
    },
    isLoading: false,
    error: null,
    fetchWebSDRs: () => { },
    checkHealth: () => { },
    refreshAll: () => { },
    lastHealthCheck: new Date().toISOString(),
}));

export const createMockSessionStore = () => create(() => ({
    knownSources: [
        { id: 1, name: 'Source 1', frequency_mhz: 145.5, is_validated: true },
        { id: 2, name: 'Source 2', frequency_mhz: 430.5, is_validated: false },
    ],
    sessions: [
        { id: 1, session_name: 'Session 1', status: 'completed', created_at: new Date().toISOString() },
        { id: 2, session_name: 'Session 2', status: 'pending', created_at: new Date().toISOString() },
    ],
    analytics: {
        total_sessions: 10,
        completed_sessions: 8,
        total_measurements: 100,
    },
    pagination: {
        page: 1,
        pageSize: 10,
        total: 2,
    },
    isLoading: false,
    error: null,
    fetchKnownSources: () => { },
    fetchSessions: () => { },
    fetchAnalytics: () => { },
    clearError: () => { },
}));

export const createMockAuthStore = () => create(() => ({
    user: {
        email: 'admin@heimdall.local',
        firstName: 'Admin',
        lastName: 'User',
        role: 'administrator',
        organization: 'Heimdall SDR',
        createdAt: new Date().toISOString(),
    },
    isAuthenticated: true,
    isLoading: false,
    error: null,
    logout: () => { },
    updateProfile: () => { },
}));

interface MockStoreProviderProps {
    children: ReactNode;
}

// Export stores globally during testing
let mockDashboardStore = createMockDashboardStore();
let mockWebSDRStore = createMockWebSDRStore();
let mockSessionStore = createMockSessionStore();
let mockAuthStore = createMockAuthStore();

export const getMockStores = () => ({
    mockDashboardStore,
    mockWebSDRStore,
    mockSessionStore,
    mockAuthStore,
});

export const resetMockStores = () => {
    mockDashboardStore = createMockDashboardStore();
    mockWebSDRStore = createMockWebSDRStore();
    mockSessionStore = createMockSessionStore();
    mockAuthStore = createMockAuthStore();
};

export const MockStoreProvider: React.FC<MockStoreProviderProps> = ({ children }) => {
    // This provider doesn't actually need to render anything special
    // The mock stores are injected via vi.mock() in setup.ts
    return <>{children}</>;
};
