import { vi } from 'vitest';

/**
 * Helper per creare mock di store Zustand per i test
 */

export const createAuthStoreMock = () => ({
    user: { email: 'admin@heimdall.local', id: '1', role: 'admin' },
    isAuthenticated: true,
    login: vi.fn(),
    logout: vi.fn(),
    setUser: vi.fn(),
});

export const createDashboardStoreMock = () => ({
    data: {
        modelInfo: {
            accuracy: 0.28,
            predictions_total: 150,
            predictions_successful: 135,
        },
        servicesHealth: {
            'api-gateway': { status: 'healthy', latency_ms: 10 },
            'backend': { status: 'healthy', latency_ms: 50 },
            'training': { status: 'healthy', latency_ms: 100 },
            'inference': { status: 'healthy', latency_ms: 30 },
            'data-ingestion-web': { status: 'healthy', latency_ms: 20 },
        },
    },
    metrics: {
        active_websdrs: 7,
        total_signals: 42,
        system_uptime: 168,
        accuracy: 95.2,
    },
    isLoading: false,
    error: null,
    fetchDashboardData: vi.fn(),
    lastUpdate: new Date().toISOString(),
});

export const createWebSDRStoreMock = () => ({
    websdrs: [
        { id: 1, name: 'Turin', location_name: 'Turin, Italy', frequency_start: 137, frequency_end: 138, is_active: true },
        { id: 2, name: 'Milan', location_name: 'Milan, Italy', frequency_start: 137, frequency_end: 138, is_active: true },
        { id: 3, name: 'Genoa', location_name: 'Genoa, Italy', frequency_start: 137, frequency_end: 138, is_active: true },
        { id: 4, name: 'Alessandria', location_name: 'Alessandria, Italy', frequency_start: 137, frequency_end: 138, is_active: true },
        { id: 5, name: 'Asti', location_name: 'Asti, Italy', frequency_start: 137, frequency_end: 138, is_active: true },
        { id: 6, name: 'La Spezia', location_name: 'La Spezia, Italy', frequency_start: 137, frequency_end: 138, is_active: true },
        { id: 7, name: 'Piacenza', location_name: 'Piacenza, Italy', frequency_start: 137, frequency_end: 138, is_active: true },
    ],
    healthStatus: {
        '1': { status: 'online', response_time_ms: 150 },
        '2': { status: 'online', response_time_ms: 140 },
        '3': { status: 'online', response_time_ms: 160 },
        '4': { status: 'online', response_time_ms: 155 },
        '5': { status: 'online', response_time_ms: 145 },
        '6': { status: 'online', response_time_ms: 150 },
        '7': { status: 'online', response_time_ms: 158 },
    },
    isLoading: false,
    error: null,
    fetchWebSDRs: vi.fn(),
    checkHealth: vi.fn(),
    refreshAll: vi.fn(),
    lastHealthCheck: new Date().toISOString(),
});

export const createSessionStoreMock = () => ({
    knownSources: [
        { id: 1, name: 'Source 1', frequency_mhz: 145.5, is_validated: true },
        { id: 2, name: 'Source 2', frequency_mhz: 430.5, is_validated: false },
    ],
    sessions: [
        { id: 1, session_name: 'Session 1', status: 'completed', created_at: new Date().toISOString() },
        { id: 2, session_name: 'Session 2', status: 'pending', created_at: new Date().toISOString() },
    ],
    analytics: {
        total_sessions: 100,
        completed_sessions: 85,
        failed_sessions: 5,
        pending_sessions: 10,
    },
    currentPage: 1,
    totalSessions: 100,
    perPage: 10,
    statusFilter: null,
    isLoading: false,
    error: null,
    fetchKnownSources: vi.fn(),
    fetchSessions: vi.fn(),
    fetchAnalytics: vi.fn(),
    setStatusFilter: vi.fn(),
    createSession: vi.fn(),
    deleteSession: vi.fn(),
    clearError: vi.fn(),
});
