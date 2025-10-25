import { vi } from 'vitest';

const mockDashboardStore = {
    data: {
        modelInfo: {
            accuracy: 0.28,
            predictions_total: 150,
            predictions_successful: 135,
        },
        servicesHealth: {
            'api-gateway': { status: 'healthy', latency_ms: 10 },
            'rf-acquisition': { status: 'healthy', latency_ms: 50 },
            'training': { status: 'healthy', latency_ms: 30 },
            'inference': { status: 'healthy', latency_ms: 45 },
            'data-ingestion-web': { status: 'healthy', latency_ms: 20 },
        },
    },
    metrics: {
        accuracy_history: [0.25, 0.26, 0.27, 0.28],
        latency_avg_ms: 150,
    },
    isLoading: false,
    error: null,
    fetchDashboardData: vi.fn(),
};

const mockWebSDRStore = {
    websdrs: Array(7).fill(null).map((_, i) => ({
        id: i + 1,
        name: `WebSDR ${i + 1}`,
        url: `http://localhost:800${i}`,
        country: 'Italy',
        location: `Receiver ${i + 1}`,
        location_name: `City ${i + 1}, Italy`,
        latitude: 45.0 + i * 0.1,
        longitude: 7.0 + i * 0.1,
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
    loading: false,
    error: null,
    fetchWebSDRs: vi.fn(),
    checkHealth: vi.fn(),
    refreshAll: vi.fn(),
    lastHealthCheck: new Date().toISOString(),
};

const mockSessionStore = {
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
    fetchKnownSources: vi.fn(),
    fetchSessions: vi.fn(),
    fetchAnalytics: vi.fn(),
    clearError: vi.fn(),
};

const mockAuthStore = {
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
    logout: vi.fn(),
    updateProfile: vi.fn(),
};

const mockAnalyticsStore = {
    predictionMetrics: {
        total_predictions: [{ timestamp: '2025-10-24T00:00:00Z', value: 100 }],
        successful_predictions: [{ timestamp: '2025-10-24T00:00:00Z', value: 85 }],
        failed_predictions: [{ timestamp: '2025-10-24T00:00:00Z', value: 15 }],
        average_confidence: [{ timestamp: '2025-10-24T00:00:00Z', value: 0.85 }],
        average_uncertainty: [{ timestamp: '2025-10-24T00:00:00Z', value: 25 }],
    },
    websdrPerformance: [],
    systemPerformance: null,
    accuracyDistribution: {},
    isLoading: false,
    error: null,
    timeRange: '24h',
    setTimeRange: vi.fn(),
    fetchPredictionMetrics: vi.fn(),
    fetchWebSDRPerformance: vi.fn(),
    fetchSystemPerformance: vi.fn(),
    fetchAccuracyDistribution: vi.fn(),
    fetchAllAnalytics: vi.fn(),
    refreshData: vi.fn(),
};

const mockLocalizationStore = {
    recentLocalizations: [],
    isLoading: false,
    error: null,
    predictLocalization: vi.fn(),
    predictLocalizationBatch: vi.fn(),
    fetchRecentLocalizations: vi.fn(),
};

const mockAcquisitionStore = {
    currentTask: null,
    taskHistory: [],
    isLoading: false,
    error: null,
    triggerAcquisition: vi.fn(),
    pollTaskStatus: vi.fn(),
    clearCurrentTask: vi.fn(),
};

const mockSystemStore = {
    health: {
        status: 'healthy',
        services: {
            postgres: 'healthy',
            rabbitmq: 'healthy',
            redis: 'healthy',
            minio: 'healthy',
        },
    },
    isLoading: false,
    error: null,
    fetchSystemHealth: vi.fn(),
};

export const useDashboardStore = vi.fn(() => mockDashboardStore);
export const useWebSDRStore = vi.fn(() => mockWebSDRStore);
export const useSessionStore = vi.fn(() => mockSessionStore);
export const useAuthStore = vi.fn(() => mockAuthStore);
export const useAnalyticsStore = vi.fn(() => mockAnalyticsStore);
export const useLocalizationStore = vi.fn(() => mockLocalizationStore);
export const useAcquisitionStore = vi.fn(() => mockAcquisitionStore);
export const useSystemStore = vi.fn(() => mockSystemStore);
