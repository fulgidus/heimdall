/**
 * Mock API response fixtures for component testing
 */

export const mockServiceHealth = {
    'api-gateway': { status: 'healthy', version: '1.0.0', latency_ms: 10 },
    'rf-acquisition': { status: 'healthy', version: '1.0.0', latency_ms: 50 },
    'inference': { status: 'healthy', version: '1.0.0', latency_ms: 30 },
    'training': { status: 'healthy', version: '1.0.0', latency_ms: 100 },
};

export const mockServiceHealthUnhealthy = {
    'api-gateway': { status: 'unhealthy', version: '1.0.0', latency_ms: 10 },
    'rf-acquisition': { status: 'degraded', version: '1.0.0', latency_ms: 250 },
    'inference': { status: 'healthy', version: '1.0.0', latency_ms: 30 },
};

export const mockSessions = [
    {
        id: 1,
        session_name: 'Test Session 1',
        timestamp: '2025-10-25T10:00:00Z',
        status: 'pending',
        frequency_mhz: 145.5,
        duration_seconds: 60,
        known_source_id: 1,
        created_at: '2025-10-25T10:00:00Z',
    },
    {
        id: 2,
        session_name: 'Test Session 2',
        timestamp: '2025-10-25T11:00:00Z',
        status: 'completed',
        frequency_mhz: 430.5,
        duration_seconds: 120,
        known_source_id: 2,
        created_at: '2025-10-25T11:00:00Z',
    },
    {
        id: 3,
        session_name: 'Test Session 3',
        timestamp: '2025-10-25T12:00:00Z',
        status: 'failed',
        frequency_mhz: 145.8,
        duration_seconds: 30,
        known_source_id: 1,
        created_at: '2025-10-25T12:00:00Z',
    },
];

export const mockWebSDRs = [
    {
        id: 1,
        name: 'Turin',
        url: 'http://websdr-turin.example.com',
        location_name: 'Turin, Italy',
        latitude: 45.0703,
        longitude: 7.6869,
        frequency_start: 137,
        frequency_end: 138,
        is_active: true,
        status: 'online',
    },
    {
        id: 2,
        name: 'Milan',
        url: 'http://websdr-milan.example.com',
        location_name: 'Milan, Italy',
        latitude: 45.4642,
        longitude: 9.1900,
        frequency_start: 137,
        frequency_end: 138,
        is_active: true,
        status: 'online',
    },
    {
        id: 3,
        name: 'Genoa',
        url: 'http://websdr-genoa.example.com',
        location_name: 'Genoa, Italy',
        latitude: 44.4056,
        longitude: 8.9463,
        frequency_start: 137,
        frequency_end: 138,
        is_active: true,
        status: 'online',
    },
    {
        id: 4,
        name: 'Alessandria',
        url: 'http://websdr-alessandria.example.com',
        location_name: 'Alessandria, Italy',
        latitude: 44.9136,
        longitude: 8.6147,
        frequency_start: 137,
        frequency_end: 138,
        is_active: true,
        status: 'online',
    },
    {
        id: 5,
        name: 'Asti',
        url: 'http://websdr-asti.example.com',
        location_name: 'Asti, Italy',
        latitude: 44.9009,
        longitude: 8.2065,
        frequency_start: 137,
        frequency_end: 138,
        is_active: true,
        status: 'online',
    },
    {
        id: 6,
        name: 'La Spezia',
        url: 'http://websdr-laspezia.example.com',
        location_name: 'La Spezia, Italy',
        latitude: 44.1023,
        longitude: 9.8246,
        frequency_start: 137,
        frequency_end: 138,
        is_active: true,
        status: 'online',
    },
    {
        id: 7,
        name: 'Piacenza',
        url: 'http://websdr-piacenza.example.com',
        location_name: 'Piacenza, Italy',
        latitude: 45.0526,
        longitude: 9.6930,
        frequency_start: 137,
        frequency_end: 138,
        is_active: true,
        status: 'online',
    },
];

export const mockLocalizations = [
    {
        id: 1,
        session_id: 2,
        estimated_latitude: 45.1234,
        estimated_longitude: 7.5678,
        uncertainty_meters: 25.5,
        confidence: 0.95,
        timestamp: '2025-10-25T11:00:30Z',
    },
    {
        id: 2,
        session_id: 2,
        estimated_latitude: 45.4567,
        estimated_longitude: 9.1234,
        uncertainty_meters: 30.2,
        confidence: 0.92,
        timestamp: '2025-10-25T11:01:00Z',
    },
];

export const mockKnownSources = [
    {
        id: 1,
        name: 'Test Beacon 1',
        frequency_mhz: 145.5,
        frequency_hz: 145500000,
        latitude: 45.1234,
        longitude: 7.5678,
        is_validated: true,
        description: 'Test beacon for validation',
    },
    {
        id: 2,
        name: 'Test Beacon 2',
        frequency_mhz: 430.5,
        frequency_hz: 430500000,
        latitude: 45.5678,
        longitude: 7.1234,
        is_validated: false,
        description: 'Unvalidated test beacon',
    },
];

export const mockAnalytics = {
    total_sessions: 100,
    completed_sessions: 85,
    failed_sessions: 5,
    pending_sessions: 10,
    success_rate: 85.0,
    avg_localization_accuracy_m: 28.5,
    total_measurements: 425,
};

export const mockPredictionMetrics = {
    total_predictions: 150,
    successful_predictions: 135,
    failed_predictions: 15,
    avg_accuracy_m: 28.5,
    avg_uncertainty_m: 12.3,
    accuracy_history: [
        { timestamp: '2025-10-25T08:00:00Z', accuracy_m: 30.2 },
        { timestamp: '2025-10-25T09:00:00Z', accuracy_m: 28.1 },
        { timestamp: '2025-10-25T10:00:00Z', accuracy_m: 26.8 },
        { timestamp: '2025-10-25T11:00:00Z', accuracy_m: 28.5 },
    ],
};

export const mockWebSDRPerformance = {
    total_acquisitions: 500,
    successful_acquisitions: 475,
    failed_acquisitions: 25,
    avg_response_time_ms: 150,
    by_websdr: [
        { websdr_id: 1, name: 'Turin', success_rate: 96, avg_response_time_ms: 140 },
        { websdr_id: 2, name: 'Milan', success_rate: 95, avg_response_time_ms: 150 },
        { websdr_id: 3, name: 'Genoa', success_rate: 97, avg_response_time_ms: 145 },
        { websdr_id: 4, name: 'Alessandria', success_rate: 94, avg_response_time_ms: 155 },
        { websdr_id: 5, name: 'Asti', success_rate: 95, avg_response_time_ms: 148 },
        { websdr_id: 6, name: 'La Spezia', success_rate: 96, avg_response_time_ms: 152 },
        { websdr_id: 7, name: 'Piacenza', success_rate: 93, avg_response_time_ms: 160 },
    ],
};
