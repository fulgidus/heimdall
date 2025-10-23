import { vi } from 'vitest';

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

export const useSessionStore = vi.fn(() => mockSessionStore);
