/**
 * Session API Service Integration Tests
 * 
 * Tests HTTP integration with the session management service.
 * Uses axios-mock-adapter to mock HTTP responses while testing real axios client behavior.
 * 
 * NOTE: Tests focus on real API behavior, not just matching mock data.
 * Edge cases and error handling are prioritized.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import MockAdapter from 'axios-mock-adapter';
import api from '@/lib/api';
import {
    listSessions,
    getSession,
    createSession,
    updateSessionStatus,
    updateSessionApproval,
    deleteSession,
    getSessionAnalytics,
    listKnownSources,
    createKnownSource,
    type RecordingSession,
    type RecordingSessionWithDetails,
    type RecordingSessionCreate,
    type SessionListResponse,
    type SessionAnalytics,
    type KnownSource,
    type KnownSourceCreate,
} from './session';

// Mock the auth store
vi.mock('@/store', () => ({
    useAuthStore: {
        getState: vi.fn(() => ({ token: null })),
    },
}));

let mock: MockAdapter;

beforeEach(() => {
    mock = new MockAdapter(api);
});

afterEach(() => {
    mock.reset();
    mock.restore();
    vi.clearAllMocks();
});

describe('listSessions', () => {
    it('should list sessions with default pagination', async () => {
        const mockResponse: SessionListResponse = {
            sessions: [
                {
                    id: 1,
                    session_name: 'Test Session 1',
                    frequency_mhz: 145.500,
                    duration_seconds: 300,
                    status: 'completed',
                    created_at: '2025-10-25T10:00:00Z',
                    source_name: 'Repeater A',
                    source_frequency: 145500000,
                    source_latitude: 45.0,
                    source_longitude: 7.6,
                    measurements_count: 125,
                    approval_status: 'approved',
                },
                {
                    id: 2,
                    session_name: 'Test Session 2',
                    frequency_mhz: 145.500,
                    duration_seconds: 180,
                    status: 'pending',
                    created_at: '2025-10-25T11:00:00Z',
                    source_name: 'Repeater B',
                    source_frequency: 145500000,
                    source_latitude: 45.1,
                    source_longitude: 7.7,
                    measurements_count: 0,
                    approval_status: 'pending',
                },
            ],
            total: 2,
            page: 1,
            per_page: 20,
        };

        mock.onGet('/api/v1/sessions').reply(200, mockResponse);

        const result = await listSessions({});

        expect(result.sessions).toHaveLength(2);
        expect(result.total).toBe(2);
        expect(result.page).toBe(1);
        expect(result.per_page).toBe(20);
    });

    it('should handle pagination parameters', async () => {
        const mockResponse: SessionListResponse = {
            sessions: [],
            total: 45,
            page: 3,
            per_page: 10,
        };

        mock.onGet('/api/v1/sessions', { params: { page: 3, per_page: 10 } }).reply(200, mockResponse);

        const result = await listSessions({ page: 3, per_page: 10 });

        expect(result.page).toBe(3);
        expect(result.per_page).toBe(10);
        expect(result.total).toBe(45);
        expect(result.sessions).toHaveLength(0); // Page 3 might be empty if total is 45 with 10 per page
    });

    it('should filter by status', async () => {
        const mockResponse: SessionListResponse = {
            sessions: [
                {
                    id: 3,
                    session_name: 'Completed Session',
                    frequency_mhz: 145.500,
                    duration_seconds: 300,
                    status: 'completed',
                    created_at: '2025-10-25T10:00:00Z',
                    measurements_count: 150,
                },
            ],
            total: 1,
            page: 1,
            per_page: 20,
        };

        mock.onGet('/api/v1/sessions', { params: { status: 'completed' } }).reply(200, mockResponse);

        const result = await listSessions({ status: 'completed' });

        expect(result.sessions).toHaveLength(1);
        expect(result.sessions[0].status).toBe('completed');
    });

    it('should filter by approval status', async () => {
        const mockResponse: SessionListResponse = {
            sessions: [
                {
                    id: 4,
                    session_name: 'Pending Approval',
                    frequency_mhz: 145.500,
                    duration_seconds: 200,
                    status: 'completed',
                    created_at: '2025-10-25T12:00:00Z',
                    approval_status: 'pending',
                    measurements_count: 100,
                },
            ],
            total: 1,
            page: 1,
            per_page: 20,
        };

        mock.onGet('/api/v1/sessions', { params: { approval_status: 'pending' } }).reply(200, mockResponse);

        const result = await listSessions({ approval_status: 'pending' });

        expect(result.sessions).toHaveLength(1);
        expect(result.sessions[0].approval_status).toBe('pending');
    });

    it('should combine multiple filters', async () => {
        const mockResponse: SessionListResponse = {
            sessions: [],
            total: 0,
            page: 1,
            per_page: 20,
        };

        mock.onGet('/api/v1/sessions', {
            params: {
                status: 'completed',
                approval_status: 'approved',
                page: 1,
                per_page: 20,
            }
        }).reply(200, mockResponse);

        const result = await listSessions({
            status: 'completed',
            approval_status: 'approved',
            page: 1,
            per_page: 20,
        });

        expect(result.total).toBe(0);
    });

    it('should handle empty results', async () => {
        const mockResponse: SessionListResponse = {
            sessions: [],
            total: 0,
            page: 1,
            per_page: 20,
        };

        mock.onGet('/api/v1/sessions').reply(200, mockResponse);

        const result = await listSessions({});

        expect(result.sessions).toHaveLength(0);
        expect(result.total).toBe(0);
    });

    it('should handle 500 error from backend', async () => {
        mock.onGet('/api/v1/sessions').reply(500, {
            detail: 'Database connection failed',
        });

        await expect(listSessions({})).rejects.toThrow();
    });

    it('should handle network errors', async () => {
        mock.onGet('/api/v1/sessions').networkError();

        await expect(listSessions({})).rejects.toThrow();
    });
});

describe('getSession', () => {
    it('should get a specific session by ID', async () => {
        const mockSession: RecordingSessionWithDetails = {
            id: 1,
            session_name: 'Detailed Session',
            frequency_mhz: 145.500,
            duration_seconds: 300,
            status: 'completed',
            created_at: '2025-10-25T10:00:00Z',
            source_name: 'Repeater A',
            source_frequency: 145500000,
            source_latitude: 45.0642,
            source_longitude: 7.6603,
            measurements_count: 125,
            approval_status: 'approved',
            notes: 'Test recording session',
        };

        mock.onGet('/api/v1/sessions/1').reply(200, mockSession);

        const result = await getSession(1);

        expect(result.id).toBe(1);
        expect(result.session_name).toBe('Detailed Session');
        expect(result.measurements_count).toBe(125);
        expect(result.notes).toBe('Test recording session');
    });

    it('should handle 404 error for non-existent session', async () => {
        mock.onGet('/api/v1/sessions/9999').reply(404, {
            detail: 'Session not found',
        });

        await expect(getSession(9999)).rejects.toThrow();
    });

    it('should handle sessions with null optional fields', async () => {
        const mockSession: RecordingSessionWithDetails = {
            id: 2,
            session_name: 'Minimal Session',
            frequency_mhz: 145.500,
            duration_seconds: 180,
            status: 'pending',
            created_at: '2025-10-25T11:00:00Z',
            celery_task_id: null,
            result_metadata: null,
            minio_path: null,
            error_message: null,
            started_at: null,
            completed_at: null,
        };

        mock.onGet('/api/v1/sessions/2').reply(200, mockSession);

        const result = await getSession(2);

        expect(result.id).toBe(2);
        expect(result.celery_task_id).toBeNull();
        expect(result.started_at).toBeNull();
    });
});

describe('createSession', () => {
    it('should create a new session successfully', async () => {
        const request: RecordingSessionCreate = {
            session_name: 'New Recording',
            frequency_mhz: 145.500,
            duration_seconds: 300,
            notes: 'Test notes',
        };

        const mockResponse: RecordingSession = {
            id: 10,
            session_name: 'New Recording',
            frequency_mhz: 145.500,
            duration_seconds: 300,
            status: 'pending',
            created_at: '2025-10-25T13:00:00Z',
        };

        mock.onPost('/api/v1/sessions').reply(200, mockResponse);

        const result = await createSession(request);

        expect(result.id).toBe(10);
        expect(result.session_name).toBe('New Recording');
        expect(result.status).toBe('pending');
        expect(result.frequency_mhz).toBe(145.500);
    });

    it('should handle validation errors (400)', async () => {
        const request: RecordingSessionCreate = {
            session_name: '',
            frequency_mhz: -1, // Invalid frequency
            duration_seconds: 0,
        };

        mock.onPost('/api/v1/sessions').reply(400, {
            detail: 'Invalid session parameters',
        });

        await expect(createSession(request)).rejects.toThrow();
    });

    it('should handle duplicate session name conflicts (409)', async () => {
        const request: RecordingSessionCreate = {
            session_name: 'Existing Session',
            frequency_mhz: 145.500,
            duration_seconds: 300,
        };

        mock.onPost('/api/v1/sessions').reply(409, {
            detail: 'Session name already exists',
        });

        await expect(createSession(request)).rejects.toThrow();
    });
});

describe('updateSessionStatus', () => {
    it('should update session status successfully', async () => {
        const mockResponse: RecordingSession = {
            id: 1,
            session_name: 'Test Session',
            frequency_mhz: 145.500,
            duration_seconds: 300,
            status: 'in_progress',
            celery_task_id: 'task-abc-123',
            created_at: '2025-10-25T10:00:00Z',
            started_at: '2025-10-25T10:05:00Z',
        };

        mock.onPatch('/api/v1/sessions/1/status').reply((config) => {
            // Verify query params
            expect(config.params?.status).toBe('in_progress');
            expect(config.params?.celery_task_id).toBe('task-abc-123');
            return [200, mockResponse];
        });

        const result = await updateSessionStatus(1, 'in_progress', 'task-abc-123');

        expect(result.status).toBe('in_progress');
        expect(result.celery_task_id).toBe('task-abc-123');
    });

    it('should update to completed status', async () => {
        const mockResponse: RecordingSession = {
            id: 1,
            session_name: 'Test Session',
            frequency_mhz: 145.500,
            duration_seconds: 300,
            status: 'completed',
            created_at: '2025-10-25T10:00:00Z',
            started_at: '2025-10-25T10:05:00Z',
            completed_at: '2025-10-25T10:10:00Z',
        };

        mock.onPatch('/api/v1/sessions/1/status').reply((config) => {
            expect(config.params?.status).toBe('completed');
            return [200, mockResponse];
        });

        const result = await updateSessionStatus(1, 'completed');

        expect(result.status).toBe('completed');
        expect(result.completed_at).toBeDefined();
    });

    it('should handle invalid status transitions', async () => {
        mock.onPatch('/api/v1/sessions/1/status').reply(400, {
            detail: 'Invalid status transition',
        });

        await expect(updateSessionStatus(1, 'invalid_status')).rejects.toThrow();
    });
});

describe('updateSessionApproval', () => {
    it('should approve a session', async () => {
        const mockResponse: RecordingSession = {
            id: 1,
            session_name: 'Test Session',
            frequency_mhz: 145.500,
            duration_seconds: 300,
            status: 'completed',
            created_at: '2025-10-25T10:00:00Z',
        };

        mock.onPatch('/api/v1/sessions/1/approval').reply((config) => {
            expect(config.params?.approval_status).toBe('approved');
            return [200, mockResponse];
        });

        const result = await updateSessionApproval(1, 'approved');

        expect(result).toBeDefined();
    });

    it('should reject a session', async () => {
        const mockResponse: RecordingSession = {
            id: 2,
            session_name: 'Rejected Session',
            frequency_mhz: 145.500,
            duration_seconds: 300,
            status: 'completed',
            created_at: '2025-10-25T10:00:00Z',
        };

        mock.onPatch('/api/v1/sessions/2/approval').reply((config) => {
            expect(config.params?.approval_status).toBe('rejected');
            return [200, mockResponse];
        });

        const result = await updateSessionApproval(2, 'rejected');

        expect(result).toBeDefined();
    });

    it('should handle 404 for non-existent session', async () => {
        mock.onPatch('/api/v1/sessions/9999/approval').reply(404);

        await expect(updateSessionApproval(9999, 'approved')).rejects.toThrow();
    });
});

describe('deleteSession', () => {
    it('should delete a session successfully', async () => {
        mock.onDelete('/api/v1/sessions/1').reply(204, '');

        await expect(deleteSession(1)).resolves.toBeUndefined();
    });

    it('should handle 404 for non-existent session', async () => {
        mock.onDelete('/api/v1/sessions/9999').reply(404, {
            detail: 'Session not found',
        });

        await expect(deleteSession(9999)).rejects.toThrow();
    });

    it('should handle 409 if session has dependent data', async () => {
        mock.onDelete('/api/v1/sessions/1').reply(409, {
            detail: 'Cannot delete session with existing measurements',
        });

        await expect(deleteSession(1)).rejects.toThrow();
    });
});

describe('getSessionAnalytics', () => {
    it('should fetch session analytics successfully', async () => {
        const mockAnalytics: SessionAnalytics = {
            total_sessions: 150,
            completed_sessions: 120,
            failed_sessions: 10,
            pending_sessions: 20,
            success_rate: 0.80,
            total_measurements: 15000,
            average_duration_seconds: 285.5,
            average_accuracy_meters: 28.3,
        };

        mock.onGet('/api/v1/sessions/analytics').reply(200, mockAnalytics);

        const result = await getSessionAnalytics();

        expect(result.total_sessions).toBe(150);
        expect(result.success_rate).toBeCloseTo(0.80, 2);
        expect(result.total_measurements).toBeGreaterThan(0);
        
        // Validate success rate calculation
        const expectedSuccessRate = result.completed_sessions / result.total_sessions;
        expect(result.success_rate).toBeCloseTo(expectedSuccessRate, 2);
    });

    it('should handle analytics with no sessions', async () => {
        const mockAnalytics: SessionAnalytics = {
            total_sessions: 0,
            completed_sessions: 0,
            failed_sessions: 0,
            pending_sessions: 0,
            success_rate: 0,
            total_measurements: 0,
        };

        mock.onGet('/api/v1/sessions/analytics').reply(200, mockAnalytics);

        const result = await getSessionAnalytics();

        expect(result.total_sessions).toBe(0);
        expect(result.success_rate).toBe(0);
    });

    it('should handle 503 error when analytics unavailable', async () => {
        mock.onGet('/api/v1/sessions/analytics').reply(503);

        await expect(getSessionAnalytics()).rejects.toThrow();
    });
});

describe('listKnownSources', () => {
    it('should list all known sources', async () => {
        const mockSources: KnownSource[] = [
            {
                id: '1',
                name: 'Repeater A',
                description: 'Main repeater',
                frequency_hz: 145500000,
                latitude: 45.0642,
                longitude: 7.6603,
                power_dbm: 50,
                source_type: 'repeater',
                is_validated: true,
                created_at: '2025-10-01T00:00:00Z',
                updated_at: '2025-10-01T00:00:00Z',
            },
            {
                id: '2',
                name: 'Beacon B',
                frequency_hz: 144800000,
                latitude: 45.1,
                longitude: 7.7,
                source_type: 'beacon',
                is_validated: false,
                created_at: '2025-10-10T00:00:00Z',
                updated_at: '2025-10-10T00:00:00Z',
            },
        ];

        mock.onGet('/api/v1/sessions/known-sources').reply(200, mockSources);

        const result = await listKnownSources();

        expect(result).toHaveLength(2);
        expect(result[0].is_validated).toBe(true);
        expect(result[1].is_validated).toBe(false);
    });

    it('should handle empty list', async () => {
        mock.onGet('/api/v1/sessions/known-sources').reply(200, []);

        const result = await listKnownSources();

        expect(result).toHaveLength(0);
    });

    it('should handle 500 error', async () => {
        mock.onGet('/api/v1/sessions/known-sources').reply(500);

        await expect(listKnownSources()).rejects.toThrow();
    });
});

describe('createKnownSource', () => {
    it('should create a new known source', async () => {
        const request: KnownSourceCreate = {
            name: 'New Repeater',
            description: 'Test repeater',
            frequency_hz: 145500000,
            latitude: 45.0,
            longitude: 7.6,
            power_dbm: 50,
            source_type: 'repeater',
            is_validated: false,
        };

        const mockResponse: KnownSource = {
            id: '123',
            ...request,
            is_validated: false,
            created_at: '2025-10-25T13:00:00Z',
            updated_at: '2025-10-25T13:00:00Z',
        };

        mock.onPost('/api/v1/sessions/known-sources').reply(200, mockResponse);

        const result = await createKnownSource(request);

        expect(result.id).toBe('123');
        expect(result.name).toBe('New Repeater');
        expect(result.frequency_hz).toBe(145500000);
    });

    it('should handle validation errors for invalid coordinates', async () => {
        const request: KnownSourceCreate = {
            name: 'Invalid Source',
            frequency_hz: 145500000,
            latitude: 999, // Invalid latitude
            longitude: 999, // Invalid longitude
        };

        mock.onPost('/api/v1/sessions/known-sources').reply(400, {
            detail: 'Invalid coordinates',
        });

        await expect(createKnownSource(request)).rejects.toThrow();
    });

    it('should handle duplicate name conflicts', async () => {
        const request: KnownSourceCreate = {
            name: 'Existing Source',
            frequency_hz: 145500000,
            latitude: 45.0,
            longitude: 7.6,
        };

        mock.onPost('/api/v1/sessions/known-sources').reply(409, {
            detail: 'Source name already exists',
        });

        await expect(createKnownSource(request)).rejects.toThrow();
    });
});

describe('Edge Cases and Real-World Scenarios', () => {
    it('should handle concurrent session listing requests', async () => {
        const mockResponse: SessionListResponse = {
            sessions: [],
            total: 10,
            page: 1,
            per_page: 20,
        };

        mock.onGet('/api/v1/sessions').reply(200, mockResponse);

        const requests = Array.from({ length: 5 }, () => listSessions({}));

        const results = await Promise.all(requests);

        expect(results).toHaveLength(5);
        results.forEach(result => {
            expect(result.total).toBe(10);
        });
    });

    it('should validate session timestamps are chronological', async () => {
        const mockSession: RecordingSessionWithDetails = {
            id: 1,
            session_name: 'Test Session',
            frequency_mhz: 145.500,
            duration_seconds: 300,
            status: 'completed',
            created_at: '2025-10-25T10:00:00Z',
            started_at: '2025-10-25T10:05:00Z',
            completed_at: '2025-10-25T10:10:00Z',
        };

        mock.onGet('/api/v1/sessions/1').reply(200, mockSession);

        const result = await getSession(1);

        const created = new Date(result.created_at!);
        const started = new Date(result.started_at!);
        const completed = new Date(result.completed_at!);

        expect(started.getTime()).toBeGreaterThanOrEqual(created.getTime());
        expect(completed.getTime()).toBeGreaterThanOrEqual(started.getTime());
    });

    it('should handle timeout during session creation', async () => {
        const request: RecordingSessionCreate = {
            session_name: 'Timeout Test',
            frequency_mhz: 145.500,
            duration_seconds: 300,
        };

        mock.onPost('/api/v1/sessions').timeout();

        await expect(createSession(request)).rejects.toThrow();
    });
});
