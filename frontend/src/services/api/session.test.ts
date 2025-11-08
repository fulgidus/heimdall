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
          id: '123e4567-e89b-12d3-a456-426614174001',
          known_source_id: '223e4567-e89b-12d3-a456-426614174000',
          session_name: 'Test Session 1',
          session_start: '2025-10-25T10:00:00Z',
          duration_seconds: 300,
          status: 'completed',
          created_at: '2025-10-25T10:00:00Z',
          updated_at: '2025-10-25T10:05:00Z',
          source_name: 'Repeater A',
          source_frequency: 145500000,
          source_latitude: 45.0,
          source_longitude: 7.6,
          measurements_count: 125,
          approval_status: 'approved',
        },
        {
          id: '123e4567-e89b-12d3-a456-426614174002',
          known_source_id: '223e4567-e89b-12d3-a456-426614174000',
          session_name: 'Test Session 2',
          session_start: '2025-10-25T11:00:00Z',
          duration_seconds: 180,
          status: 'pending',
          created_at: '2025-10-25T11:00:00Z',
          updated_at: '2025-10-25T11:00:00Z',
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

    mock.onGet('/v1/sessions').reply(200, mockResponse);

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

    mock.onGet('/v1/sessions', { params: { page: 3, per_page: 10 } }).reply(200, mockResponse);

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
          id: '123e4567-e89b-12d3-a456-426614174003',
          known_source_id: '223e4567-e89b-12d3-a456-426614174000',
          session_name: 'Completed Session',
          session_start: '2025-10-25T10:00:00Z',
          duration_seconds: 300,
          status: 'completed',
          created_at: '2025-10-25T10:00:00Z',
          updated_at: '2025-10-25T10:05:00Z',
          measurements_count: 150,
        },
      ],
      total: 1,
      page: 1,
      per_page: 20,
    };

    mock.onGet('/v1/sessions', { params: { status: 'completed' } }).reply(200, mockResponse);

    const result = await listSessions({ status: 'completed' });

    expect(result.sessions).toHaveLength(1);
    expect(result.sessions[0].status).toBe('completed');
  });

  it('should filter by approval status', async () => {
    const mockResponse: SessionListResponse = {
      sessions: [
        {
          id: '123e4567-e89b-12d3-a456-426614174004',
          known_source_id: '223e4567-e89b-12d3-a456-426614174000',
          session_name: 'Pending Approval',
          session_start: '2025-10-25T12:00:00Z',
          duration_seconds: 200,
          status: 'completed',
          created_at: '2025-10-25T12:00:00Z',
          updated_at: '2025-10-25T12:03:00Z',
          approval_status: 'pending',
          measurements_count: 100,
        },
      ],
      total: 1,
      page: 1,
      per_page: 20,
    };

    mock
      .onGet('/v1/sessions', { params: { approval_status: 'pending' } })
      .reply(200, mockResponse);

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

    mock
      .onGet('/v1/sessions', {
        params: {
          status: 'completed',
          approval_status: 'approved',
          page: 1,
          per_page: 20,
        },
      })
      .reply(200, mockResponse);

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

    mock.onGet('/v1/sessions').reply(200, mockResponse);

    const result = await listSessions({});

    expect(result.sessions).toHaveLength(0);
    expect(result.total).toBe(0);
  });

  it('should handle 500 error from backend', async () => {
    mock.onGet('/v1/sessions').reply(500, {
      detail: 'Database connection failed',
    });

    await expect(listSessions({})).rejects.toThrow();
  });

  it('should handle network errors', async () => {
    mock.onGet('/v1/sessions').networkError();

    await expect(listSessions({})).rejects.toThrow();
  });
});

describe('getSession', () => {
  it('should get a specific session by ID', async () => {
    const mockSession: RecordingSessionWithDetails = {
      id: '123e4567-e89b-12d3-a456-426614174001',
      known_source_id: '223e4567-e89b-12d3-a456-426614174000',
      session_name: 'Detailed Session',
      session_start: '2025-10-25T10:00:00Z',
      duration_seconds: 300,
      status: 'completed',
      created_at: '2025-10-25T10:00:00Z',
      updated_at: '2025-10-25T10:05:00Z',
      source_name: 'Repeater A',
      source_frequency: 145500000,
      source_latitude: 45.0642,
      source_longitude: 7.6603,
      measurements_count: 125,
      approval_status: 'approved',
      notes: 'Test recording session',
    };

    mock.onGet('/v1/sessions/123e4567-e89b-12d3-a456-426614174001').reply(200, mockSession);

    const result = await getSession('123e4567-e89b-12d3-a456-426614174001');

    expect(result.id).toBe('123e4567-e89b-12d3-a456-426614174001');
    expect(result.session_name).toBe('Detailed Session');
    expect(result.measurements_count).toBe(125);
    expect(result.notes).toBe('Test recording session');
  });

  it('should handle 404 error for non-existent session', async () => {
    mock.onGet('/v1/sessions/999e4567-e89b-12d3-a456-426614174999').reply(404, {
      detail: 'Session not found',
    });

    await expect(getSession('999e4567-e89b-12d3-a456-426614174999')).rejects.toThrow();
  });

  it('should handle sessions with null optional fields', async () => {
    const mockSession: RecordingSessionWithDetails = {
      id: '123e4567-e89b-12d3-a456-426614174002',
      known_source_id: '223e4567-e89b-12d3-a456-426614174000',
      session_name: 'Minimal Session',
      session_start: '2025-10-25T11:00:00Z',
      session_end: null,
      duration_seconds: 180,
      status: 'pending',
      created_at: '2025-10-25T11:00:00Z',
      updated_at: '2025-10-25T11:00:00Z',
      celery_task_id: null,
    };

    mock.onGet('/v1/sessions/123e4567-e89b-12d3-a456-426614174002').reply(200, mockSession);

    const result = await getSession('123e4567-e89b-12d3-a456-426614174002');

    expect(result.id).toBe('123e4567-e89b-12d3-a456-426614174002');
    expect(result.celery_task_id).toBeNull();
    expect(result.session_end).toBeNull();
    expect(result.notes).toBeUndefined();
  });
});

describe('createSession', () => {
  it('should create a new session successfully', async () => {
    const request: RecordingSessionCreate = {
      known_source_id: '123e4567-e89b-12d3-a456-426614174000',
      session_name: 'New Recording',
      frequency_hz: 145500000, // 145.500 MHz in Hz
      duration_seconds: 300,
      notes: 'Test notes',
    };

    const mockResponse: RecordingSession = {
      id: '123e4567-e89b-12d3-a456-426614174010',
      known_source_id: '123e4567-e89b-12d3-a456-426614174000',
      session_name: 'New Recording',
      session_start: '2025-10-25T13:00:00Z',
      duration_seconds: 300,
      status: 'pending',
      created_at: '2025-10-25T13:00:00Z',
      updated_at: '2025-10-25T13:00:00Z',
    };

    mock.onPost('/v1/sessions').reply(200, mockResponse);

    const result = await createSession(request);

    expect(result.id).toBe('123e4567-e89b-12d3-a456-426614174010');
    expect(result.session_name).toBe('New Recording');
    expect(result.status).toBe('pending');
  });

  it('should handle validation errors (400)', async () => {
    const request: RecordingSessionCreate = {
      known_source_id: '123e4567-e89b-12d3-a456-426614174000',
      session_name: '',
      frequency_hz: -1, // Invalid frequency
      duration_seconds: 0,
    };

    mock.onPost('/v1/sessions').reply(400, {
      detail: 'Invalid session parameters',
    });

    await expect(createSession(request)).rejects.toThrow();
  });

  it('should handle duplicate session name conflicts (409)', async () => {
    const request: RecordingSessionCreate = {
      known_source_id: '123e4567-e89b-12d3-a456-426614174000',
      session_name: 'Existing Session',
      frequency_hz: 145500000, // 145.500 MHz in Hz
      duration_seconds: 300,
    };

    mock.onPost('/v1/sessions').reply(409, {
      detail: 'Session name already exists',
    });

    await expect(createSession(request)).rejects.toThrow();
  });
});

describe('updateSessionStatus', () => {
  it('should update session status successfully', async () => {
    const mockResponse: RecordingSession = {
      id: '123e4567-e89b-12d3-a456-426614174001',
      known_source_id: '223e4567-e89b-12d3-a456-426614174000',
      session_name: 'Test Session',
      session_start: '2025-10-25T10:00:00Z',
      duration_seconds: 300,
      status: 'in_progress',
      celery_task_id: 'task-abc-123',
      created_at: '2025-10-25T10:00:00Z',
      updated_at: '2025-10-25T10:05:00Z',
    };

    mock.onPatch('/v1/sessions/123e4567-e89b-12d3-a456-426614174001/status').reply(config => {
      // Verify query params
      expect(config.params?.status).toBe('in_progress');
      expect(config.params?.celery_task_id).toBe('task-abc-123');
      return [200, mockResponse];
    });

    const result = await updateSessionStatus(
      '123e4567-e89b-12d3-a456-426614174001',
      'in_progress',
      'task-abc-123'
    );

    expect(result.status).toBe('in_progress');
    expect(result.celery_task_id).toBe('task-abc-123');
  });

  it('should update to completed status', async () => {
    const mockResponse: RecordingSession = {
      id: '123e4567-e89b-12d3-a456-426614174001',
      known_source_id: '223e4567-e89b-12d3-a456-426614174000',
      session_name: 'Test Session',
      session_start: '2025-10-25T10:00:00Z',
      session_end: '2025-10-25T10:10:00Z',
      duration_seconds: 300,
      status: 'completed',
      created_at: '2025-10-25T10:00:00Z',
      updated_at: '2025-10-25T10:10:00Z',
    };

    mock.onPatch('/v1/sessions/123e4567-e89b-12d3-a456-426614174001/status').reply(config => {
      expect(config.params?.status).toBe('completed');
      return [200, mockResponse];
    });

    const result = await updateSessionStatus('123e4567-e89b-12d3-a456-426614174001', 'completed');

    expect(result.status).toBe('completed');
    expect(result.session_end).toBeDefined();
  });

  it('should handle invalid status transitions', async () => {
    mock.onPatch('/v1/sessions/123e4567-e89b-12d3-a456-426614174001/status').reply(400, {
      detail: 'Invalid status transition',
    });

    await expect(
      updateSessionStatus('123e4567-e89b-12d3-a456-426614174001', 'invalid_status')
    ).rejects.toThrow();
  });
});

describe('updateSessionApproval', () => {
  it('should approve a session', async () => {
    const mockResponse: RecordingSession = {
      id: '123e4567-e89b-12d3-a456-426614174001',
      known_source_id: '223e4567-e89b-12d3-a456-426614174000',
      session_name: 'Test Session',
      session_start: '2025-10-25T10:00:00Z',
      duration_seconds: 300,
      status: 'completed',
      created_at: '2025-10-25T10:00:00Z',
      updated_at: '2025-10-25T10:05:00Z',
    };

    mock.onPatch('/v1/sessions/123e4567-e89b-12d3-a456-426614174001/approval').reply(config => {
      expect(config.params?.approval_status).toBe('approved');
      return [200, mockResponse];
    });

    const result = await updateSessionApproval('123e4567-e89b-12d3-a456-426614174001', 'approved');

    expect(result).toBeDefined();
  });

  it('should reject a session', async () => {
    const mockResponse: RecordingSession = {
      id: '123e4567-e89b-12d3-a456-426614174002',
      known_source_id: '223e4567-e89b-12d3-a456-426614174000',
      session_name: 'Rejected Session',
      session_start: '2025-10-25T10:00:00Z',
      duration_seconds: 300,
      status: 'completed',
      created_at: '2025-10-25T10:00:00Z',
      updated_at: '2025-10-25T10:05:00Z',
    };

    mock.onPatch('/v1/sessions/123e4567-e89b-12d3-a456-426614174002/approval').reply(config => {
      expect(config.params?.approval_status).toBe('rejected');
      return [200, mockResponse];
    });

    const result = await updateSessionApproval('123e4567-e89b-12d3-a456-426614174002', 'rejected');

    expect(result).toBeDefined();
  });

  it('should handle 404 for non-existent session', async () => {
    mock.onPatch('/v1/sessions/999e4567-e89b-12d3-a456-426614174999/approval').reply(404);

    await expect(
      updateSessionApproval('999e4567-e89b-12d3-a456-426614174999', 'approved')
    ).rejects.toThrow();
  });
});

describe('deleteSession', () => {
  it('should delete a session successfully', async () => {
    mock.onDelete('/v1/sessions/123e4567-e89b-12d3-a456-426614174001').reply(204, '');

    await expect(deleteSession('123e4567-e89b-12d3-a456-426614174001')).resolves.toBeUndefined();
  });

  it('should handle 404 for non-existent session', async () => {
    mock.onDelete('/v1/sessions/999e4567-e89b-12d3-a456-426614174999').reply(404, {
      detail: 'Session not found',
    });

    await expect(deleteSession('999e4567-e89b-12d3-a456-426614174999')).rejects.toThrow();
  });

  it('should handle 409 if session has dependent data', async () => {
    mock.onDelete('/v1/sessions/123e4567-e89b-12d3-a456-426614174001').reply(409, {
      detail: 'Cannot delete session with existing measurements',
    });

    await expect(deleteSession('123e4567-e89b-12d3-a456-426614174001')).rejects.toThrow();
  });
});

describe('getSessionAnalytics', () => {
  it('should fetch session analytics successfully', async () => {
    const mockAnalytics: SessionAnalytics = {
      total_sessions: 150,
      completed_sessions: 120,
      failed_sessions: 10,
      pending_sessions: 20,
      success_rate: 80,
      total_measurements: 15000,
      average_duration_seconds: 285.5,
    };

    mock.onGet('/v1/sessions/analytics').reply(200, mockAnalytics);

    const result = await getSessionAnalytics();

    expect(result.total_sessions).toBe(150);
    expect(result.success_rate).toBeCloseTo(80, 2);
    expect(result.total_measurements).toBeGreaterThan(0);

    // Validate success rate is a percentage
    expect(result.success_rate).toBeGreaterThanOrEqual(0);
    expect(result.success_rate).toBeLessThanOrEqual(100);
  });

  it('should handle analytics with no sessions', async () => {
    const mockAnalytics: SessionAnalytics = {
      total_sessions: 0,
      completed_sessions: 0,
      failed_sessions: 0,
      pending_sessions: 0,
      success_rate: 0,
      total_measurements: 0,
      average_duration_seconds: 0,
    };

    mock.onGet('/v1/sessions/analytics').reply(200, mockAnalytics);

    const result = await getSessionAnalytics();

    expect(result.total_sessions).toBe(0);
    expect(result.success_rate).toBe(0);
  });

  it('should handle 503 error when analytics unavailable', async () => {
    mock.onGet('/v1/sessions/analytics').reply(503);

    await expect(getSessionAnalytics()).rejects.toThrow();
  });
});

describe('listKnownSources', () => {
  it('should list all known sources', async () => {
    const mockSources: KnownSource[] = [
      {
        id: '123e4567-e89b-12d3-a456-426614174001',
        name: 'Repeater A',
        description: 'Main repeater',
        frequency_hz: 145500000,
        latitude: 45.0642,
        longitude: 7.6603,
        power_dbm: 50,
        source_type: 'repeater',
        is_validated: true,
        error_margin_meters: 30,
        created_at: '2025-10-01T00:00:00Z',
        updated_at: '2025-10-01T00:00:00Z',
      },
      {
        id: '123e4567-e89b-12d3-a456-426614174002',
        name: 'Beacon B',
        frequency_hz: 144800000,
        latitude: 45.1,
        longitude: 7.7,
        source_type: 'beacon',
        is_validated: false,
        error_margin_meters: 50,
        created_at: '2025-10-10T00:00:00Z',
        updated_at: '2025-10-10T00:00:00Z',
      },
    ];

    mock.onGet('/v1/sessions/known-sources').reply(200, mockSources);

    const result = await listKnownSources();

    expect(result).toHaveLength(2);
    expect(result[0].is_validated).toBe(true);
    expect(result[1].is_validated).toBe(false);
  });

  it('should handle empty list', async () => {
    mock.onGet('/v1/sessions/known-sources').reply(200, []);

    const result = await listKnownSources();

    expect(result).toHaveLength(0);
  });

  it('should handle 500 error', async () => {
    mock.onGet('/v1/sessions/known-sources').reply(500);

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
      id: '123e4567-e89b-12d3-a456-426614174123',
      ...request,
      is_validated: false,
      error_margin_meters: 0,
      created_at: '2025-10-25T13:00:00Z',
      updated_at: '2025-10-25T13:00:00Z',
    };

    mock.onPost('/v1/sessions/known-sources').reply(200, mockResponse);

    const result = await createKnownSource(request);

    expect(result.id).toBe('123e4567-e89b-12d3-a456-426614174123');
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

    mock.onPost('/v1/sessions/known-sources').reply(400, {
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

    mock.onPost('/v1/sessions/known-sources').reply(409, {
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

    mock.onGet('/v1/sessions').reply(200, mockResponse);

    const requests = Array.from({ length: 5 }, () => listSessions({}));

    const results = await Promise.all(requests);

    expect(results).toHaveLength(5);
    results.forEach(result => {
      expect(result.total).toBe(10);
    });
  });

  it('should validate session timestamps are chronological', async () => {
    const mockSession: RecordingSessionWithDetails = {
      id: '123e4567-e89b-12d3-a456-426614174001',
      known_source_id: '223e4567-e89b-12d3-a456-426614174000',
      session_name: 'Test Session',
      session_start: '2025-10-25T10:00:00Z',
      session_end: '2025-10-25T10:10:00Z',
      duration_seconds: 300,
      status: 'completed',
      created_at: '2025-10-25T10:00:00Z',
      updated_at: '2025-10-25T10:10:00Z',
    };

    mock.onGet('/v1/sessions/123e4567-e89b-12d3-a456-426614174001').reply(200, mockSession);

    const result = await getSession('123e4567-e89b-12d3-a456-426614174001');

    const created = new Date(result.created_at);
    const started = new Date(result.session_start);
    const completed = new Date(result.session_end!);

    expect(started.getTime()).toBeGreaterThanOrEqual(created.getTime());
    expect(completed.getTime()).toBeGreaterThanOrEqual(started.getTime());
  });

  it('should handle timeout during session creation', async () => {
    const request: RecordingSessionCreate = {
      known_source_id: '123e4567-e89b-12d3-a456-426614174000',
      session_name: 'Timeout Test',
      frequency_hz: 145500000, // 145.500 MHz in Hz
      duration_seconds: 300,
    };

    mock.onPost('/v1/sessions').timeout();

    await expect(createSession(request)).rejects.toThrow();
  });
});
