/**
 * Session Store Tests
 *
 * Comprehensive test suite for the sessionStore Zustand store
 * Tests all actions: CRUD operations, filtering, pagination, and state management
 * Truth-first approach: Tests real Zustand store behavior with mocked API responses
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

// Unmock the stores module for this test (we want to test the real store)
vi.unmock('@/store');
vi.unmock('@/store/sessionStore');

// Import after unmocking
import { useSessionStore } from './sessionStore';
import { sessionService } from '@/services/api';

// Mock the session service
vi.mock('@/services/api', () => ({
  sessionService: {
    listSessions: vi.fn(),
    getSession: vi.fn(),
    createSession: vi.fn(),
    updateSessionStatus: vi.fn(),
    updateSessionApproval: vi.fn(),
    deleteSession: vi.fn(),
    getSessionAnalytics: vi.fn(), // Fixed: was getAnalytics
    listKnownSources: vi.fn(),
    createKnownSource: vi.fn(),
  },
  webSDRService: {},
  acquisitionService: {},
  inferenceService: {},
  systemService: {},
  analyticsService: {},
}));

describe('Session Store (Zustand)', () => {
  beforeEach(() => {
    // Reset store to initial state before each test
    useSessionStore.setState({
      sessions: [],
      currentSession: null,
      knownSources: [],
      analytics: null,
      isLoading: false,
      error: null,
      currentPage: 1,
      totalSessions: 0,
      perPage: 20,
      statusFilter: null,
      approvalFilter: null,
    });
    vi.clearAllMocks();
  });

  describe('Store Initialization', () => {
    it('should initialize with default state', () => {
      const state = useSessionStore.getState();
      expect(state.sessions).toEqual([]);
      expect(state.currentSession).toBe(null);
      expect(state.knownSources).toEqual([]);
      expect(state.analytics).toBe(null);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBe(null);
      expect(state.currentPage).toBe(1);
      expect(state.totalSessions).toBe(0);
      expect(state.perPage).toBe(20);
      expect(state.statusFilter).toBe(null);
      expect(state.approvalFilter).toBe(null);
    });

    it('should have all required actions', () => {
      const state = useSessionStore.getState();
      expect(typeof state.fetchSessions).toBe('function');
      expect(typeof state.fetchSession).toBe('function');
      expect(typeof state.createSession).toBe('function');
      expect(typeof state.updateSessionStatus).toBe('function');
      expect(typeof state.approveSession).toBe('function');
      expect(typeof state.rejectSession).toBe('function');
      expect(typeof state.deleteSession).toBe('function');
      expect(typeof state.fetchAnalytics).toBe('function');
      expect(typeof state.fetchKnownSources).toBe('function');
      expect(typeof state.createKnownSource).toBe('function');
      expect(typeof state.setStatusFilter).toBe('function');
      expect(typeof state.setApprovalFilter).toBe('function');
      expect(typeof state.clearError).toBe('function');
    });
  });

  describe('fetchSessions Action', () => {
    it('should fetch sessions successfully', async () => {
      const mockResponse = {
        sessions: [
          {
            id: 1,
            known_source_id: 1,
            frequency_mhz: 145.5,
            status: 'completed',
            approval_status: 'approved',
            created_at: '2025-01-01T00:00:00Z',
            measurements_count: 5,
          },
          {
            id: 2,
            known_source_id: 2,
            frequency_mhz: 435.0,
            status: 'pending',
            approval_status: 'pending',
            created_at: '2025-01-02T00:00:00Z',
            measurements_count: 3,
          },
        ],
        total: 2,
        page: 1,
        per_page: 20,
      };

      vi.mocked(sessionService.listSessions).mockResolvedValue(mockResponse);

      await useSessionStore.getState().fetchSessions();

      const state = useSessionStore.getState();
      expect(state.sessions).toEqual(mockResponse.sessions);
      expect(state.totalSessions).toBe(2);
      expect(state.currentPage).toBe(1);
      expect(state.perPage).toBe(20);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBe(null);
      expect(sessionService.listSessions).toHaveBeenCalledOnce();
    });

    it('should set loading state during fetch', async () => {
      const mockResponse = { sessions: [], total: 0, page: 1, per_page: 20 };
      vi.mocked(sessionService.listSessions).mockResolvedValue(mockResponse);

      const promise = useSessionStore.getState().fetchSessions();

      // Should be loading immediately
      expect(useSessionStore.getState().isLoading).toBe(true);

      await promise;

      // Should not be loading after completion
      expect(useSessionStore.getState().isLoading).toBe(false);
    });

    it('should handle pagination parameters', async () => {
      const mockResponse = { sessions: [], total: 50, page: 2, per_page: 10 };
      vi.mocked(sessionService.listSessions).mockResolvedValue(mockResponse);

      await useSessionStore.getState().fetchSessions({ page: 2, per_page: 10 });

      expect(sessionService.listSessions).toHaveBeenCalledWith({
        page: 2,
        per_page: 10,
        status: undefined,
        approval_status: undefined,
      });
    });

    it('should handle status filter', async () => {
      const mockResponse = { sessions: [], total: 0, page: 1, per_page: 20 };
      vi.mocked(sessionService.listSessions).mockResolvedValue(mockResponse);

      useSessionStore.setState({ statusFilter: 'completed' });
      await useSessionStore.getState().fetchSessions();

      expect(sessionService.listSessions).toHaveBeenCalledWith(
        expect.objectContaining({ status: 'completed' })
      );
    });

    it('should handle approval filter', async () => {
      const mockResponse = { sessions: [], total: 0, page: 1, per_page: 20 };
      vi.mocked(sessionService.listSessions).mockResolvedValue(mockResponse);

      useSessionStore.setState({ approvalFilter: 'approved' });
      await useSessionStore.getState().fetchSessions();

      expect(sessionService.listSessions).toHaveBeenCalledWith(
        expect.objectContaining({ approval_status: 'approved' })
      );
    });

    it('should handle fetch error gracefully', async () => {
      const errorMessage = 'Failed to fetch sessions';
      vi.mocked(sessionService.listSessions).mockRejectedValue(new Error(errorMessage));

      await useSessionStore.getState().fetchSessions();

      const state = useSessionStore.getState();
      expect(state.error).toBe(errorMessage);
      expect(state.isLoading).toBe(false);
    });
  });

  describe('fetchSession Action (Single Session)', () => {
    it('should fetch single session successfully', async () => {
      const mockSession = {
        id: 1,
        known_source_id: 1,
        frequency_mhz: 145.5,
        status: 'completed',
        approval_status: 'approved',
        created_at: '2025-01-01T00:00:00Z',
        measurements_count: 5,
        measurements: [
          {
            id: 1,
            websdr_id: 1,
            snr: 15.5,
            frequency_offset_hz: 100,
            timestamp: '2025-01-01T00:00:00Z',
          },
        ],
      };

      vi.mocked(sessionService.getSession).mockResolvedValue(mockSession);

      await useSessionStore.getState().fetchSession(1);

      const state = useSessionStore.getState();
      expect(state.currentSession).toEqual(mockSession);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBe(null);
      expect(sessionService.getSession).toHaveBeenCalledWith(1);
    });

    it('should handle fetch session error', async () => {
      const errorMessage = 'Session not found';
      vi.mocked(sessionService.getSession).mockRejectedValue(new Error(errorMessage));

      await useSessionStore.getState().fetchSession(999);

      const state = useSessionStore.getState();
      expect(state.error).toBe(errorMessage);
      expect(state.currentSession).toBe(null);
      expect(state.isLoading).toBe(false);
    });
  });

  describe('createSession Action', () => {
    it('should create session successfully', async () => {
      const newSessionData = {
        known_source_id: 1,
        frequency_mhz: 145.5,
        duration_seconds: 60,
      };

      const createdSession = {
        id: 3,
        ...newSessionData,
        status: 'pending',
        approval_status: 'pending',
        created_at: '2025-01-03T00:00:00Z',
      };

      vi.mocked(sessionService.createSession).mockResolvedValue(createdSession);
      vi.mocked(sessionService.listSessions).mockResolvedValue({
        sessions: [createdSession],
        total: 1,
        page: 1,
        per_page: 20,
      });

      const result = await useSessionStore.getState().createSession(newSessionData);

      expect(result).toEqual(createdSession);
      expect(sessionService.createSession).toHaveBeenCalledWith(newSessionData);
      // Should refresh session list after creation
      expect(sessionService.listSessions).toHaveBeenCalled();
    });

    it('should handle create session error', async () => {
      const errorMessage = 'Failed to create session';
      vi.mocked(sessionService.createSession).mockRejectedValue(new Error(errorMessage));

      const newSessionData = {
        known_source_id: 1,
        frequency_mhz: 145.5,
        duration_seconds: 60,
      };

      await expect(useSessionStore.getState().createSession(newSessionData)).rejects.toThrow(
        errorMessage
      );

      const state = useSessionStore.getState();
      expect(state.error).toBe(errorMessage);
      expect(state.isLoading).toBe(false);
    });
  });

  describe('updateSessionStatus Action', () => {
    it('should update session status successfully', async () => {
      vi.mocked(sessionService.updateSessionStatus).mockResolvedValue(undefined);
      vi.mocked(sessionService.listSessions).mockResolvedValue({
        sessions: [],
        total: 0,
        page: 1,
        per_page: 20,
      });

      await useSessionStore.getState().updateSessionStatus(1, 'completed', 'task-123');

      expect(sessionService.updateSessionStatus).toHaveBeenCalledWith(1, 'completed', 'task-123');
      // Should refresh session list after update
      expect(sessionService.listSessions).toHaveBeenCalled();
    });

    it('should refresh current session if updated', async () => {
      const mockSession = {
        id: 1,
        status: 'completed',
        approval_status: 'approved',
      };

      useSessionStore.setState({ currentSession: mockSession });

      vi.mocked(sessionService.updateSessionStatus).mockResolvedValue(undefined);
      vi.mocked(sessionService.getSession).mockResolvedValue(mockSession);
      vi.mocked(sessionService.listSessions).mockResolvedValue({
        sessions: [mockSession],
        total: 1,
        page: 1,
        per_page: 20,
      });

      await useSessionStore.getState().updateSessionStatus(1, 'completed');

      // Should refresh both list and current session
      expect(sessionService.listSessions).toHaveBeenCalled();
      expect(sessionService.getSession).toHaveBeenCalledWith(1);
    });
  });

  describe('approveSession Action', () => {
    it('should approve session successfully', async () => {
      vi.mocked(sessionService.updateSessionApproval).mockResolvedValue(undefined);
      vi.mocked(sessionService.listSessions).mockResolvedValue({
        sessions: [],
        total: 0,
        page: 1,
        per_page: 20,
      });

      await useSessionStore.getState().approveSession(1);

      expect(sessionService.updateSessionApproval).toHaveBeenCalledWith(1, 'approved');
      expect(sessionService.listSessions).toHaveBeenCalled();
    });
  });

  describe('rejectSession Action', () => {
    it('should reject session successfully', async () => {
      vi.mocked(sessionService.updateSessionApproval).mockResolvedValue(undefined);
      vi.mocked(sessionService.listSessions).mockResolvedValue({
        sessions: [],
        total: 0,
        page: 1,
        per_page: 20,
      });

      await useSessionStore.getState().rejectSession(1);

      expect(sessionService.updateSessionApproval).toHaveBeenCalledWith(1, 'rejected');
      expect(sessionService.listSessions).toHaveBeenCalled();
    });
  });

  describe('deleteSession Action', () => {
    it('should delete session successfully', async () => {
      vi.mocked(sessionService.deleteSession).mockResolvedValue(undefined);
      vi.mocked(sessionService.listSessions).mockResolvedValue({
        sessions: [],
        total: 0,
        page: 1,
        per_page: 20,
      });

      await useSessionStore.getState().deleteSession(1);

      expect(sessionService.deleteSession).toHaveBeenCalledWith(1);
      // Should refresh session list after deletion
      expect(sessionService.listSessions).toHaveBeenCalled();
    });
  });

  describe('Filter Actions', () => {
    it('should set status filter', () => {
      useSessionStore.getState().setStatusFilter('completed');
      expect(useSessionStore.getState().statusFilter).toBe('completed');

      useSessionStore.getState().setStatusFilter(null);
      expect(useSessionStore.getState().statusFilter).toBe(null);
    });

    it('should set approval filter', () => {
      useSessionStore.getState().setApprovalFilter('approved');
      expect(useSessionStore.getState().approvalFilter).toBe('approved');

      useSessionStore.getState().setApprovalFilter(null);
      expect(useSessionStore.getState().approvalFilter).toBe(null);
    });
  });

  describe('Error Handling', () => {
    it('should clear error', () => {
      useSessionStore.setState({ error: 'Some error' });
      useSessionStore.getState().clearError();
      expect(useSessionStore.getState().error).toBe(null);
    });
  });

  describe('Known Sources Management', () => {
    it('should fetch known sources successfully', async () => {
      const mockSources = [
        {
          id: 1,
          name: 'Test Source 1',
          frequency_mhz: 145.5,
          latitude: 45.0,
          longitude: 7.5,
        },
        {
          id: 2,
          name: 'Test Source 2',
          frequency_mhz: 435.0,
          latitude: 44.0,
          longitude: 8.0,
        },
      ];

      vi.mocked(sessionService.listKnownSources).mockResolvedValue(mockSources);

      await useSessionStore.getState().fetchKnownSources();

      const state = useSessionStore.getState();
      expect(state.knownSources).toEqual(mockSources);
      expect(sessionService.listKnownSources).toHaveBeenCalled();
    });

    it('should create known source successfully', async () => {
      const newSource = {
        name: 'New Source',
        frequency_mhz: 145.5,
        latitude: 45.0,
        longitude: 7.5,
      };

      const createdSource = { id: 3, ...newSource };

      vi.mocked(sessionService.createKnownSource).mockResolvedValue(createdSource);
      vi.mocked(sessionService.listKnownSources).mockResolvedValue([createdSource]);

      const result = await useSessionStore.getState().createKnownSource(newSource);

      expect(result).toEqual(createdSource);
      expect(sessionService.createKnownSource).toHaveBeenCalledWith(newSource);
      // Should refresh known sources list
      expect(sessionService.listKnownSources).toHaveBeenCalled();
    });
  });

  describe('Analytics', () => {
    it('should fetch analytics successfully', async () => {
      const mockAnalytics = {
        total_sessions: 100,
        approved_sessions: 75,
        rejected_sessions: 15,
        pending_sessions: 10,
        average_measurements_per_session: 5.5,
      };

      vi.mocked(sessionService.getSessionAnalytics).mockResolvedValue(mockAnalytics);

      await useSessionStore.getState().fetchAnalytics();

      const state = useSessionStore.getState();
      expect(state.analytics).toEqual(mockAnalytics);
      expect(sessionService.getSessionAnalytics).toHaveBeenCalled();
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty session list', async () => {
      const mockResponse = { sessions: [], total: 0, page: 1, per_page: 20 };
      vi.mocked(sessionService.listSessions).mockResolvedValue(mockResponse);

      await useSessionStore.getState().fetchSessions();

      const state = useSessionStore.getState();
      expect(state.sessions).toEqual([]);
      expect(state.totalSessions).toBe(0);
    });

    it('should handle large page numbers', async () => {
      const mockResponse = { sessions: [], total: 1000, page: 50, per_page: 20 };
      vi.mocked(sessionService.listSessions).mockResolvedValue(mockResponse);

      await useSessionStore.getState().fetchSessions({ page: 50 });

      expect(useSessionStore.getState().currentPage).toBe(50);
    });

    it('should handle multiple filters simultaneously', async () => {
      const mockResponse = { sessions: [], total: 0, page: 1, per_page: 20 };
      vi.mocked(sessionService.listSessions).mockResolvedValue(mockResponse);

      useSessionStore.setState({
        statusFilter: 'completed',
        approvalFilter: 'approved',
      });

      await useSessionStore.getState().fetchSessions();

      expect(sessionService.listSessions).toHaveBeenCalledWith(
        expect.objectContaining({
          status: 'completed',
          approval_status: 'approved',
        })
      );
    });
  });
});
