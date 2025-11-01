// @ts-nocheck

/**
 * Session Store
 *
 * Manages recording session state and operations
 */

import { create } from 'zustand';
import type {
  RecordingSession,
  RecordingSessionWithDetails,
  RecordingSessionCreate,
  RecordingSessionUpdate,
  SessionAnalytics,
  KnownSource,
  KnownSourceCreate,
  KnownSourceUpdate,
} from '@/services/api/session';
import { sessionService } from '@/services/api';

interface SessionStore {
  sessions: RecordingSessionWithDetails[];
  currentSession: RecordingSessionWithDetails | null;
  knownSources: KnownSource[];
  analytics: SessionAnalytics | null;

  isLoading: boolean;
  error: string | null;

  // Pagination
  currentPage: number;
  totalSessions: number;
  perPage: number;

  // Filters
  statusFilter: string | null;
  approvalFilter: string | null;

  // Actions
  fetchSessions: (params?: {
    page?: number;
    per_page?: number;
    status?: string;
    approval_status?: string;
  }) => Promise<void>;

  fetchSession: (sessionId: string) => Promise<void>;

  createSession: (session: RecordingSessionCreate) => Promise<RecordingSession>;

  updateSession: (sessionId: string, sessionUpdate: RecordingSessionUpdate) => Promise<void>;

  updateSessionStatus: (sessionId: string, status: string, celeryTaskId?: string) => Promise<void>;

  approveSession: (sessionId: string) => Promise<void>;
  rejectSession: (sessionId: string) => Promise<void>;

  deleteSession: (sessionId: string) => Promise<void>;

  fetchAnalytics: () => Promise<void>;

  fetchKnownSources: () => Promise<void>;
  createKnownSource: (source: KnownSourceCreate) => Promise<KnownSource>;
  updateKnownSource: (sourceId: string, source: KnownSourceUpdate) => Promise<KnownSource>;
  deleteKnownSource: (sourceId: string) => Promise<void>;

  setStatusFilter: (status: string | null) => void;
  setApprovalFilter: (approval: string | null) => void;

  clearError: () => void;
}

export const useSessionStore = create<SessionStore>((set, get) => ({
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

  fetchSessions: async params => {
    set({ isLoading: true, error: null });
    try {
      const response = await sessionService.listSessions({
        page: params?.page || get().currentPage,
        per_page: params?.per_page || get().perPage,
        status: params?.status || get().statusFilter || undefined,
        approval_status: params?.approval_status || get().approvalFilter || undefined,
      });

      set({
        sessions: response.sessions,
        totalSessions: response.total,
        currentPage: response.page,
        perPage: response.per_page,
        isLoading: false,
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch sessions';
      set({ error: errorMessage, isLoading: false });
      console.error('Session fetch error:', error);
    }
  },

  fetchSession: async (sessionId: string) => {
    set({ isLoading: true, error: null });
    try {
      const session = await sessionService.getSession(sessionId);
      set({ currentSession: session, isLoading: false });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch session';
      set({ error: errorMessage, isLoading: false });
      console.error('Session fetch error:', error);
    }
  },

  createSession: async (session: RecordingSessionCreate) => {
    set({ isLoading: true, error: null });
    try {
      const newSession = await sessionService.createSession(session);
      set({ isLoading: false });

      // Refresh session list
      await get().fetchSessions();

      return newSession;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to create session';
      set({ error: errorMessage, isLoading: false });
      console.error('Session creation error:', error);
      throw error;
    }
  },

  updateSession: async (sessionId: string, sessionUpdate: RecordingSessionUpdate) => {
    set({ isLoading: true, error: null });
    try {
      await sessionService.updateSession(sessionId, sessionUpdate);
      set({ isLoading: false });

      // Refresh session list
      await get().fetchSessions();

      // Refresh current session if it's the one being updated
      if (get().currentSession?.id === sessionId) {
        await get().fetchSession(sessionId);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to update session';
      set({ error: errorMessage, isLoading: false });
      console.error('Session update error:', error);
      throw error;
    }
  },

  updateSessionStatus: async (sessionId: string, status: string, celeryTaskId?: string) => {
    try {
      await sessionService.updateSessionStatus(sessionId, status, celeryTaskId);

      // Refresh session list
      await get().fetchSessions();

      // Refresh current session if it's the one being updated
      if (get().currentSession?.id === sessionId) {
        await get().fetchSession(sessionId);
      }
    } catch (error) {
      console.error('Session status update error:', error);
      throw error;
    }
  },

  approveSession: async (sessionId: string) => {
    try {
      await sessionService.updateSessionApproval(sessionId, 'approved');

      // Refresh session list
      await get().fetchSessions();

      // Refresh current session if it's the one being approved
      if (get().currentSession?.id === sessionId) {
        await get().fetchSession(sessionId);
      }
    } catch (error) {
      console.error('Session approval error:', error);
      throw error;
    }
  },

  rejectSession: async (sessionId: string) => {
    try {
      await sessionService.updateSessionApproval(sessionId, 'rejected');

      // Refresh session list
      await get().fetchSessions();

      // Refresh current session if it's the one being rejected
      if (get().currentSession?.id === sessionId) {
        await get().fetchSession(sessionId);
      }
    } catch (error) {
      console.error('Session rejection error:', error);
      throw error;
    }
  },

  deleteSession: async (sessionId: string) => {
    try {
      await sessionService.deleteSession(sessionId);

      // Refresh session list
      await get().fetchSessions();

      // Clear current session if it was deleted
      if (get().currentSession?.id === sessionId) {
        set({ currentSession: null });
      }
    } catch (error) {
      console.error('Session deletion error:', error);
      throw error;
    }
  },

  fetchAnalytics: async () => {
    try {
      const analytics = await sessionService.getSessionAnalytics();
      set({ analytics });
    } catch (error) {
      console.error('Analytics fetch error:', error);
      // Don't set error for analytics, it's not critical
    }
  },

  fetchKnownSources: async () => {
    try {
      const sources = await sessionService.listKnownSources();
      set({ knownSources: sources });
    } catch (error) {
      console.error('Known sources fetch error:', error);
      // Don't set error for known sources, it's not critical
    }
  },

  createKnownSource: async (source: KnownSourceCreate) => {
    try {
      const newSource = await sessionService.createKnownSource(source);

      // Refresh known sources list
      await get().fetchKnownSources();

      return newSource;
    } catch (error) {
      console.error('Known source creation error:', error);
      throw error;
    }
  },

  updateKnownSource: async (sourceId: string, source: KnownSourceUpdate) => {
    try {
      const updatedSource = await sessionService.updateKnownSource(sourceId, source);

      // Refresh known sources list
      await get().fetchKnownSources();

      return updatedSource;
    } catch (error) {
      console.error('Known source update error:', error);
      throw error;
    }
  },

  deleteKnownSource: async (sourceId: string) => {
    try {
      await sessionService.deleteKnownSource(sourceId);

      // Refresh known sources list
      await get().fetchKnownSources();
    } catch (error) {
      console.error('Known source deletion error:', error);
      throw error;
    }
  },

  setStatusFilter: (status: string | null) => {
    set({ statusFilter: status, currentPage: 1 });
    get().fetchSessions();
  },

  setApprovalFilter: (approval: string | null) => {
    set({ approvalFilter: approval, currentPage: 1 });
    get().fetchSessions();
  },

  clearError: () => set({ error: null }),
}));
