import { create } from 'zustand';
import axios from 'axios';

export interface RecordingSession {
    id: number;
    session_name: string;
    frequency_mhz: number;
    duration_seconds: number;
    status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
    celery_task_id: string | null;
    result_metadata: Record<string, any> | null;
    minio_path: string | null;
    error_message: string | null;
    created_at: string;
    started_at: string | null;
    completed_at: string | null;
    websdrs_enabled: number;
}

export interface SessionStatus {
    session_id: number;
    status: string;
    progress: number;
    created_at: string;
    started_at: string | null;
    completed_at: string | null;
    error_message: string | null;
}

interface SessionStore {
    // State
    sessions: RecordingSession[];
    currentSession: RecordingSession | null;
    isLoading: boolean;
    error: string | null;

    // Actions
    createSession: (name: string, frequency: number, duration: number) => Promise<RecordingSession>;
    fetchSessions: (offset?: number, limit?: number) => Promise<void>;
    getSession: (sessionId: number) => Promise<RecordingSession>;
    getSessionStatus: (sessionId: number) => Promise<SessionStatus>;
    pollSessionStatus: (sessionId: number, intervalMs?: number) => Promise<void>;
    clearError: () => void;
    reset: () => void;
}

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const useSessionStore = create<SessionStore>((set, get) => ({
    // Initial state
    sessions: [],
    currentSession: null,
    isLoading: false,
    error: null,

    // Actions
    createSession: async (name: string, frequency: number, duration: number) => {
        set({ isLoading: true, error: null });
        try {
            const response = await axios.post(
                `${API_BASE_URL}/api/sessions/create`,
                {
                    session_name: name,
                    frequency_mhz: frequency,
                    duration_seconds: duration,
                }
            );
            const newSession = response.data;
            set((state) => ({
                sessions: [newSession, ...state.sessions],
                currentSession: newSession,
                isLoading: false,
            }));
            return newSession;
        } catch (err: any) {
            const errorMsg = err.response?.data?.detail || err.message || 'Failed to create session';
            set({ error: errorMsg, isLoading: false });
            throw new Error(errorMsg);
        }
    },

    fetchSessions: async (offset = 0, limit = 20) => {
        set({ isLoading: true, error: null });
        try {
            const response = await axios.get(`${API_BASE_URL}/api/sessions`, {
                params: { offset, limit },
            });
            set({ sessions: response.data.sessions, isLoading: false });
        } catch (err: any) {
            const errorMsg = err.response?.data?.detail || err.message || 'Failed to fetch sessions';
            set({ error: errorMsg, isLoading: false });
        }
    },

    getSession: async (sessionId: number) => {
        set({ isLoading: true, error: null });
        try {
            const response = await axios.get(`${API_BASE_URL}/api/sessions/${sessionId}`);
            const session = response.data;
            set({ currentSession: session, isLoading: false });
            return session;
        } catch (err: any) {
            const errorMsg = err.response?.data?.detail || err.message || 'Failed to fetch session';
            set({ error: errorMsg, isLoading: false });
            throw new Error(errorMsg);
        }
    },

    getSessionStatus: async (sessionId: number) => {
        try {
            const response = await axios.get(`${API_BASE_URL}/api/sessions/${sessionId}/status`);
            return response.data;
        } catch (err: any) {
            const errorMsg = err.response?.data?.detail || err.message || 'Failed to fetch session status';
            set({ error: errorMsg });
            throw new Error(errorMsg);
        }
    },

    pollSessionStatus: async (sessionId: number, intervalMs = 2000) => {
        return new Promise((resolve, reject) => {
            let maxAttempts = 300; // 10 minutes max with 2s intervals
            let attempts = 0;

            const pollInterval = setInterval(async () => {
                try {
                    const status = await get().getSessionStatus(sessionId);

                    // Update session status in the list
                    set((state) => ({
                        sessions: state.sessions.map((s) =>
                            s.id === sessionId ? { ...s, status: status.status as any } : s
                        ),
                        currentSession:
                            state.currentSession?.id === sessionId
                                ? { ...state.currentSession, status: status.status as any }
                                : state.currentSession,
                    }));

                    // Stop polling when completed or failed
                    if (status.status === 'completed' || status.status === 'failed') {
                        clearInterval(pollInterval);
                        resolve(undefined);
                    }

                    attempts++;
                    if (attempts >= maxAttempts) {
                        clearInterval(pollInterval);
                        reject(new Error('Polling timeout'));
                    }
                } catch (err) {
                    clearInterval(pollInterval);
                    reject(err);
                }
            }, intervalMs);
        });
    },

    clearError: () => set({ error: null }),

    reset: () => set({
        sessions: [],
        currentSession: null,
        isLoading: false,
        error: null,
    }),
}));
