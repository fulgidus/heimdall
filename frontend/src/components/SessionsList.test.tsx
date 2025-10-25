import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { SessionsList } from './SessionsList';
import { useSessionStore } from '../store/sessionStore';

// Mock the session store
vi.mock('../store/sessionStore', () => ({
    useSessionStore: vi.fn(),
}));

// Mock timers
vi.useFakeTimers();

describe('SessionsList', () => {
    let mockFetchSessions: ReturnType<typeof vi.fn>;

    const mockSessions = [
        {
            id: 1,
            session_name: 'Session 1',
            status: 'completed',
            approval_status: 'approved',
            frequency_mhz: 145.5,
            duration_seconds: 30,
            created_at: '2024-01-01T10:00:00Z',
            measurements_count: 7,
        },
        {
            id: 2,
            session_name: 'Session 2',
            status: 'processing',
            approval_status: 'pending',
            frequency_mhz: 433.0,
            duration_seconds: 60,
            created_at: '2024-01-01T11:00:00Z',
            measurements_count: 3,
        },
        {
            id: 3,
            session_name: 'Session 3',
            status: 'pending',
            approval_status: 'pending',
            frequency_mhz: 145.8,
            duration_seconds: 45,
            created_at: '2024-01-01T12:00:00Z',
            measurements_count: 0,
        },
    ];

    beforeEach(() => {
        mockFetchSessions = vi.fn();

        // Default mock implementation
        vi.mocked(useSessionStore).mockReturnValue({
            sessions: [],
            isLoading: false,
            fetchSessions: mockFetchSessions,
        } as any);
    });

    afterEach(() => {
        vi.clearAllMocks();
        vi.clearAllTimers();
    });

    it('should fetch sessions on mount', () => {
        render(<SessionsList />);

        expect(mockFetchSessions).toHaveBeenCalled();
    });

    it('should display loading state', () => {
        vi.mocked(useSessionStore).mockReturnValue({
            sessions: [],
            isLoading: true,
            fetchSessions: mockFetchSessions,
        } as any);

        render(<SessionsList />);

        expect(screen.getByText(/Loading/i)).toBeInTheDocument();
    });

    it('should display sessions list', () => {
        vi.mocked(useSessionStore).mockReturnValue({
            sessions: mockSessions,
            isLoading: false,
            fetchSessions: mockFetchSessions,
        } as any);

        render(<SessionsList />);

        expect(screen.getByText('Session 1')).toBeInTheDocument();
        expect(screen.getByText('Session 2')).toBeInTheDocument();
        expect(screen.getByText('Session 3')).toBeInTheDocument();
    });

    it('should display session details correctly', () => {
        vi.mocked(useSessionStore).mockReturnValue({
            sessions: [mockSessions[0]],
            isLoading: false,
            fetchSessions: mockFetchSessions,
        } as any);

        render(<SessionsList />);

        expect(screen.getByText('Session 1')).toBeInTheDocument();
        // Frequency is displayed as "145.500 MHz" (with decimal formatting)
        expect(screen.getByText(/145\.500/)).toBeInTheDocument();
        expect(screen.getByText(/MHz/)).toBeInTheDocument();
        expect(screen.getByText('30s')).toBeInTheDocument();
    });

    it('should show correct status badge for completed session', () => {
        vi.mocked(useSessionStore).mockReturnValue({
            sessions: [mockSessions[0]],
            isLoading: false,
            fetchSessions: mockFetchSessions,
        } as any);

        render(<SessionsList />);

        const statusBadge = screen.getByText('COMPLETED');
        expect(statusBadge).toBeInTheDocument();
        expect(statusBadge).toHaveClass('text-green-400');
    });

    it('should show correct status badge for processing session', () => {
        vi.mocked(useSessionStore).mockReturnValue({
            sessions: [mockSessions[1]],
            isLoading: false,
            fetchSessions: mockFetchSessions,
        } as any);

        render(<SessionsList />);

        const statusBadge = screen.getByText('PROCESSING');
        expect(statusBadge).toBeInTheDocument();
        expect(statusBadge).toHaveClass('text-blue-400');
    });

    it('should show correct status badge for pending session', () => {
        vi.mocked(useSessionStore).mockReturnValue({
            sessions: [mockSessions[2]],
            isLoading: false,
            fetchSessions: mockFetchSessions,
        } as any);

        render(<SessionsList />);

        const statusBadge = screen.getByText('PENDING');
        expect(statusBadge).toBeInTheDocument();
        expect(statusBadge).toHaveClass('text-yellow-400');
    });

    it('should call onSessionSelect when clicking view button', () => {
        const onSessionSelect = vi.fn();

        vi.mocked(useSessionStore).mockReturnValue({
            sessions: [mockSessions[0]],
            isLoading: false,
            fetchSessions: mockFetchSessions,
        } as any);

        render(<SessionsList onSessionSelect={onSessionSelect} />);

        const viewButton = screen.getByRole('button', { name: /view/i });
        fireEvent.click(viewButton);

        expect(onSessionSelect).toHaveBeenCalledWith(1);
    });

    it.skip('should auto-refresh sessions every 5 seconds by default', async () => {
        vi.mocked(useSessionStore).mockReturnValue({
            sessions: mockSessions,
            isLoading: false,
            fetchSessions: mockFetchSessions,
        } as any);

        vi.useFakeTimers();

        const { unmount } = render(<SessionsList />);

        // Clear initial mount call
        mockFetchSessions.mockClear();

        // Advance time and run timers
        act(() => {
            vi.advanceTimersByTime(5000);
        });

        // Should have called fetch after 5 seconds
        expect(mockFetchSessions).toHaveBeenCalledTimes(1);

        // Advance another 5 seconds
        act(() => {
            vi.advanceTimersByTime(5000);
        });

        // Should have called fetch again
        expect(mockFetchSessions).toHaveBeenCalledTimes(2);
        
        unmount();
        vi.useRealTimers();
    });

    it('should not auto-refresh when autoRefresh is false', async () => {
        vi.mocked(useSessionStore).mockReturnValue({
            sessions: mockSessions,
            isLoading: false,
            fetchSessions: mockFetchSessions,
        } as any);

        render(<SessionsList autoRefresh={false} />);

        // Initial call on mount
        expect(mockFetchSessions).toHaveBeenCalledTimes(1);

        // Advance time by 5 seconds
        vi.advanceTimersByTime(5000);

        // Should still be only 1 call (no auto-refresh)
        expect(mockFetchSessions).toHaveBeenCalledTimes(1);
    });

    it('should clear interval on unmount', () => {
        const clearIntervalSpy = vi.spyOn(global, 'clearInterval');

        vi.mocked(useSessionStore).mockReturnValue({
            sessions: mockSessions,
            isLoading: false,
            fetchSessions: mockFetchSessions,
        } as any);

        const { unmount } = render(<SessionsList />);

        unmount();

        expect(clearIntervalSpy).toHaveBeenCalled();
    });

    it('should display empty state when no sessions', () => {
        vi.mocked(useSessionStore).mockReturnValue({
            sessions: [],
            isLoading: false,
            fetchSessions: mockFetchSessions,
        } as any);

        render(<SessionsList />);

        expect(screen.getByText(/No recording sessions yet/i)).toBeInTheDocument();
    });

    it('should format dates correctly', () => {
        vi.mocked(useSessionStore).mockReturnValue({
            sessions: [mockSessions[0]],
            isLoading: false,
            fetchSessions: mockFetchSessions,
        } as any);

        render(<SessionsList />);

        // Date should be formatted using toLocaleString()
        const dateElement = screen.getByText(/1\/1\/2024/);
        expect(dateElement).toBeInTheDocument();
    });

    it('should handle null date gracefully', () => {
        const sessionWithNullDate = {
            ...mockSessions[0],
            created_at: null,
        };

        vi.mocked(useSessionStore).mockReturnValue({
            sessions: [sessionWithNullDate],
            isLoading: false,
            fetchSessions: mockFetchSessions,
        } as any);

        render(<SessionsList />);

        expect(screen.getByText('N/A')).toBeInTheDocument();
    });

    it('should handle failed status correctly', () => {
        const failedSession = {
            ...mockSessions[0],
            status: 'failed',
        };

        vi.mocked(useSessionStore).mockReturnValue({
            sessions: [failedSession],
            isLoading: false,
            fetchSessions: mockFetchSessions,
        } as any);

        render(<SessionsList />);

        const statusBadge = screen.getByText('FAILED');
        expect(statusBadge).toBeInTheDocument();
        expect(statusBadge).toHaveClass('text-red-400');
    });

    it('should show spinning icon for processing status', () => {
        vi.mocked(useSessionStore).mockReturnValue({
            sessions: [mockSessions[1]],
            isLoading: false,
            fetchSessions: mockFetchSessions,
        } as any);

        render(<SessionsList />);

        // Find the processing status element and check if it has the animate-spin class
        const statusElement = screen.getByText('PROCESSING').closest('span');
        const spinningIcon = statusElement?.querySelector('.animate-spin');
        expect(spinningIcon).toBeInTheDocument();
    });
});
