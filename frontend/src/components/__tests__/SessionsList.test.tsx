import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { SessionsList } from '../SessionsList';
import { mockSessions } from '@/__mocks__/apiFixtures';

// Mock the session store
const mockFetchSessions = vi.fn();
const mockUseSessionStore = vi.fn(() => ({
    sessions: mockSessions,
    isLoading: false,
    error: null,
    fetchSessions: mockFetchSessions,
}));

vi.mock('../../store/sessionStore', () => ({
    useSessionStore: () => mockUseSessionStore(),
}));

describe('SessionsList Component', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        vi.clearAllTimers();
    });

    const renderComponent = (props = {}) => {
        return render(
            <BrowserRouter>
                <SessionsList {...props} />
            </BrowserRouter>
        );
    };

    it('renders without crashing', () => {
        renderComponent();
        expect(screen.getByText(/Recording Sessions/i)).toBeInTheDocument();
    });

    it('displays session list with mock data', async () => {
        renderComponent();

        await waitFor(() => {
            expect(screen.getByText('Test Session 1')).toBeInTheDocument();
            expect(screen.getByText('Test Session 2')).toBeInTheDocument();
            expect(screen.getByText('Test Session 3')).toBeInTheDocument();
        });
    });

    it('calls fetchSessions on mount', () => {
        renderComponent();
        expect(mockFetchSessions).toHaveBeenCalled();
    });

    it('displays pending status badge correctly', async () => {
        renderComponent();

        await waitFor(() => {
            const pendingElements = screen.getAllByText(/pending/i);
            expect(pendingElements.length).toBeGreaterThan(0);
        });
    });

    it('displays completed status badge correctly', async () => {
        renderComponent();

        await waitFor(() => {
            const completedElements = screen.getAllByText(/completed/i);
            expect(completedElements.length).toBeGreaterThan(0);
        });
    });

    it('displays failed status badge correctly', async () => {
        renderComponent();

        await waitFor(() => {
            const failedElements = screen.getAllByText(/failed/i);
            expect(failedElements.length).toBeGreaterThan(0);
        });
    });

    it('shows loading state when isLoading is true', () => {
        mockUseSessionStore.mockReturnValue({
            sessions: [],
            isLoading: true,
            error: null,
            fetchSessions: mockFetchSessions,
        });

        renderComponent();
        expect(screen.getByText(/loading/i)).toBeInTheDocument();
    });

    it('shows empty state when no sessions exist', () => {
        mockUseSessionStore.mockReturnValue({
            sessions: [],
            isLoading: false,
            error: null,
            fetchSessions: mockFetchSessions,
        });

        renderComponent();
        expect(screen.getByText(/no recording sessions yet/i)).toBeInTheDocument();
    });

    it('calls onSessionSelect when session is clicked', async () => {
        const onSessionSelect = vi.fn();
        renderComponent({ onSessionSelect });

        await waitFor(() => {
            const sessionRow = screen.getByText('Test Session 1').closest('tr');
            if (sessionRow) {
                sessionRow.click();
                expect(onSessionSelect).toHaveBeenCalledWith(1);
            }
        });
    });

    it('displays session frequency correctly', async () => {
        renderComponent();

        await waitFor(() => {
            expect(screen.getByText(/145.5/)).toBeInTheDocument();
            expect(screen.getByText(/430.5/)).toBeInTheDocument();
        });
    });

    it('displays session duration correctly', async () => {
        renderComponent();

        await waitFor(() => {
            // Duration is formatted as "1m 0s" and "2m 0s"
            expect(screen.getByText(/1m 0s/)).toBeInTheDocument();
            expect(screen.getByText(/2m 0s/)).toBeInTheDocument();
        });
    });

    it('auto-refreshes sessions when autoRefresh is true', () => {
        vi.useFakeTimers();
        renderComponent({ autoRefresh: true });

        // Initial call
        expect(mockFetchSessions).toHaveBeenCalledTimes(1);

        // Fast-forward 5 seconds
        vi.advanceTimersByTime(5000);
        expect(mockFetchSessions).toHaveBeenCalledTimes(2);

        // Fast-forward another 5 seconds
        vi.advanceTimersByTime(5000);
        expect(mockFetchSessions).toHaveBeenCalledTimes(3);

        vi.useRealTimers();
    });

    it('does not auto-refresh when autoRefresh is false', () => {
        vi.useFakeTimers();
        renderComponent({ autoRefresh: false });

        // Initial call only
        expect(mockFetchSessions).toHaveBeenCalledTimes(1);

        // Fast-forward 10 seconds
        vi.advanceTimersByTime(10000);
        expect(mockFetchSessions).toHaveBeenCalledTimes(1);

        vi.useRealTimers();
    });

    it('displays empty state when error exists and no sessions loaded', () => {
        mockUseSessionStore.mockReturnValue({
            sessions: [],
            isLoading: false,
            error: 'Failed to fetch sessions',
            fetchSessions: mockFetchSessions,
        });

        renderComponent();
        // Component shows empty state even with error (no explicit error display in UI)
        expect(screen.getByText(/no recording sessions/i)).toBeInTheDocument();
    });
});
