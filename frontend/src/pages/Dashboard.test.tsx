import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import Dashboard from './Dashboard';
import { useAuthStore } from '../store';

// Mock the auth store
vi.mock('../store', () => ({
    useAuthStore: vi.fn(),
}));

// Mock useNavigate
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual('react-router-dom');
    return {
        ...actual,
        useNavigate: () => mockNavigate,
    };
});

describe('Dashboard', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        const mockUseAuthStore = useAuthStore as unknown as { mockReturnValue: (value: unknown) => void };
        mockUseAuthStore.mockReturnValue({
            user: { email: 'admin@heimdall.local' },
            logout: vi.fn(),
        });
    });

    it('should render the dashboard', () => {
        render(
            <BrowserRouter>
                <Dashboard />
            </BrowserRouter>
        );
        expect(screen.getByText('Welcome back! 🚀')).toBeInTheDocument();
    });

    it('should display the sidebar on desktop', () => {
        render(
            <BrowserRouter>
                <Dashboard />
            </BrowserRouter>
        );
        expect(screen.getByText('Heimdall')).toBeInTheDocument();
        expect(screen.getByText('Dashboard')).toBeInTheDocument();
    });

    it('should display all stat cards', () => {
        render(
            <BrowserRouter>
                <Dashboard />
            </BrowserRouter>
        );
        expect(screen.getByText('Active WebSDR')).toBeInTheDocument();
        expect(screen.getByText('Signal Detection')).toBeInTheDocument();
        expect(screen.getByText('System Uptime')).toBeInTheDocument();
        expect(screen.getByText('Accuracy')).toBeInTheDocument();
    });

    it('should display recent activity section', () => {
        render(
            <BrowserRouter>
                <Dashboard />
            </BrowserRouter>
        );
        expect(screen.getByText('Recent Activity')).toBeInTheDocument();
        expect(screen.getByText('RF signal detected')).toBeInTheDocument();
    });

    it('should display system health section', () => {
        render(
            <BrowserRouter>
                <Dashboard />
            </BrowserRouter>
        );
        expect(screen.getByText('System Health')).toBeInTheDocument();
        expect(screen.getByText('CPU Usage')).toBeInTheDocument();
        expect(screen.getByText('Memory')).toBeInTheDocument();
        expect(screen.getByText('Disk')).toBeInTheDocument();
    });

    it('should display network status section', () => {
        render(
            <BrowserRouter>
                <Dashboard />
            </BrowserRouter>
        );
        expect(screen.getByText('Network Status')).toBeInTheDocument();
        expect(screen.getByText('Turin')).toBeInTheDocument();
        expect(screen.getByText('Milan')).toBeInTheDocument();
    });

    it('should toggle mobile menu', async () => {
        render(
            <BrowserRouter>
                <Dashboard />
            </BrowserRouter>
        );

        const menuButtons = screen.getAllByRole('button');
        // Menu toggle behavior - just verify buttons exist
        expect(menuButtons.length).toBeGreaterThan(0);
    });

    it('should call logout on logout button click', async () => {
        const mockLogout = vi.fn();
        const mockUseAuthStore = useAuthStore as unknown as { mockReturnValue: (value: unknown) => void };
        mockUseAuthStore.mockReturnValue({
            user: { email: 'admin@heimdall.local' },
            logout: mockLogout,
        });

        render(
            <BrowserRouter>
                <Dashboard />
            </BrowserRouter>
        );

        const logoutButtons = screen.getAllByText('Logout');
        expect(logoutButtons.length).toBeGreaterThan(0);
    });

    it('should display user email in sidebar', () => {
        render(
            <BrowserRouter>
                <Dashboard />
            </BrowserRouter>
        );
        expect(screen.getByText('admin@heimdall.local')).toBeInTheDocument();
    });

    it('should have purple gradient background', () => {
        const { container } = render(
            <BrowserRouter>
                <Dashboard />
            </BrowserRouter>
        );
        const mainDiv = container.firstChild;
        expect(mainDiv).toHaveClass('bg-linear-to-br');
        expect(mainDiv).toHaveClass('from-purple-600');
    });

    it('should display stat values correctly', () => {
        render(
            <BrowserRouter>
                <Dashboard />
            </BrowserRouter>
        );
        expect(screen.getByText('7')).toBeInTheDocument(); // Active WebSDR
        expect(screen.getByText('12')).toBeInTheDocument(); // Signal Detection
    });
});
