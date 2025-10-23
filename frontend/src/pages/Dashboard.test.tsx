import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { SidebarProvider } from '@/components/ui/sidebar';
import Dashboard from './Dashboard';
import { useAuthStore } from '../store';

// Mock the auth store and dashboard stores
vi.mock('../store', async () => {
    const actual = await vi.importActual('../store');
    return {
        ...actual,
        useAuthStore: vi.fn(),
        useDashboardStore: vi.fn(),
        useWebSDRStore: vi.fn(),
    };
});

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
        
        // Import the mocked stores
        const { useAuthStore, useDashboardStore, useWebSDRStore } = require('../store');
        
        (useAuthStore as any).mockReturnValue({
            user: { email: 'admin@heimdall.local' },
            logout: vi.fn(),
        });

        (useDashboardStore as any).mockReturnValue({
            stats: {
                total_sessions: 42,
                completed_sessions: 38,
                total_measurements: 1024,
                success_rate: 90.5,
            },
            recentActivity: [
                { type: 'session_created', timestamp: new Date().toISOString() },
            ],
            systemHealth: { status: 'healthy', uptime_hours: 168 },
            fetchDashboardData: vi.fn(),
            lastUpdate: new Date().toISOString(),
        });

        (useWebSDRStore as any).mockReturnValue({
            websdrs: [
                { id: 'websdr1', name: 'WebSDR 1', frequency: 145.5, status: 'online' },
                { id: 'websdr2', name: 'WebSDR 2', frequency: 433.0, status: 'online' },
            ],
            healthStatus: { online_count: 2, total_count: 2 },
        });
    });

    const renderDashboard = () => {
        return render(
            <BrowserRouter>
                <SidebarProvider>
                    <Dashboard />
                </SidebarProvider>
            </BrowserRouter>
        );
    };

    it('should render the dashboard', () => {
        renderDashboard();
        // Get the main heading (h2) not sidebar item
        expect(screen.getByRole('heading', { level: 2, name: /Dashboard/i })).toBeInTheDocument();
    });

    it('should display the sidebar with Heimdall branding', () => {
        renderDashboard();
        // Get the main heading (h1) for Heimdall branding
        expect(screen.getByRole('heading', { level: 1, name: /Heimdall/i })).toBeInTheDocument();
    });

    it('should display sidebar navigation items', () => {
        renderDashboard();
        // Use getAllByText to get all instances and check they exist
        const dashboardItems = screen.getAllByText('Dashboard');
        expect(dashboardItems.length).toBeGreaterThan(0);
        expect(screen.getByText('Localization')).toBeInTheDocument();
        expect(screen.getByText('WebSDR')).toBeInTheDocument();
        expect(screen.getByText('Activity')).toBeInTheDocument();
        expect(screen.getByText('Analytics')).toBeInTheDocument();
    });

    it('should display all stat cards', () => {
        renderDashboard();
        expect(screen.getByText('Active WebSDR')).toBeInTheDocument();
        // Use getAllByText since "Signal Detection" appears twice (card label and activity)
        const signalDetection = screen.getAllByText('Signal Detection');
        expect(signalDetection.length).toBeGreaterThan(0);
        expect(screen.getByText('System Uptime')).toBeInTheDocument();
        expect(screen.getByText('Accuracy')).toBeInTheDocument();
    });

    it('should display stat values', () => {
        renderDashboard();
        expect(screen.getByText('7/7')).toBeInTheDocument();
        expect(screen.getByText('12')).toBeInTheDocument();
        expect(screen.getByText('99.8%')).toBeInTheDocument();
        expect(screen.getByText('95.2%')).toBeInTheDocument();
    });

    it('should display recent activity section', () => {
        renderDashboard();
        expect(screen.getByText('Recent Activity')).toBeInTheDocument();
        // Use getAllByText since "Signal Detection" appears in both card label and activity
        const activities = screen.getAllByText('Signal Detection');
        expect(activities.length).toBeGreaterThan(0);
        expect(screen.getByText('WebSDR Update')).toBeInTheDocument();
    });

    it('should display system health section', () => {
        renderDashboard();
        expect(screen.getByText('System Health')).toBeInTheDocument();
        expect(screen.getAllByText('CPU')).toBeDefined();
        expect(screen.getAllByText('Memory')).toBeDefined();
        expect(screen.getAllByText('Disk')).toBeDefined();
    });

    it('should display WebSDR network status', () => {
        renderDashboard();
        expect(screen.getByText('WebSDR Network Status')).toBeInTheDocument();
        expect(screen.getByText('Turin')).toBeInTheDocument();
        expect(screen.getByText('Milan')).toBeInTheDocument();
        expect(screen.getByText('Genoa')).toBeInTheDocument();
    });

    it('should display user email in sidebar', () => {
        renderDashboard();
        // Use getAllByText since email appears in both sidebar footer and dropdown menu
        const emailElements = screen.getAllByText('admin@heimdall.local');
        expect(emailElements.length).toBeGreaterThan(0);
    });

    it('should display logout button', () => {
        renderDashboard();
        const logoutButtons = screen.getAllByText('Logout');
        expect(logoutButtons.length).toBeGreaterThan(0);
    });

    it('should display welcome message with user name', () => {
        renderDashboard();
        expect(screen.getByText(/Welcome back, admin/)).toBeInTheDocument();
    });
});
